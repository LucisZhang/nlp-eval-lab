// Triage Router Lab — in-browser live inference engines.
// Vanilla ES module, zero dependencies, no external URLs (ever).
//
// Two engines, both exposing the same shape:
//   { predict(text) -> Promise<{label, p_max, probs, latency_ms}>, meta: {...} }
//
//   loadTierA(url)                      TF-IDF + multinomial LogReg + isotonic, pure JS.
//   loadTierB2(baseUrl, {onProgress})   DistilBERT int8 ONNX via vendored onnxruntime-web.
//
// DESIGN RULE FOR THIS FILE: exactness over cleverness. Every place where a
// JavaScript primitive could disagree with the Python/Rust reference carries a
// WHY comment naming the reference function that was read. The reference
// implementations consulted (versions pinned in this repo's uv lockfile):
//
//   sklearn 1.9.0
//     feature_extraction/text.py :: _preprocess, build_tokenizer, _word_ngrams,
//                                   _char_wb_ngrams, TfidfTransformer.transform
//     preprocessing/_data.py     :: normalize  (-> inplace_csr_row_normalize_l2)
//     linear_model/_logistic.py  :: LogisticRegression.predict_proba
//     utils/extmath.py           :: softmax
//     utils/_response.py         :: _get_response_values
//     utils/validation.py        :: _check_response_method
//     calibration.py             :: _CalibratedClassifier.predict_proba
//     isotonic.py                :: IsotonicRegression._build_f, _transform
//   scipy 1.18.0
//     interpolate/_interpolate.py :: interp1d.__init__ (delegation decision),
//                                    _call_linear_np -> numpy.interp
//   numpy 2.5.1
//     compiled_base.c :: arr_interp (boundary + slope conventions)
//   huggingface tokenizers (spec taken from demo/live/tier_b2/tokenizer.json)
//     normalizers/bert.rs        :: BertNormalizer (clean_text / chinese chars /
//                                   strip_accents / lowercase, in that order)
//     pre_tokenizers/bert.rs     :: BertPreTokenizer (is_bert_punc)
//     models/wordpiece.rs        :: WordPiece::tokenize

// ----------------------------------------------------------------------------
// shared helpers
// ----------------------------------------------------------------------------

// base64 -> Float32Array. The payload is documented as a little-endian
// Float32Array dump. WHY this is safe: every platform that runs a browser is
// little-endian (x86-64, arm64, wasm32 is defined little-endian), so a
// Float32Array view over the decoded bytes reproduces the producer's values
// bit-for-bit. There is no DataView loop here on purpose — it would be ~20x
// slower on a 10 MB coefficient blob for a portability case that cannot occur.
export function b64ToFloat32Array(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  if (bytes.byteLength % 4 !== 0) {
    throw new Error(`base64 float32 payload is not a multiple of 4 bytes (${bytes.byteLength})`);
  }
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

// sklearn.utils.extmath.softmax: subtract the row max, exp, divide by the sum.
// The max-subtraction is not just for overflow safety — it changes the rounding,
// so we replicate it rather than doing the mathematically equivalent direct form.
export function softmax(scores) {
  let maxProb = -Infinity;
  for (let i = 0; i < scores.length; i++) if (scores[i] > maxProb) maxProb = scores[i];
  const out = new Array(scores.length);
  let sumProb = 0;
  for (let i = 0; i < scores.length; i++) {
    // sklearn does `X -= max_prob` then `np.exp(X, out=X)`: subtraction first,
    // in place, so exp() sees the already-rounded difference. Same here.
    const shifted = scores[i] - maxProb;
    out[i] = Math.exp(shifted);
    sumProb += out[i];
  }
  for (let i = 0; i < out.length; i++) out[i] /= sumProb;
  return out;
}

// np.argmax tie-break: the FIRST maximal index wins (strict >, never >=).
function argmax(values) {
  let best = 0;
  for (let i = 1; i < values.length; i++) if (values[i] > values[best]) best = i;
  return best;
}

function nowMs() {
  return (typeof performance !== "undefined" && performance.now) ? performance.now() : Date.now();
}

async function fetchJSONWithSize(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`fetch ${url} failed: HTTP ${res.status}`);
  // arrayBuffer -> TextDecoder so we get an exact byte count for meta.size_bytes
  // without a second pass over the (large) payload.
  const buf = await res.arrayBuffer();
  const text = new TextDecoder("utf-8").decode(buf);
  return { json: JSON.parse(text), sizeBytes: buf.byteLength };
}

// ============================================================================
// TIER A — TF-IDF (word 1-2 + char_wb 3-5) -> multinomial LogReg -> isotonic
// ============================================================================

// Python's `str.isspace()` / re `\s` (str mode) whitespace set, enumerated by
// scanning the full codepoint space with CPython. WHY not JS `\s`: the two sets
// are NOT equal. JS `\s` additionally contains U+FEFF (ZWNBSP) and is missing
// U+001C-U+001F (file/group/record/unit separators) and U+0085 (NEL), all four
// of which Python treats as whitespace. Using JS `\s` here would silently glue
// or split char_wb "words" differently from the training-time vectorizer.
const PY_WHITESPACE_RUN = /[\t\n\v\f\r\x1c-\x1f \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]+/u;

// sklearn's default token_pattern is r"(?u)\b\w\w+\b" applied with re.findall,
// which yields every MAXIMAL run of `\w` whose length is >= 2 (runs of length 1
// are dropped entirely; findall never returns overlapping or partial runs
// because `\w\w+` is greedy and the trailing `\b` forces the run's end).
//
// Python's `\w` (str mode) is `Py_UNICODE_ISALNUM(ch) || ch == '_'`, i.e.
// L* + Nd + Nl + No + '_'. That is exactly JS `[\p{L}\p{N}_]`. Verified by
// enumerating all 0x110000 codepoints in both runtimes: zero codepoints matched
// by Python-\w are missed by [\p{L}\p{N}_].
//
// KNOWN RESIDUAL DIVERGENCE (accepted, documented): the two runtimes ship
// different Unicode Character Database versions, so ~9.6k codepoints that a
// newer UCD assigns to L/N are `\p{L}`/`\p{N}` in the browser but were
// unassigned (and therefore non-word) for the training-time Python. All of them
// are in recently-added blocks (Garay, Sunuwar, Todhri, new CJK, ...). A
// narrative containing one would tokenize differently, but the affected term
// could not be in the frozen vocabulary anyway, so the only possible effect is a
// boundary shift in an adjacent term. CFPB narratives are ASCII + curly quotes
// + "XXXX" redactions, so this is unreachable in practice.
const WORD_TOKEN_RE = /[\p{L}\p{N}_]{2,}/gu;

function tokenizeWords(loweredText) {
  return loweredText.match(WORD_TOKEN_RE) || [];
}

function incr(counts, key) {
  counts.set(key, (counts.get(key) || 0) + 1);
}

// sklearn VectorizerMixin._word_ngrams with ngram_range = (min_n, max_n).
// Unigrams pass through untouched; n-grams for n >= 2 are " ".join of a sliding
// window. The `min(max_n + 1, n_original_tokens + 1)` bound means a document
// with fewer tokens than n contributes no n-grams of that order.
function wordNgramCounts(tokens, minN, maxN, counts) {
  let start = minN;
  if (minN === 1) {
    for (let i = 0; i < tokens.length; i++) incr(counts, tokens[i]);
    start = 2;
  }
  const nTokens = tokens.length;
  for (let n = start; n < Math.min(maxN + 1, nTokens + 1); n++) {
    for (let i = 0; i + n <= nTokens; i++) {
      incr(counts, tokens.slice(i, i + n).join(" "));
    }
  }
}

// Split on whitespace the way Python's `str.split()` (no argument) does:
// maximal runs of whitespace are separators and leading/trailing runs produce no
// empty fields.
//
// sklearn's _char_wb_ngrams first does `self._white_spaces.sub(" ", doc)` with
// `_white_spaces = re.compile(r"\s\s+")`, then `doc.split()`. That substitution
// is provably a no-op for the resulting word list: it only ever replaces a run
// of >= 2 whitespace chars with a single whitespace char, and split() treats any
// run of >= 1 whitespace as one separator. We therefore skip the substitution
// (avoiding a JS-`\s`-vs-Python-`\s` mismatch, see PY_WHITESPACE_RUN) and split
// directly with Python's whitespace set.
function splitWhitespacePython(text) {
  const parts = text.split(PY_WHITESPACE_RUN);
  const out = [];
  for (let i = 0; i < parts.length; i++) if (parts[i].length > 0) out.push(parts[i]);
  return out;
}

const SURROGATE_RE = /[\uD800-\uDFFF]/;

// sklearn VectorizerMixin._char_wb_ngrams, transcribed literally:
//
//   for w in text.split():
//       w = " " + w + " "
//       for n in range(min_n, max_n + 1):
//           offset = 0
//           emit w[0:n]                       # may be SHORTER than n
//           while offset + n < len(w): offset += 1; emit w[offset:offset+n]
//           if offset == 0: break             # short word counted once only
//
// Two behaviours that are easy to get wrong and are reproduced exactly:
//   (1) when the padded word is shorter than n, `w[0:n]` still emits the whole
//       padded word (a SHORT n-gram enters the vocabulary), and
//   (2) the `if offset == 0: break` then abandons all larger n for that word, so
//       a short word contributes that padded form exactly once, not once per n.
//       Note `len(w) == n` also leaves offset at 0 and triggers the break.
//
// Python indexes str by CODEPOINT; JS indexes by UTF-16 code unit. For any word
// containing an astral character (emoji, some CJK ext) naive .slice() would cut
// surrogate pairs and produce n-grams Python never produces. We take the
// codepoint path only when a surrogate is present, keeping the common ASCII path
// allocation-free.
function charWbNgramCounts(loweredText, minN, maxN, counts) {
  const words = splitWhitespacePython(loweredText);
  for (let wi = 0; wi < words.length; wi++) {
    const w = " " + words[wi] + " ";
    const astral = SURROGATE_RE.test(w);
    const cps = astral ? Array.from(w) : null;
    const wLen = astral ? cps.length : w.length;
    const slice = astral
      ? (a, b) => cps.slice(a, b).join("")
      : (a, b) => w.slice(a, b);
    for (let n = minN; n <= maxN; n++) {
      let offset = 0;
      incr(counts, slice(0, n));
      while (offset + n < wLen) {
        offset += 1;
        incr(counts, slice(offset, offset + n));
      }
      if (offset === 0) break;
    }
  }
}

function buildVocabMap(vocab) {
  const map = new Map();
  for (let i = 0; i < vocab.length; i++) map.set(vocab[i], i);
  return map;
}

// One TF-IDF branch (word or char_wb): raw counts -> sublinear tf -> * idf ->
// L2 row normalisation, exactly as TfidfTransformer.transform does it:
//
//   if sublinear_tf: np.log(X.data, X.data); X.data += 1.0
//   X.data *= self.idf_[X.indices]
//   X = normalize(X, norm="l2")          # -> inplace_csr_row_normalize_l2
//
// inplace_csr_row_normalize_l2 (sparsefuncs_fast.pyx) accumulates
// `sum_ += v*v` in CSR index order, takes sqrt, and DIVIDES each value by it
// (not multiply-by-reciprocal). CountVectorizer.transform calls X.sort_indices()
// so CSR index order is ascending column order — hence the sort below, which
// fixes the float summation order to match Python's.
//
// Returns entries as {cols: Int32Array-like array, vals: number[]}, ascending.
function branchVector(counts, vocabMap, idf, sublinearTf) {
  const cols = [];
  const vals = [];
  for (const [term, count] of counts) {
    const col = vocabMap.get(term);
    if (col === undefined) continue; // out-of-vocabulary terms are dropped
    cols.push(col);
    vals.push(count);
  }
  // sort (col, val) pairs by column ascending
  const order = cols.map((_, i) => i);
  order.sort((a, b) => cols[a] - cols[b]);
  const sCols = new Array(order.length);
  const sVals = new Array(order.length);
  for (let k = 0; k < order.length; k++) {
    const i = order[k];
    sCols[k] = cols[i];
    let v = sVals[k] = vals[i];
    // sublinear tf: log() then += 1.0, in that order (matches the two numpy ops)
    if (sublinearTf) v = Math.log(v) + 1.0;
    // idf_ is float64 in Python but float32 in the exported payload; the widening
    // to double here is exact, the quantisation happened at export time.
    v *= idf[sCols[k]];
    sVals[k] = v;
  }
  let sumSq = 0;
  for (let k = 0; k < sVals.length; k++) sumSq += sVals[k] * sVals[k];
  if (sumSq !== 0.0) {
    const norm = Math.sqrt(sumSq);
    for (let k = 0; k < sVals.length; k++) sVals[k] /= norm;
  }
  return { cols: sCols, vals: sVals };
}

// numpy.interp, transcribed from arr_interp in numpy/_core/src/multiarray/
// compiled_base.c. scipy's interp1d delegates to np.interp whenever x and y are
// 1-D float64 and fill_value is not "extrapolate" — which is exactly the case
// sklearn's IsotonicRegression._build_f constructs, so THIS is the code path
// sklearn actually takes (the de-Boor-style two-weight formula in
// scipy's _call_linear is dead code here and would round differently).
//
// numpy's exact conventions, all reproduced:
//   x < xp[0]            -> fp[0]        (left fill)
//   x > xp[-1]           -> fp[-1]       (right fill)
//   j == len(xp) - 1     -> fp[-1]
//   xp[j] == x           -> fp[j]        ("avoid potential non-finite interpolation")
//   otherwise            -> slope * (x - xp[j]) + fp[j],
//                           slope = (fp[j+1] - fp[j]) / (xp[j+1] - xp[j])
// Note the slope form, NOT the (x-x0)/(x1-x0)*y1 + (x1-x)/(x1-x0)*y0 form.
function npInterp(x, xp, fp) {
  const n = xp.length;
  if (n === 0) return NaN;
  if (n === 1) return fp[0];
  if (x < xp[0]) return fp[0];
  if (x > xp[n - 1]) return fp[n - 1];
  // find j with xp[j] <= x < xp[j+1]
  let lo = 0;
  let hi = n - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (xp[mid] <= x) lo = mid; else hi = mid;
  }
  if (lo === n - 1) return fp[n - 1];
  if (xp[lo] === x) return fp[lo];
  const slope = (fp[lo + 1] - fp[lo]) / (xp[lo + 1] - xp[lo]);
  return slope * (x - xp[lo]) + fp[lo];
}

// IsotonicRegression(out_of_bounds="clip").predict, from isotonic.py::_transform:
//   T = np.clip(T, self.X_min_, self.X_max_)   then   res = self.f_(T)
// X_min_/X_max_ are min/max of the (sorted, deduplicated, trimmed) thresholds.
// _build_f also special-cases len(y) == 1 as a constant predictor.
function isotonicPredict(cal, value) {
  const xs = cal.x;
  const ys = cal.y;
  if (!xs || xs.length === 0) return NaN;
  if (ys.length === 1) return ys[0];
  const xMin = xs[0];
  const xMax = xs[xs.length - 1];
  let t = value;
  if (t < xMin) t = xMin;
  if (t > xMax) t = xMax;
  return npInterp(t, xs, ys);
}

// Which space do the isotonic calibrators live in?
//
// This is the single most load-bearing ambiguity in the Tier A port, so it is
// resolved explicitly and reported in meta rather than assumed.
//
// sklearn's _CalibratedClassifier.predict_proba calls
//   _get_response_values(self.estimator, X, response_method=["decision_function",
//                                                            "predict_proba"])
// and _check_response_method returns the FIRST method in that list that the
// estimator implements. A Pipeline ending in LogisticRegression implements
// decision_function, and FrozenEstimator delegates attribute lookups — so with
// this repo's `CalibratedClassifierCV(FrozenEstimator(pipe), method="isotonic")`
// the calibrators are fitted on RAW DECISION MARGINS, not on softmax
// probabilities. Confirmed empirically against sklearn 1.9.0: the fitted
// X_thresholds_ spanned roughly [-1.46, +4.25], far outside [0, 1].
//
// The exporter may nevertheless have chosen to export a softmax-space
// calibration. Resolution order:
//   1. an explicit field  (calibration.input_space / calibration.input)
//   2. keywords in calibration.semantics
//   3. a range heuristic  (any threshold outside [0,1] => decision_function)
// and the choice plus its source land in meta so the agreement harness can
// display them.
function resolveCalibrationInput(calibration, override) {
  if (override) return { input: override, source: "caller override" };
  if (!calibration) return { input: "softmax", source: "no calibration block" };

  const explicit = calibration.input_space || calibration.input;
  if (typeof explicit === "string") {
    const v = explicit.toLowerCase();
    if (v.includes("decision")) return { input: "decision_function", source: "calibration.input_space" };
    if (v.includes("softmax") || v.includes("proba")) return { input: "softmax", source: "calibration.input_space" };
  }

  const sem = typeof calibration.semantics === "string" ? calibration.semantics.toLowerCase() : "";
  if (sem.includes("decision_function") || sem.includes("decision function") || sem.includes("margin")) {
    return { input: "decision_function", source: "calibration.semantics" };
  }
  if (sem.includes("softmax") || sem.includes("predict_proba")) {
    return { input: "softmax", source: "calibration.semantics" };
  }

  const perClass = calibration.per_class || [];
  for (let c = 0; c < perClass.length; c++) {
    const xs = perClass[c].x || [];
    for (let i = 0; i < xs.length; i++) {
      if (xs[i] < 0 || xs[i] > 1) {
        return { input: "decision_function", source: "heuristic: threshold outside [0,1]" };
      }
    }
  }
  return { input: "softmax", source: "heuristic: all thresholds within [0,1]" };
}

/**
 * Load the pure-JS Tier A engine.
 *
 * @param {string} url  URL of tier_a_live.json (schema_version "live-v1")
 * @param {{calibrationInput?: "decision_function"|"softmax"}} [options]
 * @returns {Promise<{predict: (text: string) => Promise<object>, meta: object}>}
 */
export async function loadTierA(url, options = {}) {
  const t0 = nowMs();
  const { json: payload, sizeBytes } = await fetchJSONWithSize(url);

  const classLabels = payload.class_labels || [];
  const nClasses = classLabels.length;
  if (!nClasses) throw new Error("tier_a_live.json: class_labels is empty or missing");

  const word = payload.word || null;
  const char = payload.char || null;
  if (!word && !char) throw new Error("tier_a_live.json: neither a word nor a char branch is present");

  const branches = [];
  let offset = 0;
  for (const [name, spec] of [["word", word], ["char", char]]) {
    if (!spec) continue;
    const idf = b64ToFloat32Array(spec.idf_b64);
    if (idf.length !== spec.vocab.length) {
      throw new Error(`tier_a_live.json: ${name} idf length ${idf.length} != vocab length ${spec.vocab.length}`);
    }
    branches.push({
      name,
      vocabMap: buildVocabMap(spec.vocab),
      idf,
      ngramRange: spec.ngram_range || (name === "word" ? [1, 2] : [3, 5]),
      // per-branch override wins, else the document-level flag
      sublinearTf: spec.sublinear_tf !== undefined ? !!spec.sublinear_tf : !!payload.sublinear_tf,
      offset,
      size: spec.vocab.length,
    });
    offset += spec.vocab.length;
  }
  const nFeatures = offset;

  const coef = b64ToFloat32Array(payload.coef_b64);
  if (coef.length !== nClasses * nFeatures) {
    throw new Error(
      `tier_a_live.json: coef has ${coef.length} values, expected n_classes*n_features = ${nClasses * nFeatures}`,
    );
  }
  const intercept = payload.intercept || new Array(nClasses).fill(0);
  if (intercept.length !== nClasses) {
    throw new Error(`tier_a_live.json: intercept has ${intercept.length} values, expected ${nClasses}`);
  }

  const calibration = payload.calibration || null;
  const calMethod = calibration ? (calibration.method || "isotonic") : "none";
  const { input: calInput, source: calInputSource } = resolveCalibrationInput(calibration, options.calibrationInput);
  const calPerClass = calibration && Array.isArray(calibration.per_class) ? calibration.per_class : null;
  if (calPerClass && calPerClass.length !== nClasses) {
    throw new Error(`tier_a_live.json: calibration.per_class has ${calPerClass.length} entries, expected ${nClasses}`);
  }
  if (calibration && calMethod !== "isotonic") {
    // Fail loud rather than silently returning uncalibrated numbers that would
    // quietly disagree with the frozen records.
    throw new Error(`tier_a_live.json: unsupported calibration.method ${JSON.stringify(calMethod)}`);
  }
  // npInterp assumes ascending, deduplicated thresholds — which is exactly what
  // IsotonicRegression.X_thresholds_ guarantees (_build_y sorts with lexsort,
  // deduplicates via _make_unique, then trims). Validate rather than trust: an
  // exporter that reordered or transposed the arrays would otherwise produce
  // plausible-looking but wrong probabilities instead of an error.
  if (calPerClass) {
    calPerClass.forEach((cal, c) => {
      const xs = cal.x || [];
      const ys = cal.y || [];
      if (xs.length !== ys.length || xs.length === 0) {
        throw new Error(`tier_a_live.json: calibration.per_class[${c}] has x/y lengths ${xs.length}/${ys.length}`);
      }
      for (let i = 1; i < xs.length; i++) {
        if (!(xs[i] > xs[i - 1])) {
          throw new Error(
            `tier_a_live.json: calibration.per_class[${c}].x is not strictly ascending at index ${i} ` +
            `(${xs[i - 1]} -> ${xs[i]})`,
          );
        }
      }
    });
  }

  function featurize(text) {
    // sklearn's preprocessor is `_preprocess(doc, accent_function=None, lower=True)`
    // i.e. plain str.lower() over the WHOLE document. CPython's str.lower()
    // implements the Final_Sigma context rule (verified: "ΟΔΟΣ".lower() ==
    // "οδος"), and so does String.prototype.toLowerCase() on a whole string —
    // so whole-string toLowerCase() is the matching primitive here.
    // (Contrast with the Tier B2 tokenizer below, which must lowercase
    // per-character because HF's Rust uses char::to_lowercase.)
    // Per-codepoint case mappings were verified identical across the two
    // runtimes for every codepoint CPython lowercases.
    const lowered = text.toLowerCase();
    const entries = [];
    for (const branch of branches) {
      const counts = new Map();
      if (branch.name === "word") {
        wordNgramCounts(tokenizeWords(lowered), branch.ngramRange[0], branch.ngramRange[1], counts);
      } else {
        charWbNgramCounts(lowered, branch.ngramRange[0], branch.ngramRange[1], counts);
      }
      const vec = branchVector(counts, branch.vocabMap, branch.idf, branch.sublinearTf);
      // FeatureUnion hstacks the branches in declaration order (word, then char),
      // so global column = branch offset + local column. Branch entries are
      // already ascending, and offsets are disjoint and increasing, so the
      // concatenation stays globally ascending — matching scipy's hstack output.
      for (let k = 0; k < vec.cols.length; k++) {
        entries.push([branch.offset + vec.cols[k], vec.vals[k]]);
      }
    }
    return entries;
  }

  function decisionFunction(entries) {
    // LinearClassifierMixin.decision_function = safe_sparse_dot(X, coef_.T) + intercept_.
    // scipy's csr_matvecs accumulates one output column at a time over the row's
    // nonzeros in ascending index order, and the intercept is added afterwards by
    // a separate numpy op — replicated here (products first, intercept last).
    //
    // NOTE: the features are sparse (a few thousand nonzeros) but coef is dense,
    // so we iterate nonzeros and stride into the dense row-major coef. No
    // n_features-wide dense vector is ever allocated.
    const scores = new Array(nClasses);
    for (let c = 0; c < nClasses; c++) {
      const base = c * nFeatures;
      let acc = 0;
      for (let k = 0; k < entries.length; k++) acc += entries[k][1] * coef[base + entries[k][0]];
      scores[c] = acc + intercept[c];
    }
    return scores;
  }

  function calibrate(scores) {
    if (!calPerClass) return softmax(scores);
    // _CalibratedClassifier.predict_proba: each per-class calibrator is applied
    // one-vs-all to its own column of the response, then rows are renormalised.
    const calInputs = calInput === "softmax" ? softmax(scores) : scores;
    const proba = new Array(nClasses);
    let denominator = 0;
    for (let c = 0; c < nClasses; c++) {
      proba[c] = isotonicPredict(calPerClass[c], calInputs[c]);
      denominator += proba[c];
    }
    if (denominator !== 0) {
      for (let c = 0; c < nClasses; c++) proba[c] /= denominator;
    } else {
      // sklearn: np.divide(proba, denominator, out=uniform_proba, where=denominator != 0)
      // — rows whose calibrators all returned exactly zero keep the uniform
      // distribution 1/n_classes rather than becoming NaN.
      for (let c = 0; c < nClasses; c++) proba[c] = 1 / nClasses;
    }
    // sklearn's final guard: probabilities that minimally exceed 1.0 are snapped.
    for (let c = 0; c < nClasses; c++) {
      if (proba[c] > 1.0 && proba[c] <= 1.0 + 1e-5) proba[c] = 1.0;
    }
    return proba;
  }

  function predictSync(text) {
    const start = nowMs();
    const proba = calibrate(decisionFunction(featurize(text || "")));
    const best = argmax(proba);
    const probs = {};
    for (let c = 0; c < nClasses; c++) probs[classLabels[c]] = proba[c];
    return {
      label: classLabels[best],
      p_max: proba[best],
      probs,
      latency_ms: nowMs() - start,
    };
  }

  return {
    // Sync internally (no I/O), Promise-returning for API symmetry with Tier B2.
    predict: (text) => Promise.resolve(predictSync(text)),
    meta: {
      tier: "A",
      engine: "tfidf+logreg+isotonic (pure JS)",
      schema_version: payload.schema_version,
      provenance: payload.provenance || null,
      class_labels: classLabels,
      n_features: nFeatures,
      n_features_word: word ? word.vocab.length : 0,
      n_features_char: char ? char.vocab.length : 0,
      sublinear_tf: !!payload.sublinear_tf,
      calibration_method: calMethod,
      calibration_input: calPerClass ? calInput : "none",
      calibration_input_source: calPerClass ? calInputSource : "no per-class calibrators",
      size_bytes: sizeBytes,
      load_ms: nowMs() - t0,
      source_url: url,
    },
  };
}

// ============================================================================
// TIER B2 — DistilBERT int8 ONNX, WordPiece tokenizer in JS
// ============================================================================

// --- BertNormalizer -------------------------------------------------------
//
// tokenizer.json declares:
//   {clean_text: true, handle_chinese_chars: true, strip_accents: null, lowercase: true}
// The Rust normalize() runs, in this order:
//   1. clean_text, 2. handle_chinese_chars,
//   3. strip_accents = self.strip_accents.unwrap_or(self.lowercase)  -> TRUE here
//      (this is what "strip_accents: null means strip when lowercasing" means),
//   4. lowercase.
// Note accents are stripped BEFORE lowercasing, which matters for e.g. U+0130.

// Rust's is_control(c): false for \t \n \r, otherwise `c.is_other()` — the
// unicode_categories "Other" group, i.e. Cc | Cf | Co | Cs | Cn. JS supports all
// five as property escapes, so this is a direct transcription. Cn (unassigned)
// is the version-sensitive one: a codepoint unassigned in the browser's UCD but
// assigned in the Rust crate's UCD (or vice versa) would be deleted by one side
// only. Same class of residual risk as the Tier A \w note above, and equally
// unreachable for CFPB text.
const HF_CONTROL_RE = /[\p{Cc}\p{Cf}\p{Co}\p{Cs}\p{Cn}]/u;

// Rust's is_whitespace(c): \t \n \r, plus char::is_whitespace (the Unicode
// White_Space property). Spelled out explicitly instead of using JS `\s` because
// the two sets differ (JS `\s` has U+FEFF, White_Space does not). In practice
// U+FEFF is Cf and is already deleted by the control filter above, and the
// White_Space members that are Cc (\v \f ) are also already deleted — a
// subtlety worth stating: \v and \f are DELETED by clean_text, they do NOT
// become spaces.
const HF_WHITESPACE_RE = /[\t\n\v\f\r \x85\xa0\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]/u;

function isChineseChar(cp) {
  // is_chinese_char from tokenizers' bert normalizer (CJK Unified Ideographs +
  // extensions + compatibility ideographs). Deliberately NOT \p{Script=Han}:
  // the Rust code uses these literal ranges.
  return (
    (cp >= 0x4e00 && cp <= 0x9fff) ||
    (cp >= 0x3400 && cp <= 0x4dbf) ||
    (cp >= 0x20000 && cp <= 0x2a6df) ||
    (cp >= 0x2a700 && cp <= 0x2b73f) ||
    (cp >= 0x2b740 && cp <= 0x2b81f) ||
    (cp >= 0x2b920 && cp <= 0x2ceaf) ||
    (cp >= 0xf900 && cp <= 0xfaff) ||
    (cp >= 0x2f800 && cp <= 0x2fa1f)
  );
}

function bertNormalize(text) {
  // 1. clean_text: drop NUL, U+FFFD and control chars; map whitespace -> " ".
  let out = "";
  for (const ch of text) {
    if (ch === "\u0000" || ch === "\uFFFD") continue;
    if (ch !== "\t" && ch !== "\n" && ch !== "\r" && HF_CONTROL_RE.test(ch)) continue;
    out += HF_WHITESPACE_RE.test(ch) ? " " : ch;
  }
  // 2. handle_chinese_chars: pad each CJK codepoint with spaces on both sides.
  let padded = "";
  for (const ch of out) {
    if (isChineseChar(ch.codePointAt(0))) padded += " " + ch + " ";
    else padded += ch;
  }
  // 3. strip_accents (implied by strip_accents=null + lowercase=true):
  //    NFD, then drop every nonspacing mark (category Mn). The string is left
  //    decomposed — HF does not recompose.
  const decomposed = padded.normalize("NFD").replace(/\p{Mn}/gu, "");
  // 4. lowercase, PER CODEPOINT.
  //    WHY per codepoint: NormalizedString::lowercase maps each char through
  //    Rust's char::to_lowercase, which has no string context and therefore
  //    never produces a final sigma. String.prototype.toLowerCase() on the whole
  //    string DOES apply the Final_Sigma rule ("ΟΔΟΣ" -> "οδος" vs per-char
  //    "οδοσ"), which would silently disagree with the training tokenizer on any
  //    uppercase Greek input. Multi-char expansions (U+0130 -> "i" + U+0307,
  //    U+1E9E -> "ß") are preserved because we concatenate the full mapping.
  let lowered = "";
  for (const ch of decomposed) lowered += ch.toLowerCase();
  return lowered;
}

// --- BertPreTokenizer -----------------------------------------------------
//
// pretokenized.split(char::is_whitespace, Removed)  then
// pretokenized.split(is_bert_punc, Isolated)
// is_bert_punc(c) = c.is_ascii_punctuation() || c.is_punctuation()
//   ascii_punctuation = U+21..2F, U+3A..40, U+5B..60, U+7B..7E
//   is_punctuation    = Pc|Pd|Pe|Pf|Pi|Po|Ps  == JS \p{P}
// Note the ASCII half is NOT a subset of \p{P}: $ + < = > ^ ` | ~ are S* but
// still count as punctuation here. Empty splits are dropped by tokenizers.
function isBertPunc(cp) {
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) || (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
    return true;
  }
  return /\p{P}/u.test(String.fromCodePoint(cp));
}

function bertPreTokenize(normalized) {
  const pieces = [];
  let cur = "";
  for (const ch of normalized) {
    const cp = ch.codePointAt(0);
    if (HF_WHITESPACE_RE.test(ch)) {
      if (cur) { pieces.push(cur); cur = ""; }
    } else if (isBertPunc(cp)) {
      if (cur) { pieces.push(cur); cur = ""; }
      pieces.push(ch); // Isolated: the delimiter is its own piece
    } else {
      cur += ch;
    }
  }
  if (cur) pieces.push(cur);
  return pieces;
}

// --- WordPiece ------------------------------------------------------------
//
// models/wordpiece.rs::tokenize — greedy longest-match-first from the left,
// '##' prefix on every non-initial piece, whole word -> [UNK] if any position
// fails, and an immediate [UNK] when the word exceeds max_input_chars_per_word.
// Lengths and slices are in CODEPOINTS (Array.from), not UTF-16 code units.
function wordpieceTokenize(word, vocab, unkToken, prefix, maxInputChars) {
  const cps = Array.from(word);
  if (cps.length > maxInputChars) return [unkToken];
  const subTokens = [];
  let start = 0;
  while (start < cps.length) {
    let end = cps.length;
    let curStr = null;
    while (start < end) {
      let substr = cps.slice(start, end).join("");
      if (start > 0) substr = prefix + substr;
      if (vocab.has(substr)) { curStr = substr; break; }
      end -= 1;
    }
    if (curStr === null) return [unkToken]; // is_bad: the whole word becomes UNK
    subTokens.push(curStr);
    start = end;
  }
  return subTokens;
}

function buildTokenizer(tokenizerJson) {
  const model = tokenizerJson.model || {};
  const vocabObj = model.vocab || {};
  const vocab = new Map();
  for (const key of Object.keys(vocabObj)) vocab.set(key, vocabObj[key]);
  const unkToken = model.unk_token || "[UNK]";
  const prefix = model.continuing_subword_prefix || "##";
  const maxInputChars = model.max_input_chars_per_word || 100;

  // Added/special tokens are matched against the RAW text before normalisation
  // (AddedVocabulary::extract_and_normalize), leftmost-longest. Only tokens with
  // normalized=false are matched verbatim, which is the case for all five BERT
  // specials. Implemented so that a user pasting "[SEP]" into the playground
  // behaves like the Python tokenizer instead of being lowercased into pieces.
  const added = (tokenizerJson.added_tokens || [])
    .filter((t) => t.normalized === false && typeof t.content === "string" && t.content.length)
    .sort((a, b) => b.content.length - a.content.length);

  const truncation = tokenizerJson.truncation || {};
  const maxLength = truncation.max_length || 256;

  const clsId = vocab.get("[CLS]");
  const sepId = vocab.get("[SEP]");
  if (clsId === undefined || sepId === undefined) {
    throw new Error("tokenizer.json: vocab is missing [CLS] and/or [SEP]");
  }

  function tokenizeSegment(raw, outIds) {
    for (const piece of bertPreTokenize(bertNormalize(raw))) {
      for (const tok of wordpieceTokenize(piece, vocab, unkToken, prefix, maxInputChars)) {
        const id = vocab.get(tok);
        outIds.push(id === undefined ? vocab.get(unkToken) : id);
      }
    }
  }

  function encode(text) {
    const ids = [];
    let rest = text || "";
    while (rest.length) {
      let hitIdx = -1;
      let hit = null;
      for (const t of added) {
        const idx = rest.indexOf(t.content);
        if (idx !== -1 && (hitIdx === -1 || idx < hitIdx)) { hitIdx = idx; hit = t; }
      }
      if (hit === null) { tokenizeSegment(rest, ids); break; }
      if (hitIdx > 0) tokenizeSegment(rest.slice(0, hitIdx), ids);
      ids.push(hit.id);
      rest = rest.slice(hitIdx + hit.content.length);
    }

    // Truncation: tokenizers subtracts the post-processor's added-token count
    // from max_length before truncating the content, so 256 total means 254
    // content tokens plus [CLS] and [SEP]. direction "Right" = drop the tail.
    const budget = maxLength - 2;
    const kept = ids.length > budget ? ids.slice(0, budget) : ids;
    const inputIds = [clsId, ...kept, sepId];
    // Single sequence, no padding => attention mask is all ones.
    return { input_ids: inputIds, attention_mask: inputIds.map(() => 1) };
  }

  return { encode, vocabSize: vocab.size, maxLength };
}

// --- onnxruntime-web loader ----------------------------------------------

let ortLoadPromise = null;

function loadOrtRuntime(ortDirUrl) {
  if (typeof document === "undefined") {
    return Promise.reject(new Error("Tier B2 requires a DOM (onnxruntime-web is loaded as a <script>)"));
  }
  if (ortLoadPromise) return ortLoadPromise;
  ortLoadPromise = new Promise((resolve, reject) => {
    if (window.ort) { resolve(window.ort); return; }
    const script = document.createElement("script");
    // ort.wasm.min.js is a UMD bundle: it assigns the global `ort`. It is not an
    // ES module, so it cannot be `import`ed — inject a classic <script> and wait.
    script.src = new URL("ort.wasm.min.js", ortDirUrl).href;
    script.async = true;
    script.onload = () => {
      if (!window.ort) reject(new Error("ort.wasm.min.js loaded but window.ort is undefined"));
      else resolve(window.ort);
    };
    script.onerror = () => reject(new Error(`failed to load ${script.src}`));
    document.head.appendChild(script);
  });
  return ortLoadPromise;
}

async function fetchWithProgress(url, onProgress) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`fetch ${url} failed: HTTP ${res.status}`);
  const lenHeader = res.headers.get("Content-Length");
  const total = lenHeader ? parseInt(lenHeader, 10) : 0;
  if (!res.body || !total || !onProgress) {
    // No Content-Length (chunked / compressed) or no listener: report
    // indeterminate progress once, then fall back to a single buffered read.
    if (onProgress) onProgress(null);
    const buf = await res.arrayBuffer();
    if (onProgress) onProgress(1);
    return buf;
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    onProgress(Math.min(received / total, 1));
  }
  const out = new Uint8Array(received);
  let at = 0;
  for (const c of chunks) { out.set(c, at); at += c.byteLength; }
  return out.buffer;
}

/**
 * Load the Tier B2 engine (DistilBERT int8 ONNX). The ~67 MB model is fetched
 * ONLY when this function is called — nothing here runs on module import.
 *
 * @param {string} baseUrl  directory containing model.int8.onnx, tokenizer.json,
 *                          live_config.json (trailing slash optional)
 * @param {{onProgress?: (frac: number|null) => void, ortDir?: string}} [options]
 */
export async function loadTierB2(baseUrl, options = {}) {
  const t0 = nowMs();
  const onProgress = options.onProgress || null;
  const base = baseUrl.endsWith("/") ? baseUrl : baseUrl + "/";

  const [tokenizerRes, configRes] = await Promise.all([
    fetchJSONWithSize(base + "tokenizer.json"),
    fetchJSONWithSize(base + "live_config.json"),
  ]);
  const tokenizer = buildTokenizer(tokenizerRes.json);
  const liveConfig = configRes.json;
  const classLabels = liveConfig.class_labels || [];
  if (!classLabels.length) throw new Error("live_config.json: class_labels is empty or missing");
  const temperature = liveConfig.temperature === undefined ? 1.0 : Number(liveConfig.temperature);
  if (!(temperature > 0)) throw new Error(`live_config.json: temperature must be > 0 (got ${liveConfig.temperature})`);

  // The ORT runtime lives at demo/vendor/ort/ — resolved against THIS module's
  // URL so the engine works from any page depth.
  const ortDirUrl = options.ortDir
    ? new URL(options.ortDir, document.baseURI)
    : new URL("../vendor/ort/", import.meta.url);
  const ort = await loadOrtRuntime(ortDirUrl);

  // wasmPaths MUST be an ABSOLUTE URL string. A bare relative prefix such as
  // "../vendor/ort/" is resolved by the runtime against the wrong base (the
  // blob/worker context, not this module), and the .mjs/.wasm fetch 404s.
  // Tested — do not "simplify" this back to a relative string.
  ort.env.wasm.wasmPaths = ortDirUrl.href;
  // Single-threaded: the threaded build needs cross-origin isolation
  // (COOP/COEP), which a plain static host (GitHub Pages, python http.server)
  // does not provide. numThreads=1 keeps it working everywhere.
  ort.env.wasm.numThreads = 1;

  const modelBuf = await fetchWithProgress(base + "model.int8.onnx", onProgress);
  const session = await ort.InferenceSession.create(modelBuf, { executionProviders: ["wasm"] });

  const inputNames = session.inputNames || [];
  if (!inputNames.includes("input_ids")) {
    throw new Error(`ONNX model does not expose an 'input_ids' input (has: ${inputNames.join(", ")})`);
  }
  const wantsMask = inputNames.includes("attention_mask");
  const outputName = (session.outputNames || []).includes("logits")
    ? "logits"
    : (session.outputNames || [])[0];

  async function predict(text) {
    const start = nowMs();
    const enc = tokenizer.encode(text || "");
    const n = enc.input_ids.length;
    // int64 inputs => BigInt64Array. BigInt conversion is exact for token ids.
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(enc.input_ids, BigInt), [1, n]),
    };
    if (wantsMask) {
      feeds.attention_mask = new ort.Tensor("int64", BigInt64Array.from(enc.attention_mask, BigInt), [1, n]);
    }
    const out = await session.run(feeds);
    const logits = out[outputName].data;
    if (logits.length !== classLabels.length) {
      throw new Error(`model returned ${logits.length} logits, expected ${classLabels.length}`);
    }
    // tier_b.py: probs = softmax_np(eval_logits / temperature). Divide first,
    // then softmax — the same op order, so the same rounding.
    const scaled = new Array(logits.length);
    for (let i = 0; i < logits.length; i++) scaled[i] = logits[i] / temperature;
    const proba = softmax(scaled);
    const best = argmax(proba);
    const probs = {};
    for (let i = 0; i < classLabels.length; i++) probs[classLabels[i]] = proba[i];
    return {
      label: classLabels[best],
      p_max: proba[best],
      probs,
      latency_ms: nowMs() - start,
      n_tokens: n,
    };
  }

  return {
    predict,
    meta: {
      tier: "B2",
      engine: "distilbert int8 onnx (onnxruntime-web, wasm, 1 thread)",
      class_labels: classLabels,
      temperature,
      max_length: tokenizer.maxLength,
      vocab_size: tokenizer.vocabSize,
      size_bytes: modelBuf.byteLength,
      tokenizer_bytes: tokenizerRes.sizeBytes,
      ort_version: (ort.env && ort.env.versions && ort.env.versions.web) || null,
      provenance: liveConfig.provenance || null,
      quantization: liveConfig.quantization || "int8 (dynamic)",
      input_names: inputNames,
      output_name: outputName,
      load_ms: nowMs() - t0,
      source_url: base,
    },
  };
}
