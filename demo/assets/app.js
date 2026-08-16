// Triage Router — static demo app.
// Vanilla ES module, zero dependencies. All data loaded via fetch of ./data/*.json.
// Every panel degrades to a visible banner (never a blank page / console-only error)
// when its data file is missing or malformed.

const DATA_DIR = "./data/";

const OKABE_ITO = {
  blue: "#0072b2",
  orange: "#e69f00",
  green: "#009e73",
  vermillion: "#d55e00",
  purple: "#cc79a7",
  yellow: "#f0e442",
  skyblue: "#56b4e9",
  black: "#4a4f57",
};

// ----------------------------------------------------------------------------
// data cache / fetch helper
// ----------------------------------------------------------------------------

const dataCache = {};

async function loadJSON(name) {
  if (name in dataCache) return dataCache[name];
  try {
    const res = await fetch(DATA_DIR + name, { cache: "no-store" });
    if (!res.ok) {
      dataCache[name] = { __missing: true, __status: res.status };
      return dataCache[name];
    }
    const json = await res.json();
    dataCache[name] = json;
    return json;
  } catch (err) {
    dataCache[name] = { __missing: true, __error: String(err) };
    return dataCache[name];
  }
}

function isMissing(obj) {
  return !obj || obj.__missing === true;
}

function missingBanner(fileName, extra) {
  const div = document.createElement("div");
  div.className = "data-missing-banner";
  div.textContent = `Data file missing or failed to load: ${fileName}. ` +
    (extra || "This panel will populate once the file is committed by the build pipeline.");
  return div;
}

// ----------------------------------------------------------------------------
// small formatting helpers
// ----------------------------------------------------------------------------

function fmtNum(x, digits = 3) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return Number(x).toFixed(digits);
}

function fmtUsd(x, digits = 4) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return "$" + Number(x).toFixed(digits);
}

function fmtPct(x, digits = 1) {
  if (x === null || x === undefined || Number.isNaN(x)) return "—";
  return (Number(x) * 100).toFixed(digits) + "%";
}

function metricStr(m, digits = 3) {
  if (!m) return "—";
  if (m.ci_lo !== undefined && m.ci_hi !== undefined) {
    return `${fmtNum(m.point, digits)} [${fmtNum(m.ci_lo, digits)}, ${fmtNum(m.ci_hi, digits)}]`;
  }
  return fmtNum(m.point, digits);
}

function el(tag, className, children) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (children !== undefined) {
    (Array.isArray(children) ? children : [children]).forEach((c) => {
      if (c === null || c === undefined) return;
      if (typeof c === "string") node.appendChild(document.createTextNode(c));
      else node.appendChild(c);
    });
  }
  return node;
}

function evidenceBadge(cls) {
  const known = ["measured", "estimated", "projected", "derived", "provenance"];
  const norm = known.includes(cls) ? cls : "derived";
  const span = el("span", `evidence-badge ${norm}`, cls || "unknown");
  return span;
}

// ---- repo links -------------------------------------------------------------
// One place that turns a repo-relative path / commit sha into a URL. `repo.url_base` is a
// single constant in src/triage_lab/demo_build.py and is EMPTY until this repo is pushed to
// GitHub; while it is empty every reference renders as plain <code> with a chip saying so,
// so the page never ships a dead link. Kinds: blob (a file), commit (a sha), actions (CI).
const REPO_LINK_PENDING_NOTE = "link resolves after GitHub push";

function repoUrl(repo, kind, ref) {
  const base = (repo && repo.url_base) || "";
  if (!base) return null;
  const branch = (repo && repo.default_branch) || "main";
  if (kind === "actions") return `${base}/actions`;
  if (kind === "commit") return `${base}/commit/${ref}`;
  return `${base}/blob/${branch}/${ref}`;
}

function repoRef(repo, kind, ref, label) {
  const text = label || ref;
  const href = repoUrl(repo, kind, ref);
  if (!href) {
    return el("span", "repo-ref", [
      el("code", "repo-path", text),
      el("span", "repo-ref-chip", REPO_LINK_PENDING_NOTE),
    ]);
  }
  const link = el("a", "repo-path", text);
  link.href = href;
  link.target = "_blank";
  link.rel = "noopener noreferrer";
  return el("span", "repo-ref", link);
}

// run-id chip: 8-char mono prefix, clickable -> opens receipts drawer
function runChip(runId) {
  if (!runId) return null;
  const chip = el("span", "run-chip", runId.slice(0, 8));
  chip.title = runId;
  chip.addEventListener("click", () => openDrawerForRun(runId));
  return chip;
}

function sourceChip(sourcePath) {
  if (!sourcePath) return null;
  const chip = el("span", "source-chip", sourcePath);
  chip.title = sourcePath;
  return chip;
}

// ----------------------------------------------------------------------------
// receipts drawer (global)
// ----------------------------------------------------------------------------

let runsIndexCache = null;
let receiptsCache = null;

async function openDrawerForRun(runId) {
  const overlay = document.getElementById("drawer-overlay");
  const drawer = document.getElementById("drawer");
  const title = document.getElementById("drawer-title");
  const body = document.getElementById("drawer-body");

  title.textContent = "run " + runId.slice(0, 8) + "…";
  body.textContent = "loading…";
  overlay.classList.add("open");
  drawer.classList.add("open");

  if (!runsIndexCache) runsIndexCache = await loadJSON("runs_index.json");
  if (!receiptsCache) receiptsCache = await loadJSON("receipts.json");

  body.innerHTML = "";

  if (isMissing(runsIndexCache)) {
    body.appendChild(missingBanner("runs_index.json"));
    return;
  }

  const record = runsIndexCache[runId];
  if (!record) {
    body.appendChild(el("p", null, `No runs_index.json entry found for run id ${runId}.`));
    return;
  }

  const table = el("table", "kv-table");
  const rows = [
    ["config name", record.config_name],
    ["config path", record.config_path],
    ["config sha256", record.config_sha256],
    ["slice", record.slice],
    ["timestamp (UTC)", record.timestamp_utc],
    ["git SHA", record.git_sha],
    ["tier", record.tier],
    ["model label", record.model_label],
    ["cost (USD)", fmtUsd(record.cost_usd, 6)],
    ["wall clock (s)", fmtNum(record.wall_clock_seconds, 2)],
  ];
  rows.forEach(([k, v]) => {
    const tr = el("tr", null, [el("td", "k", k), el("td", null, v === undefined || v === null ? "—" : String(v))]);
    table.appendChild(tr);
  });
  body.appendChild(table);

  if (record.metrics) {
    body.appendChild(el("h4", null, "metrics"));
    const mTable = el("table", "kv-table");
    Object.entries(record.metrics).forEach(([k, v]) => {
      const valStr = v && typeof v === "object" && "point" in v ? metricStr(v) : JSON.stringify(v);
      mTable.appendChild(el("tr", null, [el("td", "k", k), el("td", null, valStr)]));
    });
    body.appendChild(mTable);
  }

  if (record.dataset) {
    body.appendChild(el("h4", null, "dataset"));
    body.appendChild(el("pre", "raw-json", JSON.stringify(record.dataset, null, 2)));
  }

  // matching Tier C receipts aggregate — receipts.json.runs is keyed by run_id
  if (!isMissing(receiptsCache)) {
    const receiptRuns = receiptsCache.runs || {};
    const match = receiptRuns[runId] || Object.values(receiptRuns).find((r) => r.run_id === runId);
    if (match) {
      body.appendChild(el("h4", null, "Tier C receipts aggregate"));
      const rTable = el("table", "kv-table");
      const rrows = [
        ["model", match.model],
        ["n_calls", match.n_calls],
        ["total cost (USD)", fmtUsd(match.total_cost_usd, 4)],
        ["parse failures", match.parse_failures],
        ["prompt tokens", match.token_totals ? match.token_totals.prompt : undefined],
        ["completion tokens", match.token_totals ? match.token_totals.completion : undefined],
        ["provider mix", match.provider_mix ? JSON.stringify(match.provider_mix) : undefined],
        ["raw log path", match.raw_log_path],
      ];
      rrows.forEach(([k, v]) => {
        rTable.appendChild(el("tr", null, [el("td", "k", k), el("td", null, v === undefined || v === null ? "—" : String(v))]));
      });
      body.appendChild(rTable);
    }
  }
}

function closeDrawer() {
  document.getElementById("drawer-overlay").classList.remove("open");
  document.getElementById("drawer").classList.remove("open");
}

// ----------------------------------------------------------------------------
// nav / tabs
// ----------------------------------------------------------------------------

const PANEL_ORDER = ["playground", "frontier", "policy", "drift", "calibration", "receipts", "casestudy"];
const PANEL_LABELS = {
  playground: "Playground",
  frontier: "Frontier",
  policy: "Policy",
  drift: "Drift",
  calibration: "Calibration",
  receipts: "Receipts",
  casestudy: "Case study",
};

function showPanel(name) {
  PANEL_ORDER.forEach((p) => {
    const panel = document.getElementById("panel-" + p);
    if (panel) panel.classList.toggle("active", p === name);
  });
  document.querySelectorAll(".nav-link").forEach((a) => {
    a.classList.toggle("active", a.dataset.panel === name);
  });
  if (location.hash.slice(1) !== name) {
    history.replaceState(null, "", "#" + name);
  }
}

function initNav() {
  document.querySelectorAll(".sidenav .nav-link").forEach((a) => {
    a.addEventListener("click", (e) => {
      e.preventDefault();
      showPanel(a.dataset.panel);
    });
  });

  const topTabs = document.getElementById("top-tabs");
  PANEL_ORDER.forEach((p, i) => {
    const a = el("a", "nav-link", [
      el("span", "nav-num", String(i + 1).padStart(2, "0")),
      el("span", "nav-label", PANEL_LABELS[p]),
    ]);
    a.href = "#" + p;
    a.dataset.panel = p;
    a.addEventListener("click", (e) => {
      e.preventDefault();
      showPanel(p);
    });
    topTabs.appendChild(a);
  });

  const initial = PANEL_ORDER.includes(location.hash.slice(1)) ? location.hash.slice(1) : "playground";
  showPanel(initial);

  window.addEventListener("hashchange", () => {
    const p = location.hash.slice(1);
    if (PANEL_ORDER.includes(p)) showPanel(p);
  });
}

// ----------------------------------------------------------------------------
// SVG chart primitives (hand-rolled, no libs)
// ----------------------------------------------------------------------------

const SVG_NS = "http://www.w3.org/2000/svg";

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  Object.entries(attrs || {}).forEach(([k, v]) => node.setAttribute(k, v));
  return node;
}

// Log-axis ticks. Decades first; if fewer than three of them fall inside the
// domain the decade is subdivided by successively finer mantissa sets, so a
// sub-decade span (which is what the cost axis actually has) still gets an axis
// a reader can measure against.
function logTicks(domain) {
  const [d0, d1] = domain;
  const MANTISSAS = [[1], [1, 5], [1, 2, 5], [1, 1.5, 2, 3, 5, 7]];
  for (const ms of MANTISSAS) {
    const ticks = [];
    for (let e = Math.floor(Math.log10(d0)); e <= Math.ceil(Math.log10(d1)); e++) {
      ms.forEach((m) => {
        const v = m * Math.pow(10, e);
        if (v >= d0 && v <= d1) ticks.push(v);
      });
    }
    if (ticks.length >= 3 || ms === MANTISSAS[MANTISSAS.length - 1]) {
      return ticks.sort((a, b) => a - b);
    }
  }
  return [];
}

// linear scale
function makeLinearScale(domain, range) {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  return (v) => r0 + ((v - d0) / (d1 - d0 || 1)) * (r1 - r0);
}

// log10 scale (domain must be > 0)
function makeLogScale(domain, range) {
  const [d0, d1] = domain;
  const ld0 = Math.log10(Math.max(d0, 1e-9));
  const ld1 = Math.log10(Math.max(d1, 1e-9));
  const [r0, r1] = range;
  return (v) => r0 + ((Math.log10(Math.max(v, 1e-9)) - ld0) / (ld1 - ld0 || 1)) * (r1 - r0);
}

// Axis furniture. Tick values are data, so they set in the mono stack with
// tabular figures (via .chart-tick); axis names are labels, so they set in the
// small-caps metadata register (via .chart-axis-label). Both label offsets are
// derived from the caller's margins rather than hardcoded — the reliability
// diagrams used to draw their x-axis name 2px past the bottom of the viewBox,
// which clipped "confidence" on every small multiple.
function drawAxes(svg, opts) {
  const { x0, x1, y0, y1, xTicks, yTicks, xLabel, yLabel } = opts;
  const H = Number(svg.getAttribute("height")) || y1;

  svg.appendChild(svgEl("line", { class: "chart-axis", x1: x0, y1: y1, x2: x1, y2: y1 }));
  svg.appendChild(svgEl("line", { class: "chart-axis", x1: x0, y1: y0, x2: x0, y2: y1 }));

  (xTicks || []).forEach(({ x, label }) => {
    svg.appendChild(svgEl("line", { class: "chart-gridline", x1: x, y1: y0, x2: x, y2: y1 }));
    const t = svgEl("text", { class: "chart-tick", x, y: y1 + 15, "text-anchor": "middle" });
    t.textContent = label;
    svg.appendChild(t);
  });

  (yTicks || []).forEach(({ y, label }) => {
    svg.appendChild(svgEl("line", { class: "chart-gridline", x1: x0, y1: y, x2: x1, y2: y }));
    const t = svgEl("text", { class: "chart-tick", x: x0 - 9, y: y + 3.5, "text-anchor": "end" });
    t.textContent = label;
    svg.appendChild(t);
  });

  if (xLabel) {
    // sit on the last text baseline that still fits inside the viewBox
    const t = svgEl("text", {
      class: "chart-axis-label", x: (x0 + x1) / 2, y: Math.min(y1 + 33, H - 4),
      "text-anchor": "middle",
    });
    t.textContent = xLabel;
    svg.appendChild(t);
  }
  if (yLabel) {
    const t = svgEl("text", {
      class: "chart-axis-label", x: -((y0 + y1) / 2), y: 11, "text-anchor": "middle",
      transform: "rotate(-90)",
    });
    t.textContent = yLabel;
    svg.appendChild(t);
  }
}

// ---- legend rows ------------------------------------------------------------
// Legends always live below the plot as a list. With ten drift arms there is no
// empty quadrant left to float into, and an in-plot legend would sit on top of
// the lines it names. The swatch mirrors the mark it stands for: a rule for a
// line series, a disc or square for a scatter marker.
function legendRow(color, label, shape) {
  const item = el("span", null);
  const swatch = el("span", "legend-swatch" + (shape ? " legend-" + shape : ""));
  if (shape === "dot" || shape === "square") swatch.style.background = color;
  else swatch.style.color = color;
  item.appendChild(swatch);
  item.appendChild(el("span", "legend-text", label));
  return item;
}

function pendingLegendRow(label) {
  const item = el("span", null);
  item.appendChild(el("span", "legend-swatch pending"));
  item.appendChild(el("span", "legend-text", label));
  return item;
}

// ----------------------------------------------------------------------------
// PANEL 1: Triage playground
// ----------------------------------------------------------------------------

let samplesState = { samples: [], filtered: [], selectedId: null };

async function initPlayground() {
  const bannerSlot = document.getElementById("playground-banner-slot");
  const layout = document.getElementById("playground-layout");
  bannerSlot.innerHTML = "";

  const data = await loadJSON("samples.json");
  if (isMissing(data) || !Array.isArray(data.samples)) {
    bannerSlot.appendChild(missingBanner("samples.json"));
    return;
  }

  layout.style.display = "";
  samplesState.samples = data.samples;
  samplesState.filtered = data.samples;

  // populate class filter
  const classSet = new Set(data.samples.map((s) => s.y_true));
  const filterSel = document.getElementById("sample-class-filter");
  Array.from(classSet).sort().forEach((c) => {
    const opt = el("option", null, c);
    opt.value = c;
    filterSel.appendChild(opt);
  });

  const searchInput = document.getElementById("sample-search");
  searchInput.addEventListener("input", applyPlaygroundFilters);
  filterSel.addEventListener("change", applyPlaygroundFilters);

  renderSampleList();

  const liveSlot = document.getElementById("live-inference-slot");
  if (liveSlot) {
    liveSlot.innerHTML = "";
    liveSlot.appendChild(await buildLiveInferenceSection());
  }
}

function applyPlaygroundFilters() {
  const q = document.getElementById("sample-search").value.trim().toLowerCase();
  const cls = document.getElementById("sample-class-filter").value;
  samplesState.filtered = samplesState.samples.filter((s) => {
    if (cls && s.y_true !== cls) return false;
    if (q && !(s.narrative || "").toLowerCase().includes(q)) return false;
    return true;
  });
  renderSampleList();
}

function renderSampleList() {
  const list = document.getElementById("sample-list");
  list.innerHTML = "";
  samplesState.filtered.forEach((s) => {
    const item = el("div", "sample-list-item");
    if (s.complaint_id === samplesState.selectedId) item.classList.add("selected");
    const snippet = (s.narrative || "").slice(0, 90).replace(/\s+/g, " ");
    item.appendChild(el("span", "id", "#" + s.complaint_id + " · "));
    item.appendChild(el("span", "label", s.y_true));
    item.appendChild(el("span", "snippet", snippet + "…"));
    item.addEventListener("click", () => {
      samplesState.selectedId = s.complaint_id;
      renderSampleList();
      renderSampleDetail(s);
    });
    list.appendChild(item);
  });
  if (!samplesState.filtered.length) {
    list.appendChild(el("div", "sample-list-item", "No narratives match the current filter."));
  }
}

function tierCardFor(label, tierData) {
  if (!tierData) return el("div", "tier-card pending-slot", ["no data"]);
  if (tierData.pending) {
    return el("div", "tier-card pending-slot", [
      el("h4", null, label),
      el("div", null, "pending Tier B backfill"),
    ]);
  }
  const correctFlag = tierData.correct;
  const cardClass = "tier-card" + (correctFlag === true ? " correct" : correctFlag === false ? " incorrect" : "");
  const card = el("div", cardClass);
  card.appendChild(el("h4", null, label));
  card.appendChild(el("div", "kv", [el("span", "k", "label"), el("span", null, tierData.label || "—")]));
  if (tierData.p_max !== undefined) {
    card.appendChild(el("div", "kv", [el("span", "k", "p_max"), el("span", null, fmtNum(tierData.p_max, 3))]));
  }
  if (correctFlag !== undefined) {
    card.appendChild(el("div", "kv", [el("span", "k", "correct"), el("span", null, correctFlag ? "yes" : "no")]));
  }
  if (tierData.cost_usd !== undefined) {
    card.appendChild(el("div", "kv", [el("span", "k", "cost"), el("span", null, fmtUsd(tierData.cost_usd, 6))]));
  }
  if (tierData.latency_ms !== undefined) {
    card.appendChild(el("div", "kv", [el("span", "k", "latency"), el("span", null, fmtNum(tierData.latency_ms, 0) + " ms")]));
  }
  if (tierData.provider) {
    card.appendChild(el("div", "kv", [el("span", "k", "provider"), el("span", null, tierData.provider)]));
  }
  if (tierData.prompt_tokens !== undefined) {
    card.appendChild(el("div", "kv", [
      el("span", "k", "tokens"),
      el("span", null, `${tierData.prompt_tokens}p / ${tierData.completion_tokens}c`),
    ]));
  }
  if (tierData.parse_failed) {
    card.appendChild(el("div", "kv", [el("span", "k", "parse_failed"), el("span", null, "true")]));
  }
  if (tierData.run_id) {
    const kv = el("div", "kv", [el("span", "k", "run"), null]);
    kv.appendChild(runChip(tierData.run_id));
    card.appendChild(kv);
  }
  return card;
}

function renderSampleDetail(s) {
  const wrap = document.getElementById("sample-detail");
  wrap.innerHTML = "";

  wrap.appendChild(el("div", null, [
    el("strong", null, "true label: "),
    el("span", null, s.y_true),
  ]));
  wrap.appendChild(el("div", "narrative-block", s.narrative));

  const strip = el("div", "tier-strip");
  const tiers = s.tiers || {};
  strip.appendChild(tierCardFor("Tier A (LogReg)", tiers.tier_a_logreg));
  strip.appendChild(tierCardFor("Tier B1 (ModernBERT)", tiers.tier_b1));
  strip.appendChild(tierCardFor("Tier B2 (DistilBERT)", tiers.tier_b2));
  strip.appendChild(tierCardFor("Haiku 4.5", tiers.haiku));
  strip.appendChild(tierCardFor("Sonnet 5", tiers.sonnet));
  wrap.appendChild(strip);

  if (s.router && Array.isArray(s.router.path)) {
    wrap.appendChild(el("h4", null, "router decision path"));
    const pathDiv = el("div", "router-path");
    s.router.path.forEach((step, i) => {
      pathDiv.appendChild(el("span", "step", step));
      if (i < s.router.path.length - 1) pathDiv.appendChild(el("span", "arrow", "→"));
    });
    wrap.appendChild(pathDiv);
    if (s.router.policy) {
      wrap.appendChild(el("div", "panel-desc", `policy: ${s.router.policy}${s.router.tau !== undefined ? ", tau=" + fmtNum(s.router.tau, 3) : ""}${s.router.note ? " — " + s.router.note : ""}`));
    }
  }
}

// ----------------------------------------------------------------------------
// Live in-browser inference (Tier A / Tier B2) — lazy-loaded engine module.
// demo/assets/live.js is written by a concurrent task; it is imported lazily
// via dynamic import() so the rest of the demo works even if it is missing.
// ----------------------------------------------------------------------------

const liveState = {
  tierA: { engine: null, loading: false, error: null },
  tierB2: { engine: null, loading: false, progress: null, consent: false, error: null },
  agreementReport: undefined, // undefined = not fetched yet, null = missing
};

const LIVE_DISCLOSURE_TEXT = "Approximate in-browser implementation (int8 / re-implemented pipeline). " +
  "Official numbers are the frozen harness records in results/runs.jsonl — see the receipts drawer. " +
  "Browser-vs-Python agreement on the curated 200: see agreement report.";

async function fetchAgreementReport() {
  if (liveState.agreementReport !== undefined) return liveState.agreementReport;
  try {
    const res = await fetch("./live/agreement_report.json", { cache: "no-store" });
    if (!res.ok) {
      liveState.agreementReport = null;
      return null;
    }
    liveState.agreementReport = await res.json();
    return liveState.agreementReport;
  } catch (err) {
    liveState.agreementReport = null;
    return null;
  }
}

function agreementSummaryText(report) {
  if (!report) return "agreement report pending";
  const parts = [];
  const a = report.tier_a;
  if (a && a.label_agreement_vs_official !== undefined) {
    parts.push(`Tier A ${fmtPct(a.label_agreement_vs_official, 1)}`);
  }
  const b2 = report.tier_b2 && report.tier_b2.vs_official_fp32;
  if (b2 && b2.label_agreement_vs_official !== undefined) {
    parts.push(`Tier B2 ${fmtPct(b2.label_agreement_vs_official, 1)} vs official fp32`);
  }
  return parts.length ? parts.join(", ") : "agreement report pending";
}

async function buildLiveDisclosure() {
  const wrap = el("div", "live-disclosure");
  wrap.appendChild(el("span", null, LIVE_DISCLOSURE_TEXT + " "));
  const rateSpan = el("span", "live-disclosure-rates", "loading agreement report…");
  wrap.appendChild(rateSpan);
  fetchAgreementReport().then((report) => {
    rateSpan.textContent = "(" + agreementSummaryText(report) + ")";
  });
  return wrap;
}

function liveResultCard(title, result, official) {
  const card = el("div", "tier-card live-result-card");
  card.appendChild(el("h4", null, title));
  if (!result) return card;
  card.appendChild(el("div", "kv", [el("span", "k", "label"), el("span", null, result.label || "—")]));
  card.appendChild(el("div", "kv", [el("span", "k", "p_max"), el("span", null, fmtNum(result.p_max, 3))]));
  card.appendChild(el("div", "kv", [el("span", "k", "latency"), el("span", null, fmtNum(result.latency_ms, 0) + " ms")]));
  if (result.probs) {
    const top3 = Object.entries(result.probs).sort((a, b) => b[1] - a[1]).slice(0, 3);
    const bars = el("div", "prob-bars");
    top3.forEach(([cls, p]) => {
      const row = el("div", "prob-bar-row");
      row.appendChild(el("span", "prob-bar-label", cls));
      const track = el("div", "prob-bar-track");
      const fill = el("div", "prob-bar-fill");
      fill.style.width = fmtPct(p, 0);
      track.appendChild(fill);
      row.appendChild(track);
      row.appendChild(el("span", "prob-bar-pct", fmtPct(p, 1)));
      bars.appendChild(row);
    });
    card.appendChild(bars);
  }
  if (official && official.label !== undefined) {
    const agree = official.label === result.label;
    card.appendChild(el("div", "kv", [
      el("span", "k", "vs precomputed card"),
      el("span", agree ? null : "live-disagree", `official: ${official.label}${agree ? " (agrees)" : " (disagrees)"}`),
    ]));
  }
  return card;
}

function liveErrorBox(msg) {
  return el("div", "data-missing-banner live-error", "Live inference error: " + msg);
}

async function runLiveInference(tierKey, text, resultSlot, official) {
  resultSlot.innerHTML = "";
  if (!text || !text.trim()) {
    resultSlot.appendChild(el("div", "panel-desc", "No text to run — select a sample or paste text."));
    return;
  }
  const state = tierKey === "tier_a" ? liveState.tierA : liveState.tierB2;
  if (!state.engine) {
    resultSlot.appendChild(el("div", "panel-desc", "Engine not loaded yet."));
    return;
  }
  try {
    resultSlot.appendChild(el("div", "panel-desc", "running…"));
    const result = await state.engine.predict(text);
    resultSlot.innerHTML = "";
    resultSlot.appendChild(liveResultCard(tierKey === "tier_a" ? "Tier A (live)" : "Tier B2 (live)", result, official));
  } catch (err) {
    resultSlot.innerHTML = "";
    resultSlot.appendChild(liveErrorBox(String(err && err.message ? err.message : err)));
  }
}

function currentLiveText() {
  const pasted = document.getElementById("live-paste-text");
  const pastedVal = pasted ? pasted.value.trim() : "";
  if (pastedVal) return pastedVal;
  const s = samplesState.samples.find((x) => x.complaint_id === samplesState.selectedId);
  return s ? (s.narrative || "") : "";
}

function currentOfficialTierData(tierKey) {
  // Pasted text has no precomputed record — comparing it against the selected
  // sample's official card would be misleading, so suppress the comparison.
  const pasted = document.getElementById("live-paste-text");
  if (pasted && pasted.value.trim()) return null;
  const s = samplesState.samples.find((x) => x.complaint_id === samplesState.selectedId);
  if (!s || !s.tiers) return null;
  const key = tierKey === "tier_a" ? "tier_a_logreg" : "tier_b2";
  return s.tiers[key] || null;
}

async function buildLiveInferenceSection() {
  const section = el("div", "live-inference-section card");
  section.appendChild(el("h3", null, "Live in-browser inference"));

  section.appendChild(await buildLiveDisclosure());

  const pasteWrap = el("div", "live-paste-wrap");
  pasteWrap.appendChild(el("label", "live-paste-label", "…or paste your own complaint text"));
  const textarea = document.createElement("textarea");
  textarea.id = "live-paste-text";
  textarea.rows = 3;
  textarea.placeholder = "Paste a complaint narrative to run live inference on custom text…";
  pasteWrap.appendChild(textarea);
  section.appendChild(pasteWrap);

  // ---- Tier A ----
  const tierAWrap = el("div", "live-tier-block");
  tierAWrap.appendChild(el("h4", null, "Tier A live"));
  const tierAStatus = el("div", "panel-desc live-status");
  const tierARunBtn = el("button", null, "Run Tier A live");
  tierARunBtn.type = "button";
  const tierAResultSlot = el("div", "live-result-slot");
  tierAWrap.appendChild(tierAStatus);
  tierAWrap.appendChild(tierARunBtn);
  tierAWrap.appendChild(tierAResultSlot);
  section.appendChild(tierAWrap);

  tierARunBtn.addEventListener("click", async () => {
    if (!liveState.tierA.engine && !liveState.tierA.loading) {
      liveState.tierA.loading = true;
      tierAStatus.textContent = "loading weights…";
      try {
        // best-effort size probe (HEAD) purely for the loading-status label
        try {
          const head = await fetch("./live/tier_a/tier_a_live.json", { method: "HEAD", cache: "no-store" });
          const len = head.headers.get("content-length");
          if (len) tierAStatus.textContent = `loading weights (~${(Number(len) / 1e6).toFixed(2)} MB)…`;
        } catch (probeErr) {
          // ignore — size label is best-effort only
        }
        const mod = await import("./live.js");
        const engine = await mod.loadTierA("./live/tier_a/tier_a_live.json");
        liveState.tierA.engine = engine;
        const sizeMb = engine.meta && engine.meta.size_bytes ? (engine.meta.size_bytes / 1e6).toFixed(2) + " MB" : "size unknown";
        tierAStatus.textContent = `Tier A engine loaded (${sizeMb}).`;
      } catch (err) {
        liveState.tierA.loading = false;
        tierAStatus.textContent = "";
        tierAResultSlot.innerHTML = "";
        tierAResultSlot.appendChild(liveErrorBox(String(err && err.message ? err.message : err)));
        return;
      }
      liveState.tierA.loading = false;
    }
    await runLiveInference("tier_a", currentLiveText(), tierAResultSlot, currentOfficialTierData("tier_a"));
  });

  // ---- Tier B2 ----
  const tierBWrap = el("div", "live-tier-block");
  tierBWrap.appendChild(el("h4", null, "Tier B2 live"));
  tierBWrap.appendChild(el("div", "panel-desc", "Runs DistilBERT (int8, ONNX) locally via WebAssembly. This downloads a large file."));
  const consentBtn = el("button", null, "Load DistilBERT int8 model (~64 MB download)");
  consentBtn.type = "button";
  const progressWrap = el("div", "live-progress-wrap", null);
  const progressBar = el("div", "live-progress-bar");
  const progressFill = el("div", "live-progress-fill");
  progressBar.appendChild(progressFill);
  progressWrap.appendChild(progressBar);
  progressWrap.style.display = "none";
  const tierBStatus = el("div", "panel-desc live-status");
  const tierBRunBtn = el("button", null, "Run Tier B2 live");
  tierBRunBtn.type = "button";
  tierBRunBtn.style.display = "none";
  const tierBResultSlot = el("div", "live-result-slot");

  tierBWrap.appendChild(consentBtn);
  tierBWrap.appendChild(progressWrap);
  tierBWrap.appendChild(tierBStatus);
  tierBWrap.appendChild(tierBRunBtn);
  tierBWrap.appendChild(tierBResultSlot);
  section.appendChild(tierBWrap);

  consentBtn.addEventListener("click", async () => {
    if (liveState.tierB2.engine || liveState.tierB2.loading) return;
    liveState.tierB2.loading = true;
    liveState.tierB2.consent = true;
    consentBtn.disabled = true;
    progressWrap.style.display = "";
    tierBStatus.textContent = "downloading model…";
    try {
      const mod = await import("./live.js");
      const engine = await mod.loadTierB2("./live/tier_b2/", {
        onProgress: (fraction) => {
          if (fraction === null || fraction === undefined) {
            progressFill.style.width = "100%";
            tierBStatus.textContent = "downloading model (progress unknown)…";
          } else {
            progressFill.style.width = Math.round(fraction * 100) + "%";
            tierBStatus.textContent = `downloading model… ${Math.round(fraction * 100)}%`;
          }
        },
      });
      liveState.tierB2.engine = engine;
      liveState.tierB2.loading = false;
      const sizeMb = engine.meta && engine.meta.size_bytes ? (engine.meta.size_bytes / 1e6).toFixed(1) : "?";
      tierBStatus.textContent = `Tier B2 engine loaded (${sizeMb} MB).`;
      progressWrap.style.display = "none";
      tierBRunBtn.style.display = "";
    } catch (err) {
      liveState.tierB2.loading = false;
      consentBtn.disabled = false;
      progressWrap.style.display = "none";
      tierBStatus.textContent = "";
      tierBResultSlot.innerHTML = "";
      tierBResultSlot.appendChild(liveErrorBox(String(err && err.message ? err.message : err)));
    }
  });

  tierBRunBtn.addEventListener("click", async () => {
    await runLiveInference("tier_b2", currentLiveText(), tierBResultSlot, currentOfficialTierData("tier_b2"));
  });

  return section;
}

// ----------------------------------------------------------------------------
// PANEL 2: Cost-quality frontier
// ----------------------------------------------------------------------------

async function initFrontier() {
  const bannerSlot = document.getElementById("frontier-banner-slot");
  const claimsWrap = document.getElementById("frontier-claims");
  const chartWrap = document.getElementById("frontier-chart-wrap");
  bannerSlot.innerHTML = "";
  claimsWrap.innerHTML = "";

  const data = await loadJSON("frontier.json");
  if (isMissing(data) || !Array.isArray(data.points)) {
    bannerSlot.appendChild(missingBanner("frontier.json"));
    return;
  }

  // frontier.json .claims.claims is a list of paired comparisons, each with
  // delta_* metric objects and an optional gate verdict. Rendered as a table,
  // not as one bordered card per claim: fourteen identically-shaped cards make
  // the deltas impossible to compare against each other, which is the only
  // thing a reader wants to do with them. In a table the columns do that work.
  const claimsList = data.claims && Array.isArray(data.claims.claims) ? data.claims.claims : [];
  if (claimsList.length) {
    claimsWrap.appendChild(claimsTable(claimsList));
    const note = el("p", "table-note", [
      "A point value is set in ink when its 95% interval clears zero and stays grey when the " +
      "interval spans it, so the columns can be read for significance directly. ",
      "Gate verdicts follow the stricter criterion recorded in the artifact: CERTIFIED requires " +
      "every gated metric to be favourable and significant. ",
    ]);
    if (data.claims && data.claims.source) {
      note.appendChild(el("span", null, "Source: "));
      note.appendChild(sourceChip(data.claims.source));
    }
    claimsWrap.appendChild(note);
  }

  chartWrap.style.display = "";
  drawFrontierChart(data);
}

// Column order is discovered from the data (union of delta_* keys, first-seen
// order) so a new gated metric appears without a code change here.
function claimDeltaKeys(claims) {
  const keys = [];
  claims.forEach((c) => Object.keys(c).forEach((k) => {
    if (k.startsWith("delta_") && c[k] && typeof c[k] === "object" && !keys.includes(k)) keys.push(k);
  }));
  return keys;
}

function deltaCell(metric) {
  const td = el("td", "claim-num");
  if (!metric || typeof metric !== "object") {
    td.appendChild(el("span", "claim-point", "—"));
    return td;
  }
  const digits = Math.abs(metric.point) >= 1 ? 2 : 4;
  const point = el("span", "claim-point" + (metric.excludes_zero ? " excludes-zero" : ""),
    (metric.point > 0 ? "+" : "") + fmtNum(metric.point, digits));
  td.appendChild(point);
  if (metric.ci_lo !== undefined && metric.ci_hi !== undefined) {
    td.appendChild(el("span", "claim-ci",
      `${fmtNum(metric.ci_lo, digits)} … ${fmtNum(metric.ci_hi, digits)}`));
  }
  return td;
}

// The gate is the one judgement on this page, so it is the one place status
// colour lands — on the word, never behind it.
function gateCell(claim) {
  const td = el("td");
  const gate = claim.gate;
  if (!gate) {
    td.appendChild(el("span", "verdict verdict--none", claim.status ? claim.status.replace(/_/g, " ") : "—"));
    return td;
  }
  let cls = "verdict--none", text = "not established";
  if (gate.certified) { cls = "verdict--pass"; text = "certified"; }
  else if (gate.any_adverse) { cls = "verdict--warn"; text = "adverse"; }
  const span = el("span", "verdict " + cls, text);
  if (claim.verdict) span.title = claim.verdict;
  td.appendChild(span);
  return td;
}

function claimsTable(claims) {
  const keys = claimDeltaKeys(claims);
  const wrap = el("div", "claims-table-wrap");
  const table = el("table", "claims-table");

  const head = el("tr", null, [el("th", null, "router arm vs baseline"), el("th", null, "eval set")]);
  keys.forEach((k) => head.appendChild(el("th", "num", "Δ " + k.replace(/^delta_/, "").replace(/_/g, " "))));
  head.appendChild(el("th", null, "gate"));
  const thead = el("thead", null, head);
  table.appendChild(thead);

  const tbody = el("tbody");
  claims.forEach((claim) => {
    const tr = el("tr");
    const armTd = el("td", null, [
      el("span", "claim-arm", claim.router || "—"),
      claim.baseline ? el("span", "claim-baseline", "vs " + claim.baseline) : null,
      el("span", "claim-family", claim.claim || ""),
    ]);
    tr.appendChild(armTd);

    // a claim that was not evaluated states why, across the metric columns,
    // rather than showing a row of em dashes
    if (claim.status && !claim.gate) {
      tr.appendChild(el("td", null, el("span", "claim-evalset", claim.evaluation_set || "—")));
      const na = el("td", "claim-na", claim.reason || claim.status.replace(/_/g, " "));
      na.setAttribute("colspan", String(keys.length + 1));
      tr.appendChild(na);
      tbody.appendChild(tr);
      return;
    }

    tr.appendChild(el("td", null, el("span", "claim-evalset", claim.evaluation_set || "—")));
    keys.forEach((k) => tr.appendChild(deltaCell(claim[k])));
    tr.appendChild(gateCell(claim));
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);
  wrap.appendChild(table);
  return wrap;
}

// ---- scatter label placement ------------------------------------------------
// Fourteen operating points cluster inside a narrow macro-F1 band, so the naive
// "draw the name up and to the right of every marker" rule stacked five labels
// on top of each other and on top of the lines. Greedy placement: try candidate
// offsets around the marker in preference order, take the first that collides
// with nothing already placed and stays inside the plot; if every candidate
// collides, step further out. A label that ends up away from its default
// position gets a hairline leader back to its marker.
const LABEL_CANDIDATES = [
  [9, -7, "start"], [9, 12, "start"], [-9, -7, "end"], [-9, 12, "end"],
  [9, -19, "start"], [9, 24, "start"], [-9, -19, "end"], [-9, 24, "end"],
  [9, 25, "start"], [-9, 25, "end"], [9, -31, "start"], [-9, -31, "end"],
];
const LABEL_CHAR_W = 6.5;
const LABEL_H = 12;

function placeLabels(items, bounds) {
  const placed = [];
  const hit = (b) => placed.some((p) =>
    b.x0 < p.x1 && b.x1 > p.x0 && b.y0 < p.y1 && b.y1 > p.y0);

  // longest names first: they are the hardest to fit and the most damaging to
  // clip, so they get first pick of the free space
  const order = items.map((it, i) => i).sort((a, b) => items[b].text.length - items[a].text.length);

  return order.map((i) => {
    const it = items[i];
    const w = it.text.length * LABEL_CHAR_W;
    let best = null;
    for (let ring = 0; ring < 4 && !best; ring++) {
      for (const [dx, dy, anchor] of LABEL_CANDIDATES) {
        const oy = dy + Math.sign(dy || 1) * ring * 13;
        const x = it.cx + dx;
        const y = it.cy + oy;
        const x0 = anchor === "end" ? x - w : x;
        const box = { x0, x1: x0 + w, y0: y - LABEL_H + 3, y1: y + 3 };
        if (box.x0 < bounds.x0 - 4 || box.x1 > bounds.x1 + 4) continue;
        if (box.y0 < bounds.y0 - 2 || box.y1 > bounds.y1 + 2) continue;
        if (hit(box)) continue;
        placed.push(box);
        best = { i, x, y, anchor, displaced: ring > 0 || Math.abs(oy) > 13 };
        break;
      }
    }
    if (!best) {
      // nothing fits: fall back to the default corner rather than dropping the
      // name — an unlabelled point is worse than a tight one
      const x = it.cx + 9;
      best = { i, x, y: it.cy - 7, anchor: "start", displaced: false };
    }
    return best;
  });
}

function drawFrontierChart(data) {
  const svg = document.getElementById("frontier-svg");
  svg.innerHTML = "";
  const legend = document.getElementById("frontier-legend");
  legend.innerHTML = "";

  const W = 760, H = 470;
  const margin = { top: 26, right: 24, bottom: 52, left: 62 };
  const x0 = margin.left, x1 = W - margin.right, y0 = margin.top, y1 = H - margin.bottom;

  const points = data.points.filter((p) => p.cost_per_1k_usd && p.macro_f1);
  if (!points.length) {
    svg.appendChild(svgEl("text", { x: 20, y: 30 }));
    return;
  }

  const costs = points.flatMap((p) => [p.cost_per_1k_usd.ci_lo ?? p.cost_per_1k_usd.point, p.cost_per_1k_usd.ci_hi ?? p.cost_per_1k_usd.point]).filter((v) => v > 0);
  const f1s = points.flatMap((p) => [p.macro_f1.ci_lo ?? p.macro_f1.point, p.macro_f1.ci_hi ?? p.macro_f1.point]);

  const xDomain = [Math.min(...costs) * 0.6, Math.max(...costs) * 1.6];
  const yDomain = [Math.max(0, Math.min(...f1s) - 0.05), Math.min(1, Math.max(...f1s) + 0.05)];

  const xScale = makeLogScale(xDomain, [x0, x1]);
  const yScale = makeLinearScale(yDomain, [y1, y0]);

  // Log ticks at decade boundaries only gave this chart a single labelled tick
  // ("$1,000"): the measured costs span well under one decade, so the axis had
  // nothing to read against. Subdivide each decade until at least three ticks
  // land inside the domain.
  const xTicks = logTicks(xDomain).map((v) => ({
    x: xScale(v), label: "$" + v.toLocaleString(undefined, { maximumFractionDigits: 2 }),
  }));
  const yTicks = niceTicks(yDomain, 5).map((v) => ({ y: yScale(v), label: v.toFixed(2) }));

  drawAxes(svg, { x0, x1, y0, y1, xTicks, yTicks, xLabel: "cost per 1k complaints (USD, log scale)", yLabel: "macro-F1" });

  const palette = Object.values(OKABE_ITO);
  let colorIdx = 0;
  const colorByKind = {};

  // whiskers under markers under labels, so nothing important is covered
  const placedGeom = [];
  points.forEach((p) => {
    const key = p.kind || "single";
    if (!(key in colorByKind)) {
      colorByKind[key] = palette[colorIdx % palette.length];
      colorIdx++;
    }
    const color = colorByKind[key];
    const cx = xScale(p.cost_per_1k_usd.point);
    const cy = yScale(p.macro_f1.point);
    placedGeom.push({ p, cx, cy, color, text: p.label || "" });

    if (p.cost_per_1k_usd.ci_lo !== undefined) {
      svg.appendChild(svgEl("line", {
        class: "chart-whisker", x1: xScale(p.cost_per_1k_usd.ci_lo), x2: xScale(p.cost_per_1k_usd.ci_hi),
        y1: cy, y2: cy, stroke: color,
      }));
    }
    if (p.macro_f1.ci_lo !== undefined) {
      svg.appendChild(svgEl("line", {
        class: "chart-whisker", x1: cx, x2: cx,
        y1: yScale(p.macro_f1.ci_lo), y2: yScale(p.macro_f1.ci_hi), stroke: color,
      }));
    }
  });

  // Fourteen operating points inside a narrow macro-F1 band cannot carry
  // fourteen spelled-out names without the names covering the data — the
  // previous version stacked five of them on one another. Keyed markers instead:
  // a numeral on the plot, the full name in the legend below, numbered left to
  // right by cost so the key reads in the same order as the axis.
  placedGeom.sort((a, b) => a.cx - b.cx);
  placedGeom.forEach((g, i) => { g.key = String(i + 1); g.text = g.key; });

  placedGeom.forEach(({ p, cx, cy, color }) => {
    const marker = p.kind === "router"
      ? svgEl("rect", { x: cx - 4, y: cy - 4, width: 8, height: 8, fill: color, stroke: "var(--paper)", "stroke-width": 1 })
      : svgEl("circle", { cx, cy, r: 4, fill: color, stroke: "var(--paper)", "stroke-width": 1 });
    marker.style.cursor = "pointer";
    marker.addEventListener("click", () => {
      if (p.run_id) openDrawerForRun(p.run_id);
    });
    const titleEl = svgEl("title", {});
    titleEl.textContent = `${p.label}\ncost: ${metricStr(p.cost_per_1k_usd)}\nmacro-F1: ${metricStr(p.macro_f1)}\n${p.cost_basis || ""}`;
    marker.appendChild(titleEl);
    svg.appendChild(marker);
  });

  placeLabels(placedGeom, { x0, x1, y0, y1 }).forEach(({ i, x, y, anchor, displaced }) => {
    const g = placedGeom[i];
    if (displaced) {
      svg.appendChild(svgEl("line", {
        class: "chart-leader",
        x1: g.cx + (anchor === "end" ? -5 : 5), y1: g.cy,
        x2: x + (anchor === "end" ? 2 : -2), y2: y - 3,
      }));
    }
    const labelText = svgEl("text", {
      class: "chart-point-key", x, y, "text-anchor": anchor, fill: g.color,
    });
    labelText.textContent = g.key;
    svg.appendChild(labelText);
  });

  placedGeom.forEach((g) => {
    const row = legendRow(g.color, `${g.key}  ${g.p.label}`, g.p.kind === "router" ? "square" : "dot");
    row.style.cursor = g.p.run_id ? "pointer" : "";
    if (g.p.run_id) row.addEventListener("click", () => openDrawerForRun(g.p.run_id));
    legend.appendChild(row);
  });
  (data.pending_points || []).forEach((pp) => {
    legend.appendChild(pendingLegendRow("pending Tier B — " + (pp.label || pp.slot)));
  });
}

// ----------------------------------------------------------------------------
// PANEL 3: Router policy builder
// ----------------------------------------------------------------------------

let policyState = null;

async function initPolicyBuilder() {
  const bannerSlot = document.getElementById("policy-banner-slot");
  const body = document.getElementById("policy-body");
  bannerSlot.innerHTML = "";

  const data = await loadJSON("policies.json");
  if (isMissing(data) || !Array.isArray(data.policies)) {
    bannerSlot.appendChild(missingBanner("policies.json"));
    return;
  }

  body.style.display = "";
  policyState = data;

  const defaults = data.cost_defaults || {};
  const sMisroute = document.getElementById("slider-misroute");
  const sHuman = document.getElementById("slider-human");
  const sApi = document.getElementById("slider-api");
  const sTau = document.getElementById("slider-tau");

  sMisroute.value = defaults.c_misroute ?? 6;
  sHuman.value = defaults.c_human ?? 2.5;
  sApi.value = 1;

  const grid = (data.tau_sweep_a_to_human && data.tau_sweep_a_to_human.grid) || [];
  const frozenTauPolicy = data.policies.find((p) => p.key === "a_to_human");
  const defaultTau = frozenTauPolicy && frozenTauPolicy.tau ? (frozenTauPolicy.tau.value ?? frozenTauPolicy.tau.point) : 0.5;
  sTau.value = defaultTau ?? 0.5;

  [sMisroute, sHuman, sApi, sTau].forEach((s) => s.addEventListener("input", renderPolicyList));

  document.getElementById("policy-reset-btn").addEventListener("click", () => {
    sMisroute.value = defaults.c_misroute ?? 6;
    sHuman.value = defaults.c_human ?? 2.5;
    sApi.value = 1;
    sTau.value = defaultTau ?? 0.5;
    renderPolicyList();
  });

  document.getElementById("policy-frozen-tau-note").textContent = data.frozen_tau_note || "";

  renderPolicyList();
}

function nearestGridPoint(grid, tau) {
  if (!grid.length) return null;
  let best = grid[0];
  let bestDist = Math.abs(grid[0].tau - tau);
  grid.forEach((g) => {
    const d = Math.abs(g.tau - tau);
    if (d < bestDist) { best = g; bestDist = d; }
  });
  return best;
}

function isAtDefaults(c_misroute, c_human, api_mult, defaults) {
  const eps = 1e-6;
  return Math.abs(c_misroute - (defaults.c_misroute ?? 6)) < eps &&
    Math.abs(c_human - (defaults.c_human ?? 2.5)) < eps &&
    Math.abs(api_mult - 1) < eps;
}

function renderPolicyList() {
  const data = policyState;
  if (!data) return;

  const c_misroute = parseFloat(document.getElementById("slider-misroute").value);
  const c_human = parseFloat(document.getElementById("slider-human").value);
  const api_mult = parseFloat(document.getElementById("slider-api").value);
  const tau = parseFloat(document.getElementById("slider-tau").value);

  document.getElementById("slider-misroute-out").textContent = "$" + c_misroute.toFixed(2);
  document.getElementById("slider-human-out").textContent = "$" + c_human.toFixed(2);
  document.getElementById("slider-api-out").textContent = api_mult.toFixed(2) + "×";
  document.getElementById("slider-tau-out").textContent = tau.toFixed(2);

  const grid = (data.tau_sweep_a_to_human && data.tau_sweep_a_to_human.grid) || [];
  const nearest = nearestGridPoint(grid, tau);
  const tauReadout = document.getElementById("policy-tau-readout");
  tauReadout.innerHTML = "";
  if (nearest) {
    tauReadout.appendChild(el("div", null, [
      el("strong", null, "a_to_human tau sweep (snapped to grid): "),
      `tau=${fmtNum(nearest.tau, 3)}, coverage=${fmtPct(nearest.coverage)}, acc_answered=${fmtNum(nearest.acc_answered, 3)}, human_rate=${fmtPct(nearest.human_rate)}`,
    ]));
  } else {
    tauReadout.appendChild(el("div", null, "no tau sweep grid available"));
  }

  const atDefaults = isAtDefaults(c_misroute, c_human, api_mult, data.cost_defaults || {});

  // compute recomputed costs, sort ascending
  const computed = data.policies.map((p) => {
    const rates = p.rates || {};
    const p_error = p.p_error_machine ?? 0;
    const apiCostPoint = p.api_cost_per_1k_usd ? p.api_cost_per_1k_usd.point : 0;
    const derivedCost = 1000 * c_misroute * p_error * (1 - (rates.human ?? 0))
      + api_mult * apiCostPoint
      + 1000 * c_human * (rates.human ?? 0);
    return { policy: p, derivedCost };
  });
  computed.sort((a, b) => a.derivedCost - b.derivedCost);

  const maxCost = Math.max(...computed.map((c) => c.derivedCost), 1e-6);

  const list = document.getElementById("policy-list");
  list.innerHTML = "";
  computed.forEach(({ policy, derivedCost }) => {
    const row = el("div", "policy-bar-row");
    const labelDiv = el("div", null, [
      policy.label || policy.key,
      policy.headline ? el("span", "tag headline", "headline") : null,
      el("br"),
      el("span", "panel-desc", `macro-F1: ${metricStr(policy.macro_f1_system)}`),
    ]);
    row.appendChild(labelDiv);

    const trackWrap = el("div", null);
    const track = el("div", "policy-bar-track");
    const fill = el("div", "policy-bar-fill");
    fill.style.width = fmtPct(derivedCost / maxCost, 0);
    track.appendChild(fill);
    trackWrap.appendChild(track);
    row.appendChild(trackWrap);

    let costLabel, tag;
    if (atDefaults && policy.expected_cost_per_1k && policy.expected_cost_per_1k.total) {
      const t = policy.expected_cost_per_1k.total;
      costLabel = `${fmtUsd(t.point, 2)} [${fmtUsd(t.ci_lo, 2)}, ${fmtUsd(t.ci_hi, 2)}]`;
      tag = el("span", "tag measured", "measured");
    } else {
      costLabel = fmtUsd(derivedCost, 2);
      tag = el("span", "tag derived", "derived (client re-solve)");
    }
    const costCell = el("div", null, [costLabel, el("br"), tag]);
    if (policy.run_refs && policy.run_refs.length) {
      costCell.appendChild(el("br"));
      policy.run_refs.forEach((rid, i) => {
        // a real space between chips, so a row of them can wrap in a narrow column
        if (i > 0) costCell.appendChild(document.createTextNode(" "));
        costCell.appendChild(runChip(rid));
      });
    }
    row.appendChild(costCell);

    list.appendChild(row);
  });
}

// ----------------------------------------------------------------------------
// PANEL 4: Drift timeline
// ----------------------------------------------------------------------------

async function initDrift() {
  const bannerSlot = document.getElementById("drift-banner-slot");
  const body = document.getElementById("drift-body");
  bannerSlot.innerHTML = "";

  const data = await loadJSON("drift.json");
  if (isMissing(data) || !data.summary || !data.summary.series || typeof data.summary.series !== "object") {
    bannerSlot.appendChild(missingBanner("drift.json"));
    return;
  }

  body.style.display = "";
  drawDriftF1Chart(data);
  drawDriftEscalationChart(data);
}

// drift.json summary.series is a dict keyed by evidence arm ("logged": per-tier raw
// eval records with a `tier`/`macro_f1` field; "escalation": per-router-policy
// records with `policy`/`macro_f1_system`/`escalation_rate`). We flatten each arm's
// record list into named line-series. Escalation records are keyed by the FULL arm
// identity (policy + terminal model + tau-fit dataset), never by policy alone: the same
// policy ships more than one arm (a_to_human and a_to_b each carry a full_cal-tau and a
// paired-subset-tau arm; a_to_c has Haiku- and Sonnet-terminal arms), and collapsing
// arms onto one key would zigzag two different measurements through one polyline.

function driftXScale(sliceOrder, x0, x1) {
  const idx = {};
  sliceOrder.forEach((s, i) => { idx[s] = i; });
  return { scale: makeLinearScale([0, Math.max(sliceOrder.length - 1, 1)], [x0, x1]), idx };
}

function driftArmKey(rec) {
  if (!rec.policy) return null;
  const terminal = rec.terminal_model && rec.terminal_model !== "human"
    ? "→" + rec.terminal_model : "";
  const dataset = rec.dataset ? ` [τ: ${rec.dataset}]` : "";
  return rec.policy + terminal + dataset;
}

function buildDriftLineSeries(summary, valueField, groupOf, labelOf) {
  const seriesDict = summary.series || {};
  const bySeriesKey = {};
  Object.values(seriesDict).forEach((records) => {
    if (!Array.isArray(records)) return;
    records.forEach((rec) => {
      const groupKey = groupOf(rec);
      if (!groupKey) return;
      const val = rec[valueField];
      if (val === undefined || val === null) return;
      if (!bySeriesKey[groupKey]) bySeriesKey[groupKey] = { label: (labelOf && labelOf(rec)) || groupKey, points: [] };
      bySeriesKey[groupKey].points.push({ slice: rec.slice, value: typeof val === "object" ? val.point : val });
    });
  });
  return bySeriesKey;
}

function drawDriftF1Chart(data) {
  const svg = document.getElementById("drift-f1-svg");
  const legendWrap = document.getElementById("drift-f1-legend");
  svg.innerHTML = "";
  legendWrap.innerHTML = "";

  const summary = data.summary;
  const sliceOrder = summary.slice_order || Object.keys(summary.slice_labels || {});
  const sliceLabels = summary.slice_labels || {};

  const tierSeries = buildDriftLineSeries(summary, "macro_f1", (r) => r.tier, (r) => r.tier_display);
  const policySeries = buildDriftLineSeries(summary, "macro_f1_system", driftArmKey);
  const allSeries = { ...tierSeries, ...policySeries };
  const seriesKeys = Object.keys(allSeries);
  if (!seriesKeys.length) return;

  const W = 760, H = 360;
  const margin = { top: 26, right: 24, bottom: 52, left: 54 };
  const x0 = margin.left, x1 = W - margin.right, y0 = margin.top, y1 = H - margin.bottom;

  const { scale: xScale, idx } = driftXScale(sliceOrder, x0, x1);
  // Crop the y-axis to the measured band. Every arm lives between roughly 0.65
  // and 0.85, so a fixed 0–1 axis squashed ten series into the top fifth of the
  // plot and made the drift — the entire point of the exhibit — invisible. The
  // caption states the crop so no one reads the slopes as steeper than they are.
  const yDomain = driftYDomain(allSeries, idx);
  const yScale = makeLinearScale(yDomain, [y1, y0]);

  const xTicks = sliceOrder.map((s) => ({ x: xScale(idx[s]), label: sliceLabels[s] || s }));
  const yTicks = niceTicks(yDomain, 5).map((v) => ({ y: yScale(v), label: v.toFixed(2) }));
  drawAxes(svg, { x0, x1, y0, y1, xTicks, yTicks, xLabel: "time slice", yLabel: "macro-F1" });

  const palette = Object.values(OKABE_ITO);
  seriesKeys.forEach((key, i) => {
    const color = palette[i % palette.length];
    // eight colourblind-safe hues, ten arms: past the palette a dash pattern
    // carries the identity so two arms never share both colour and stroke
    const dash = i >= palette.length ? "5,3" : null;
    const s = allSeries[key];
    const pts = s.points.filter((pt) => pt.slice in idx).sort((a, b) => idx[a.slice] - idx[b.slice]);
    const pathPts = pts.map((pt) => `${xScale(idx[pt.slice])},${yScale(pt.value)}`).join(" ");
    if (pts.length) {
      const line = svgEl("polyline", { points: pathPts, fill: "none", stroke: color, "stroke-width": 1.75 });
      if (dash) line.setAttribute("stroke-dasharray", dash);
      svg.appendChild(line);
      pts.forEach((pt) => {
        svg.appendChild(svgEl("circle", {
          cx: xScale(idx[pt.slice]), cy: yScale(pt.value), r: 2.5, fill: color,
        }));
      });
    }
    const row = legendRow(color, s.label);
    if (dash) row.querySelector(".legend-swatch").style.borderTopStyle = "dashed";
    legendWrap.appendChild(row);
  });

  (data.pending_series || []).forEach((ps) => {
    legendWrap.appendChild(pendingLegendRow(ps.label || ("pending Tier B — " + (ps.slot || ""))));
  });

  // Annotation lines. drift.json gives each event an `x` that may be either a
  // slice KEY ("test_drift_2026h1") or the slice's display LABEL ("2026-H1"),
  // and the old lookup only tried the key — so the panel promised "annotated
  // events" and silently drew none of them. Resolve against both. An event that
  // matches neither (a calendar date on a yearly axis) is not placed at an
  // invented position; it is listed under the figure instead of vanishing.
  const unplaced = [];
  (data.annotations || []).forEach((a) => {
    const key = a.x in idx
      ? a.x
      : Object.keys(idx).find((k) => (sliceLabels[k] || k) === a.x);
    if (key === undefined) { unplaced.push(a); return; }
    const ax = xScale(idx[key]);
    svg.appendChild(svgEl("line", {
      x1: ax, x2: ax, y1: y0, y2: y1, stroke: "var(--rule-strong)", "stroke-dasharray": "3,3",
    }));
    // annotation text hugs the top rule, above every series, and flips to the
    // left of its rule when the rule is near the right edge
    const nearRight = ax > x1 - 190;
    const t = svgEl("text", {
      class: "chart-annotation-label", x: ax + (nearRight ? -5 : 5), y: y0 - 9,
      "text-anchor": nearRight ? "end" : "start", fill: "var(--muted)",
    });
    t.textContent = a.label;
    svg.appendChild(t);
  });

  if (unplaced.length) {
    const note = el("div", "chart-events");
    note.appendChild(el("span", "meta-label", "events off this axis"));
    unplaced.forEach((a) => {
      note.appendChild(el("span", "chart-event-item", `${a.x} — ${a.label}`));
    });
    legendWrap.parentNode.insertBefore(note, legendWrap.nextSibling);
  }
}

// y-domain from the data, padded, snapped outward to a round step. Never
// inverted and never zero-height, even if a series is flat.
function driftYDomain(allSeries, idx) {
  const vals = [];
  Object.values(allSeries).forEach((s) => s.points.forEach((pt) => {
    if (pt.slice in idx && Number.isFinite(pt.value)) vals.push(pt.value);
  }));
  if (!vals.length) return [0, 1];
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const pad = Math.max((hi - lo) * 0.14, 0.02);
  lo = Math.max(0, Math.floor((lo - pad) * 20) / 20);
  hi = Math.min(1, Math.ceil((hi + pad) * 20) / 20);
  return hi > lo ? [lo, hi] : [Math.max(0, lo - 0.05), Math.min(1, lo + 0.05)];
}

function niceTicks([lo, hi], target) {
  const raw = (hi - lo) / target;
  const mag = Math.pow(10, Math.floor(Math.log10(raw)));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || mag * 10;
  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) {
    ticks.push(Math.round(v / step) * step);
  }
  return ticks;
}

function drawDriftEscalationChart(data) {
  const svg = document.getElementById("drift-esc-svg");
  const legendWrap = document.getElementById("drift-esc-legend");
  svg.innerHTML = "";
  legendWrap.innerHTML = "";

  const summary = data.summary;
  const sliceOrder = summary.slice_order || Object.keys(summary.slice_labels || {});
  const sliceLabels = summary.slice_labels || {};

  const escSeries = buildDriftLineSeries(summary, "escalation_rate", driftArmKey);
  const seriesKeys = Object.keys(escSeries);
  if (!seriesKeys.length) {
    const t = svgEl("text", { class: "chart-tick", x: 10, y: 22 });
    t.textContent = "no escalation_rate field present in drift.json summary series";
    svg.appendChild(t);
    return;
  }

  const W = 760, H = 260;
  const margin = { top: 22, right: 24, bottom: 52, left: 54 };
  const x0 = margin.left, x1 = W - margin.right, y0 = margin.top, y1 = H - margin.bottom;

  const { scale: xScale, idx } = driftXScale(sliceOrder, x0, x1);
  // A rate keeps its zero baseline — the distance from "never escalates" is the
  // quantity of interest — but the top is cropped to the measured maximum.
  const escMax = Math.max(...Object.values(escSeries).flatMap((s) =>
    s.points.filter((pt) => pt.slice in idx).map((pt) => pt.value)), 0.1);
  const yDomain = [0, Math.min(1, Math.ceil(escMax * 1.15 * 20) / 20)];
  const yScale = makeLinearScale(yDomain, [y1, y0]);
  const xTicks = sliceOrder.map((s) => ({ x: xScale(idx[s]), label: sliceLabels[s] || s }));
  const yTicks = niceTicks(yDomain, 4).map((v) => ({ y: yScale(v), label: v.toFixed(2) }));
  drawAxes(svg, { x0, x1, y0, y1, xTicks, yTicks, xLabel: "time slice", yLabel: "escalation rate" });

  const palette = Object.values(OKABE_ITO);
  seriesKeys.forEach((key, i) => {
    const color = palette[i % palette.length];
    const dash = i >= palette.length ? "5,3" : null;
    const s = escSeries[key];
    const pts = s.points.filter((pt) => pt.slice in idx).sort((a, b) => idx[a.slice] - idx[b.slice]);
    const pathPts = pts.map((pt) => `${xScale(idx[pt.slice])},${yScale(pt.value)}`).join(" ");
    if (pts.length) {
      const line = svgEl("polyline", { points: pathPts, fill: "none", stroke: color, "stroke-width": 1.75 });
      if (dash) line.setAttribute("stroke-dasharray", dash);
      svg.appendChild(line);
      pts.forEach((pt) => {
        svg.appendChild(svgEl("circle", {
          cx: xScale(idx[pt.slice]), cy: yScale(pt.value), r: 2.5, fill: color,
        }));
      });
    }
    const row = legendRow(color, s.label);
    if (dash) row.querySelector(".legend-swatch").style.borderTopStyle = "dashed";
    legendWrap.appendChild(row);
  });
}

// ----------------------------------------------------------------------------
// PANEL 5: Calibration
// ----------------------------------------------------------------------------

async function initCalibration() {
  const bannerSlot = document.getElementById("calibration-banner-slot");
  const grid = document.getElementById("calibration-grid");
  const noteWrap = document.getElementById("calibration-tier-c-note");
  bannerSlot.innerHTML = "";
  grid.innerHTML = "";
  noteWrap.innerHTML = "";

  const data = await loadJSON("calibration.json");
  if (isMissing(data) || !Array.isArray(data.exhibits)) {
    bannerSlot.appendChild(missingBanner("calibration.json"));
    return;
  }

  if (data.tier_c_note) {
    const card = el("div", "info-card");
    card.appendChild(el("div", null, data.tier_c_note.text));
    (data.tier_c_note.run_ids || []).forEach((rid) => card.appendChild(runChip(rid)));
    noteWrap.appendChild(card);
  }

  data.exhibits.forEach((ex) => {
    grid.appendChild(reliabilityCard(ex));
  });

  (data.pending || []).forEach((p) => {
    const card = el("div", "card pending");
    card.appendChild(el("h4", null, p.slot));
    card.appendChild(el("div", null, "pending Tier B backfill"));
    grid.appendChild(card);
  });
}

// Reliability diagram as a small multiple: title, quiet provenance line, plot,
// metrics under a hairline. No card border — the grid gutter separates them,
// and a box around a box around a plot was three frames for one exhibit.
function reliabilityCard(ex) {
  const fig = el("figure", "figure figure--fluid");

  const header = el("figcaption", "exhibit-header");
  header.appendChild(el("span", "exhibit-title", ex.label || ex.key));
  const meta = el("div", "exhibit-meta");
  meta.appendChild(evidenceBadge(ex.evidence_class));
  if (ex.run_id) meta.appendChild(runChip(ex.run_id));
  header.appendChild(meta);
  fig.appendChild(header);

  // The x-axis name used to be drawn 2px below the bottom of the viewBox and was
  // clipped on every exhibit; the box is now tall enough to hold it, and the
  // left margin clears the rotated y-axis name from the tick values.
  const W = 320, H = 248;
  const svg = svgEl("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img" });
  svg.setAttribute("aria-label", `Reliability diagram — ${ex.label || ex.key}`);
  const bins = ex.bins || [];
  const margin = { top: 12, right: 8, bottom: 46, left: 44 };
  const x0 = margin.left, x1 = W - margin.right, y0 = margin.top, y1 = H - margin.bottom;
  const xScale = makeLinearScale([0, 1], [x0, x1]);
  const yScale = makeLinearScale([0, 1], [y1, y0]);

  drawAxes(svg, {
    x0, x1, y0, y1,
    xTicks: [0, 0.5, 1].map((v) => ({ x: xScale(v), label: v.toFixed(1) })),
    yTicks: [0, 0.5, 1].map((v) => ({ y: yScale(v), label: v.toFixed(1) })),
    xLabel: "confidence", yLabel: "accuracy",
  });

  // perfect-calibration reference
  svg.appendChild(svgEl("line", {
    x1: xScale(0), y1: yScale(0), x2: xScale(1), y2: yScale(1),
    stroke: "var(--rule-strong)", "stroke-dasharray": "3,3",
  }));

  const barW = (x1 - x0) / Math.max(bins.length, 1);
  bins.forEach((b, i) => {
    const bx = x0 + i * barW;
    const rect = svgEl("rect", {
      x: bx + 0.75, y: yScale(b.acc ?? 0),
      width: Math.max(barW - 1.5, 1), height: y1 - yScale(b.acc ?? 0),
      fill: OKABE_ITO.skyblue, opacity: 0.9,
    });
    const titleEl = svgEl("title", {});
    titleEl.textContent = `bin [${fmtNum(b.lo, 2)}, ${fmtNum(b.hi, 2)}) n=${b.n} conf_mean=${fmtNum(b.conf_mean, 3)} acc=${fmtNum(b.acc, 3)}`;
    rect.appendChild(titleEl);
    svg.appendChild(rect);
  });

  const body = el("div", "figure-body");
  body.appendChild(svg);
  fig.appendChild(body);

  fig.appendChild(el("div", "exhibit-metrics",
    `ECE ${metricStr(ex.ece)} · Brier ${metricStr(ex.brier)}`));
  return fig;
}

// ----------------------------------------------------------------------------
// PANEL 6: Receipts / all runs
// ----------------------------------------------------------------------------

let receiptsTableState = { rows: [], sortKey: "timestamp_utc", sortDir: -1 };

async function initReceiptsPanel() {
  const bannerSlot = document.getElementById("receipts-banner-slot");
  const tableWrap = document.getElementById("receipts-table-wrap");
  bannerSlot.innerHTML = "";

  const data = await loadJSON("runs_index.json");
  if (isMissing(data)) {
    bannerSlot.appendChild(missingBanner("runs_index.json"));
    return;
  }

  const rows = Object.entries(data).map(([runId, record]) => ({ run_id: runId, ...record }));
  receiptsTableState.rows = rows;
  tableWrap.style.display = "";

  document.querySelectorAll("#receipts-table th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (receiptsTableState.sortKey === key) {
        receiptsTableState.sortDir *= -1;
      } else {
        receiptsTableState.sortKey = key;
        receiptsTableState.sortDir = 1;
      }
      renderReceiptsTable();
    });
  });

  renderReceiptsTable();
}

function renderReceiptsTable() {
  const { rows, sortKey, sortDir } = receiptsTableState;
  const sorted = [...rows].sort((a, b) => {
    const av = a[sortKey], bv = b[sortKey];
    if (av === undefined || av === null) return 1;
    if (bv === undefined || bv === null) return -1;
    if (av < bv) return -1 * sortDir;
    if (av > bv) return 1 * sortDir;
    return 0;
  });

  const tbody = document.getElementById("receipts-table-body");
  tbody.innerHTML = "";
  sorted.forEach((r) => {
    const tr = el("tr");
    tr.appendChild(el("td", null, r.timestamp_utc || "—"));
    tr.appendChild(el("td", null, r.tier || "—"));
    tr.appendChild(el("td", null, r.model_label || r.config_name || "—"));
    tr.appendChild(el("td", null, r.slice || "—"));
    tr.appendChild(el("td", null, fmtUsd(r.cost_usd, 5)));
    tr.appendChild(el("td", null, fmtNum(r.wall_clock_seconds, 1)));
    const runTd = el("td");
    runTd.appendChild(runChip(r.run_id));
    tr.appendChild(runTd);
    tbody.appendChild(tr);
  });
}

// ----------------------------------------------------------------------------
// panel 7: case study
// ----------------------------------------------------------------------------
//
// Pure rendering. Every string and every number comes out of case_study.json, which the
// build produced by copying from results/ — the page has no formatting opinions of its
// own, so a number cannot differ here from the artifact it was copied from. `display`
// strings are already embedded in the paragraphs; the `numbers` array is rendered as the
// per-section provenance table behind a disclosure, so a reader can go from a sentence to
// its source path, run chip and repro command without leaving the page.

function reproLine(commands) {
  if (!commands || !commands.length) return null;
  const wrap = el("div", "cs-repro");
  commands.forEach((cmd) => wrap.appendChild(el("code", "cs-cmd", cmd)));
  return wrap;
}

function receiptsLine(section) {
  const ids = section.run_ids || [];
  const cmds = section.repro || [];
  if (!ids.length && !cmds.length) return null;
  const wrap = el("div", "cs-receipts");
  wrap.appendChild(el("span", "cs-receipts-label", "receipts"));
  if (ids.length) {
    const chips = el("span", "cs-chips");
    ids.forEach((id) => {
      const chip = runChip(id);
      if (chip) chips.appendChild(chip);
    });
    wrap.appendChild(chips);
  }
  const repro = reproLine(cmds);
  if (repro) wrap.appendChild(repro);
  return wrap;
}

function numbersTable(numbers) {
  if (!numbers || !numbers.length) return null;
  const details = el("details", "cs-numbers");
  details.appendChild(el("summary", null, `${numbers.length} declared numbers — source, run, repro`));
  const table = el("table", "kv-table cs-numbers-table");
  const head = el("tr", null, [
    el("td", "k", "display"), el("td", "k", "what"), el("td", "k", "basis"),
    el("td", "k", "source"),
  ]);
  table.appendChild(head);
  numbers.forEach((n) => {
    const srcCell = el("td");
    const src = sourceChip(n.source);
    if (src) srcCell.appendChild(src);
    (n.run_ids || []).forEach((id) => {
      const chip = runChip(id);
      if (chip) srcCell.appendChild(chip);
    });
    if (n.repro) srcCell.appendChild(el("code", "cs-cmd", n.repro));
    if (n.note) srcCell.appendChild(el("div", "cs-note", n.note));
    table.appendChild(el("tr", null, [
      el("td", "cs-display", n.display),
      el("td", null, n.label),
      el("td", null, [evidenceBadge(n.evidence_class), el("span", "cs-basis", n.basis)]),
      srcCell,
    ]));
  });
  details.appendChild(table);
  return details;
}

function pendingCard(pending) {
  const card = el("div", "card pending");
  card.appendChild(el("span", "evidence-badge derived", "pending"));
  card.appendChild(el("div", "cs-pending-label", pending.label || pending.slot));
  return card;
}

function renderNarrativeSection(section, pageTitle) {
  const card = el("div", "card cs-section");
  // the opening section repeats the page title in the payload; printing it twice,
  // once as the panel heading and again as the first subhead, is a rendering
  // artefact rather than content, so the duplicate subhead is suppressed
  if (section.title && section.title !== pageTitle) {
    card.appendChild(el("h3", "cs-heading", section.title));
  }
  (section.paragraphs || []).forEach((p) => card.appendChild(el("p", "cs-para", p)));
  (section.gaps || []).forEach((g) => {
    card.appendChild(el("div", "info-card cs-gap", [
      el("strong", null, "not shown here: "), g,
    ]));
  });
  const receipts = receiptsLine(section);
  if (receipts) card.appendChild(receipts);
  const numbers = numbersTable(section.numbers);
  if (numbers) card.appendChild(numbers);
  if (section.pending) card.appendChild(pendingCard(section.pending));
  return card;
}

function renderVerificationSection(section) {
  const card = el("div", "card cs-section");
  card.appendChild(el("h3", "cs-heading", section.title));
  const list = el("ol", "cs-check-list");
  (section.items || []).forEach((item) => {
    const li = el("li");
    li.appendChild(el("span", "cs-check-title", item.title));
    li.appendChild(el("span", "cs-check-text", " " + item.text));
    const meta = el("div", "cs-check-meta");
    if (item.pending) meta.appendChild(el("span", "evidence-badge derived", "pending"));
    const src = sourceChip(item.source);
    if (src) meta.appendChild(src);
    (item.run_ids || []).forEach((id) => {
      const chip = runChip(id);
      if (chip) meta.appendChild(chip);
    });
    li.appendChild(meta);
    list.appendChild(li);
  });
  card.appendChild(list);
  const receipts = receiptsLine(section);
  if (receipts) card.appendChild(receipts);
  const numbers = numbersTable(section.numbers);
  if (numbers) card.appendChild(numbers);
  return card;
}

// The coursework-seed section. Same card furniture as the narrative sections, plus three
// blocks the other sections do not have: the archive's file list (rendered through
// repoRef), the lesson -> practice lineage table, and the caveat rows. Every figure inside
// it wears the `provenance` badge — self-reported, not measured here.
function renderProvenanceSection(section, repo) {
  const card = el("div", "card cs-section cs-provenance");
  card.appendChild(el("h3", "cs-heading", section.title));
  (section.paragraphs || []).forEach((p) => card.appendChild(el("p", "cs-para", p)));

  const items = section.items || [];
  if (items.length) {
    const list = el("div", "cs-seed-list");
    items.forEach((item) => {
      const row = el("div", "cs-seed");
      row.appendChild(el("div", "cs-seed-head", [
        repoRef(repo, "blob", item.path),
        el("span", "cs-seed-role", item.role),
        evidenceBadge(item.evidence_class),
      ]));
      row.appendChild(el("div", "cs-seed-text", item.text));
      list.appendChild(row);
    });
    card.appendChild(list);
  }

  const lineage = section.lineage || [];
  if (lineage.length) {
    const table = el("table", "kv-table cs-lineage-table");
    table.appendChild(el("tr", null, [
      el("td", "k", "coursework lesson"), el("td", "k", "practice it seeded here"),
    ]));
    lineage.forEach((row) => table.appendChild(el("tr", null, [
      el("td", "cs-lineage-lesson", row.lesson), el("td", null, row.practice),
    ])));
    card.appendChild(el("div", "cs-lineage", [
      el("div", "cs-block-title", "Methodology lineage"), table,
    ]));
  }

  (section.caveats || []).forEach((c) => {
    card.appendChild(el("div", "info-card cs-caveat", [el("strong", null, "caveat: "), c]));
  });
  (section.gaps || []).forEach((g) => {
    card.appendChild(el("div", "info-card cs-gap", [
      el("strong", null, "not shown here: "), g,
    ]));
  });

  const receipts = receiptsLine(section);
  if (receipts) card.appendChild(receipts);
  const numbers = numbersTable(section.numbers);
  if (numbers) card.appendChild(numbers);
  return card;
}

function renderLimitsSection(section) {
  const card = el("div", "card cs-section cs-limits");
  card.appendChild(el("h3", "cs-heading", section.title));
  const list = el("ul", "cs-limit-list");
  (section.items || []).forEach((item) => list.appendChild(el("li", null, item.text)));
  card.appendChild(list);
  return card;
}

async function initCaseStudy() {
  const bannerSlot = document.getElementById("casestudy-banner-slot");
  const body = document.getElementById("casestudy-body");
  bannerSlot.innerHTML = "";
  body.innerHTML = "";

  const data = await loadJSON("case_study.json");
  if (isMissing(data) || !Array.isArray(data.sections)) {
    bannerSlot.appendChild(missingBanner("case_study.json"));
    return;
  }

  if (data.title) document.getElementById("casestudy-title").textContent = data.title;
  document.getElementById("casestudy-source-note").textContent = data.source_note || "";

  data.sections.forEach((section) => {
    if (section.kind === "verification") body.appendChild(renderVerificationSection(section));
    else if (section.kind === "limits") body.appendChild(renderLimitsSection(section));
    else if (section.kind === "provenance") {
      body.appendChild(renderProvenanceSection(section, data.repo));
    } else body.appendChild(renderNarrativeSection(section, data.title));
  });

  const pending = data.pending || [];
  if (pending.length) {
    const wrap = el("div", "cs-section");
    wrap.appendChild(el("h3", "cs-heading", "Pending slots"));
    pending.forEach((p) => wrap.appendChild(pendingCard(p)));
    body.appendChild(wrap);
  }

  body.style.display = "";
}

// ----------------------------------------------------------------------------
// boot
// ----------------------------------------------------------------------------

function initDrawer() {
  document.getElementById("drawer-close-btn").addEventListener("click", closeDrawer);
  document.getElementById("drawer-overlay").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape") closeDrawer();
  });
}

async function boot() {
  initNav();
  initDrawer();
  await Promise.all([
    initPlayground(),
    initFrontier(),
    initPolicyBuilder(),
    initDrift(),
    initCalibration(),
    initReceiptsPanel(),
    initCaseStudy(),
  ]);
}

boot().catch((err) => {
  // last-resort visible failure, never a silent blank page
  const main = document.querySelector(".main-col");
  if (main) {
    const banner = document.createElement("div");
    banner.className = "data-missing-banner";
    banner.textContent = "Unexpected error initializing the demo: " + String(err);
    main.prepend(banner);
  }
  // eslint-disable-next-line no-console
  console.error(err);
});
