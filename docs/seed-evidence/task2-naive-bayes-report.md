# Naive Bayes Text Classification on Reuters

## 1. Introduction
This project implements a multinomial naive Bayes text classifier for the simplified Reuters dataset. The task is to classify each news document into one of five categories: `crude`, `grain`, `money-fx`, `acq`, and `earn`. The document representation is a bag-of-words model, and the classifier is trained from the provided training set and evaluated on the provided test set.

The completed system covers the full required pipeline:
- dataset preprocessing
- corpus word counting
- feature selection
- prior and posterior probability estimation with Laplace smoothing
- test-set classification
- macro F1 evaluation

## 2. Dataset and Preprocessing
The training set contains 5,787 documents and the test set contains 2,298 documents. The training class counts are:

| Class | Train documents |
| --- | ---: |
| crude | 359 |
| grain | 428 |
| money-fx | 535 |
| acq | 1617 |
| earn | 2848 |

The preprocessing stage performs the following steps:
- convert text to lowercase
- decode HTML entities such as `&lt;...&gt;`
- remove punctuation and other non-word symbols during token extraction
- tokenize the text into words
- stem each token with a Porter stemmer

The starter code asked for NLTK tokenization and stemming. In this environment, `nltk` is not installed, so the final program first tries to use NLTK and then falls back to an internal tokenizer and Porter stemmer implementation. This keeps the program runnable without external dependencies.

## 3. Method
### 3.1 Word Counting
After preprocessing, the program counts token frequencies for each class over the training corpus and writes them to `word_count.txt`. The first line stores the number of training documents in each class, and each following line stores a word and its frequency in the five classes.

The preprocessing produced 16,957 unique tokens in the training corpus.

### 3.2 Feature Selection
The required baseline model uses the top 10,000 most frequent words as features. These selected features are written to `word_dict.txt`. The first line stores the total feature-token counts in each class, and the remaining lines store each selected word with its per-class counts.

### 3.3 Probability Estimation
The classifier uses:
- prior probability: `P(c) = DocCount(c) / N_doc`
- posterior probability with Laplace smoothing:
  `P(w|c) = (Count(w,c) + 1) / (Sum_c + |V|)`

The calculated probabilities are written to `word_probability.txt`.

### 3.4 Classification
For each test document, the classifier computes the log posterior score for every class:

`log P(c|d) proportional to log P(c) + sum count(w,d) * log P(w|c)`

Only words that are present in the selected feature set are used during classification. The class with the highest score is assigned to the document, and the results are written to `classification_result.txt`.

### 3.5 Evaluation
The final score is the macro-averaged F1 across the five classes. This treats each category as a one-vs-rest classification problem and then averages the five F1 values.

## 4. Results
### 4.1 Required Baseline Result
Using the required top 10,000 features, the final macro F1 on the provided test set is:

**0.9647679093041557**

Per-class results are shown below.

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| crude | 0.945355 | 0.961111 | 0.953168 |
| grain | 0.965517 | 0.945946 | 0.955631 |
| money-fx | 0.961749 | 0.994350 | 0.977778 |
| acq | 0.950820 | 0.980282 | 0.965326 |
| earn | 0.984834 | 0.959372 | 0.971936 |

### 4.2 Optional Enhancement
I also tested a simple enhancement by increasing the number of selected features. This does not change the required submission files, but it shows the effect of feature-set size.

| Feature count | Macro F1 |
| ---: | ---: |
| 3000 | 0.9636702838478547 |
| 5000 | 0.9652496837316521 |
| 10000 | 0.9647679093041557 |
| 15000 | 0.9658523189692165 |

This suggests that the classifier is fairly stable, and slightly more features can improve performance. In this experiment, 15,000 features gave the best result among the tested settings.

## 5. Discussion
What worked:
- stemming and normalization reduced sparsity in the feature space
- Laplace smoothing prevented zero-probability failures
- using log probabilities made the classifier numerically stable
- selecting frequent features removed many rare words that add noise but little evidence

What did not work as well:
- a smaller feature set, such as 3,000 features, removed too much useful information and slightly reduced F1
- relying on raw multiplication of probabilities would be numerically unstable for long documents, so log-space scoring was necessary

## 6. Conclusion
This project successfully implements the complete naive Bayes text-classification pipeline required by the assignment. The final required model with 10,000 selected features achieved a macro F1 score of `0.9647679093041557` on the Reuters test set. The implementation is runnable from the command line and produces all required output files.

If more time were available, useful future work would include:
- trying class-balanced feature selection instead of only global frequency
- adding bigram features for domain-specific phrases
- using stopword filtering or TF-IDF style weighting for comparison
- evaluating other classifiers such as logistic regression or linear SVM as stronger baselines

## 7. Reproducibility
The main commands used are:

```bash
python3 naive-bayes.py -pps train.json train.preprocessed.json
python3 naive-bayes.py -pps test.json test.preprocessed.json
python3 naive-bayes.py -cw train.preprocessed.json word_count.txt
python3 naive-bayes.py -fs word_count.txt 10000 word_dict.txt
python3 naive-bayes.py -cp word_count.txt word_dict.txt word_probability.txt
python3 naive-bayes.py -cl word_probability.txt test.preprocessed.json classification_result.txt
python3 naive-bayes.py -f1 test.json classification_result.txt
```

## 8. AI Acknowledgement
AI assistance was used during the completion of this assignment for implementation support, debugging, and report drafting. All code, outputs, and final submission materials were reviewed and validated against the assignment requirements before submission.
