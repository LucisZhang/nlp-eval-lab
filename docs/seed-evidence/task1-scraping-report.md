# Project 1 Report: Spider to Collect & Preprocess Books to Scrape Data

## Introduction
This project builds a Python spider to crawl the Books to Scrape website, extract book metadata and descriptions, and preprocess the text for downstream NLP tasks. I collected URLs and descriptions for books on pages 1–10 (200 books total), saved raw descriptions to `./test/`, and produced a cleaned, tokenized, and stemmed corpus saved to `./corpus/` and `corpus.json`.

## Project Description
### What I did
1. **Crawled index pages (1–10):**  
   Parsed each listing page to extract `(title, url)` pairs and saved them to `url.json`.
2. **Fetched book pages:**  
   Followed each URL and extracted the description from the book detail page. Saved each description as `./test/BookName.txt` using filename-safe titles.
3. **Preprocessed text:**  
   Cleaned descriptions, tokenized, and applied stemming. Saved per-book processed text to `./corpus/BookName.txt` and the full structured corpus to `corpus.json`.

### Approach / Design
- **Spider structure:**  
  - `spider-url.py` handles index pages and URL extraction.  
  - `spider-books.py` handles per-book description extraction.  
  - `preprocess.py` handles text cleaning, tokenization, and stemming.
- **HTML parsing:**  
  Used BeautifulSoup to locate the description by finding the `<div id="product_description">` and the following `<p>` element.
- **Preprocessing pipeline:**  
  - Regular expressions to remove extra whitespace and the trailing `...more`.  
  - Lowercasing and stripping non-alphanumeric symbols.  
  - Tokenization with NLTK (`word_tokenize`) and a regex fallback.  
  - Stemming with NLTK’s `PorterStemmer`.

### What did/didn’t work and how it was solved
- **Initial URL joining issue:**  
  Directly concatenating URLs caused 404s. This was fixed by normalizing the relative path and prefixing with the correct base URL (`http://books.toscrape.com/catalogue/`).
- **Filename safety:**  
  Some book titles contain characters invalid for filenames. This was resolved by sanitizing titles (replacing `\ / : * ? " < > |` with `_` and trimming extra spaces).

### Results
- `url.json`: 200 `(title, url)` pairs.  
- `./test/`: 200 raw description files.  
- `./corpus/`: 200 processed description files.  
- `corpus.json`: 200 processed entries, each `[title, token_list]`.

## Conclusions
This project successfully implemented a small but complete crawling and preprocessing pipeline. I learned how to structure a multi-stage spider, handle HTML parsing reliably, and build a text preprocessing pipeline with regex, tokenization, and stemming.  

If I were to improve the system, I would add:
- Robust retry logic and status-code handling for network errors.
- A cache to avoid re-downloading already processed pages.
- Lemmatization or more advanced normalization to improve text quality.

## References 
- Books to Scrape (dataset source).  
- Python packages: `requests`, `BeautifulSoup4`, `lxml`, `nltk`.  
