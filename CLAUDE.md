# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A single script, `tf-idf.py`, for keyword research: it scrapes Google results for a keyword, downloads the text of each result page, and ranks the terms on those pages by TF-IDF. `README.md` (Traditional Chinese) is the user-facing setup, usage, and output guide.

## Commands

```bash
pip install -r requirements.txt
python tf-idf.py   # full run: needs network, writes google.csv and myfile.csv to the current directory
```

There are no tests, linter, or build step.

## Behavior that is not obvious from a quick read

- Everything runs at module level with no `__main__` guard, so importing or running the file executes the whole pipeline, network requests included.
- `nltk.download('stopwords')` must run before `stopwords.words('english')`; the reverse order raises `LookupError` on a machine without NLTK data.
- `google_results()` parses Google's result HTML (`div.ZINbbc` elements, `/url?q=...&sa` links). When Google changes its markup or blocks the request it silently returns `[]`, and `tf_idf_analysis()` then fails with `ValueError` because `TfidfVectorizer(min_df=2)` needs at least two fetched pages.
- The keyword `'百家樂賺錢'` is hardcoded in two calls (`google_results(...)`, which writes `google.csv`, and `tf_idf_analysis(...)`); change both together.
- Tokenization is scikit-learn's default and the stopword list is English only, so Chinese text is split only at spaces and punctuation: a whole clause becomes one token.
- `tf-idf.py` uses CRLF line endings; preserve them when editing so diffs stay minimal.

## Running in Claude Code on the web

- NLTK refuses to download through the sandbox's HTTPS proxy ("Security Violation ... refusing a proxied fetch"); set `NLTK_ALLOW_PROXIED_URLOPEN=1` for the command.
- `www.google.com` has been blocked by the sandbox network policy (proxy returns 403), so a full run cannot complete there. To exercise the analysis offline, stub `requests.get` and `urllib.request.urlopen` before executing the script with `runpy.run_path`.
