# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A single CLI script, `tf-idf.py`, for SEO content research. For each keyword it gets the top Google results for a location (default Alberta, Canada) from SerpApi, downloads the result pages, and scores the words and two-word phrases the pages share by TF-IDF. `README.md` (Traditional Chinese) is the user-facing guide. `docs/alberta-ad-compliance-checklist.md` summarises Alberta's gambling-advertising rules for content written with the tool.

## Commands

```bash
pip install -r requirements.txt
SERPAPI_API_KEY=... python tf-idf.py "alberta online casino" [--results 10] [--out output]
```

There are no tests, linter, or build step. Every 10 results cost one SerpApi search credit.

## Behavior that is not obvious from a quick read

- `search_results()` pages through SerpApi with `start` (0, 10, ...) because Google has ignored `num` since September 2025. A non-200 answer is an error (bad key, quota, unsupported location); a 200 answer without `organic_results` means Google has no more results.
- If a search fails, `main()` stops but still writes the CSVs for the keywords already done, since their credits were spent, and then exits 1. Both CSVs are rewritten on every run.
- TF-IDF uses `min_df=2`, so a keyword needs at least two fetched pages that share a term; otherwise it is skipped with a message. Terms containing digits are dropped.
- Stop words are NLTK's English list; `load_stop_words()` downloads it only when missing. Chinese text is not segmented (a run of Chinese characters becomes one token), so the tool is meant for English keywords.
- CSVs are written as UTF-8 with BOM (`utf-8-sig`) so Excel on Windows opens them correctly.
- `tf-idf.py` uses CRLF line endings; preserve them when editing so diffs stay minimal.

## Running in Claude Code on the web

- serpapi.com and www.google.com are blocked by the sandbox network policy (the proxy returns 403), so real searches cannot run there. To test, replace `requests.get` with a stub that serves both the SerpApi call and the page downloads, then run the script with `runpy.run_path(..., run_name="__main__")`.
- NLTK refuses to download through the sandbox's HTTPS proxy ("Security Violation ... refusing a proxied fetch"); set `NLTK_ALLOW_PROXIED_URLOPEN=1` for the command.
