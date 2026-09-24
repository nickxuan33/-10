"""Find the terms that the top Google results for a keyword have in common.

For each keyword, SerpApi returns the Google results that a searcher in the
chosen location sees (Alberta, Canada by default). The script downloads each
result page, keeps its visible text, and scores the words and two-word phrases
that appear on at least two pages by TF-IDF.

    python tf-idf.py "alberta online casino" "legal online casino alberta"

Needs a SerpApi key in the SERPAPI_API_KEY environment variable. Writes
output/serp.csv (the results per keyword) and output/terms.csv (the scores).
"""
import argparse
import os
import sys
from pathlib import Path

import nltk
import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

SERPAPI_URL = "https://serpapi.com/search.json"
PAGE_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                              "(KHTML, like Gecko) Chrome/128.0 Safari/537.36"}
HIDDEN_TAGS = {"style", "script", "noscript", "head", "title", "meta", "[document]"}


def search_results(keyword, api_key, n_results, location, gl, hl, google_domain):
    """Return up to n_results organic Google results for keyword as (title, url) pairs."""
    results = []
    start = 0
    while len(results) < n_results:
        response = requests.get(SERPAPI_URL, timeout=30, params={
            "engine": "google",
            "q": keyword,
            "location": location,
            "gl": gl,
            "hl": hl,
            "google_domain": google_domain,
            "start": start,
            "api_key": api_key,
        })
        try:
            data = response.json()
        except ValueError:
            raise RuntimeError(f"SerpApi answered HTTP {response.status_code} without JSON") from None
        if response.status_code != 200:
            raise RuntimeError(data.get("error", f"SerpApi answered HTTP {response.status_code}"))
        # A 200 answer without organic results means Google has no more pages.
        seen = {url for _, url in results}
        page = [(r.get("title", ""), r["link"]) for r in data.get("organic_results", [])
                if "link" in r and r["link"] not in seen]
        if not page:
            break
        results.extend(page)
        start += 10  # Google returns 10 results per page
    return results[:n_results]


def get_text(url):
    """Return the visible text of a web page, or None if it can't be fetched."""
    try:
        response = requests.get(url, headers=PAGE_HEADERS, timeout=10)
        response.raise_for_status()
    except requests.RequestException:
        return None
    if "html" not in response.headers.get("Content-Type", ""):
        return None
    soup = BeautifulSoup(response.content, "html.parser")
    texts = (t.strip() for t in soup.find_all(string=True)
             if t.parent.name not in HIDDEN_TAGS and not isinstance(t, Comment))
    return " ".join(t for t in texts if t)


def tf_idf_analysis(texts, stop_words):
    """Score the words and two-word phrases that appear in at least two of the texts."""
    vectorizer = TfidfVectorizer(min_df=2, ngram_range=(1, 2), stop_words=stop_words)
    scores = pd.DataFrame(vectorizer.fit_transform(texts).toarray(),
                          columns=vectorizer.get_feature_names_out())
    terms = pd.DataFrame({
        "term": scores.columns,
        "average_tfidf": scores.mean().to_numpy(),
        "max_tfidf": scores.max().to_numpy(),
        # Share of the pages that use the term, in percent
        "frequency": ((scores > 0).mean() * 100).round().astype(int).to_numpy(),
    })
    # Terms with digits are mostly prices, dates and bonus amounts
    terms = terms[~terms["term"].str.contains(r"\d")]
    return terms.sort_values("max_tfidf", ascending=False, ignore_index=True)


def load_stop_words():
    try:
        return stopwords.words("english")
    except LookupError:
        nltk.download("stopwords", quiet=True)
        return stopwords.words("english")


def main():
    parser = argparse.ArgumentParser(
        description="Score the terms used by the top Google results for each keyword by TF-IDF.")
    parser.add_argument("keywords", nargs="+", help='keywords to search, e.g. "alberta online casino"')
    parser.add_argument("--results", type=int, default=10,
                        help="results to analyse per keyword; every 10 cost one SerpApi search (default: 10)")
    parser.add_argument("--top", type=int, default=20, help="terms to print per keyword (default: 20)")
    parser.add_argument("--location", default="Alberta, Canada",
                        help='where the search is made from (default: "Alberta, Canada")')
    parser.add_argument("--gl", default="ca", help="Google country code (default: ca)")
    parser.add_argument("--hl", default="en", help="Google interface language (default: en)")
    parser.add_argument("--google-domain", default="google.ca", help="Google domain (default: google.ca)")
    parser.add_argument("--out", default="output", help="folder for serp.csv and terms.csv (default: output)")
    args = parser.parse_args()

    api_key = os.environ.get("SERPAPI_API_KEY")
    if not api_key:
        parser.error("set the SERPAPI_API_KEY environment variable to your SerpApi key")
    stop_words = load_stop_words()

    serp_rows, term_tables = [], []
    error = None
    for keyword in args.keywords:
        try:
            results = search_results(keyword, api_key, args.results,
                                     args.location, args.gl, args.hl, args.google_domain)
        except (RuntimeError, requests.RequestException) as e:
            # Stop, but still save the keywords already paid for
            error = f"Search for {keyword!r} failed: {e}"
            break
        texts = []
        for rank, (title, url) in enumerate(results, start=1):
            text = get_text(url)
            serp_rows.append({"keyword": keyword, "rank": rank, "title": title,
                              "url": url, "fetched": bool(text)})
            if text:
                texts.append(text)

        print(f"\n{keyword}: {len(results)} results, {len(texts)} pages fetched")
        if len(texts) < 2:
            print("  Need at least two fetched pages to compare; skipped.")
            continue
        try:
            terms = tf_idf_analysis(texts, stop_words)
        except ValueError:  # no term appears on two pages
            print("  The pages have no terms in common; skipped.")
            continue
        print(terms.head(args.top).to_string(index=False, float_format="%.4f"))
        terms.insert(0, "keyword", keyword)
        term_tables.append(terms)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    serp = pd.DataFrame(serp_rows, columns=["keyword", "rank", "title", "url", "fetched"])
    all_terms = (pd.concat(term_tables) if term_tables else
                 pd.DataFrame(columns=["keyword", "term", "average_tfidf", "max_tfidf", "frequency"]))
    # utf-8-sig so that Excel opens the files with the right encoding
    serp.to_csv(out / "serp.csv", index=False, encoding="utf-8-sig")
    all_terms.to_csv(out / "terms.csv", index=False, encoding="utf-8-sig")
    print(f"\nSaved {out / 'serp.csv'} and {out / 'terms.csv'}")
    if error:
        sys.exit(error)


if __name__ == "__main__":
    main()
