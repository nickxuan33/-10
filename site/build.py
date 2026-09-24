"""Build the Alberta casino review site as static HTML.

    python build.py            # production build into dist/
    python build.py --drafts   # preview build into dist-preview/: includes draft
                               # reviews, marks every page noindex and reports
                               # problems as warnings instead of errors

Settings live in config.yaml, operator facts in data/operators.yaml and all
text in content/ (Markdown with YAML front matter). A production build stops on
placeholder settings, missing SEO fields, broken internal links, operators that
are not verified and high-risk advertising phrases. When it stops, the previous
dist/ is left untouched, so a broken build can never be deployed by accident.
"""
import argparse
import datetime as dt
import html
import json
import re
import shutil
import sys
from pathlib import Path
from xml.sax.saxutils import escape as xml_escape

import markdown
import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

ROOT = Path(__file__).resolve().parent
TODAY = dt.date.today()
LISTING_MAX_AGE_DAYS = 35  # AiGC's approved list must be re-checked at least monthly

# Phrases that Alberta's advertising standards make risky (see
# docs/alberta-ad-compliance-checklist.md in the repository root). A page can
# use one on purpose, e.g. to say that gambling is not a way to make money, by
# listing its label under `compliance_allow` in its front matter.
HIGH_RISK_PHRASES = {
    "make money": r"mak(?:e|es|ing) money",
    "earn money": r"earn(?:s|ing)? money",
    "income": r"incomes?",
    "get rich": r"get(?:ting)? rich",
    "guarantee": r"guarantee[sd]?",
    "sure win": r"sure (?:win|thing|bet)",
    "risk-free": r"risk[- ]free",
    "can't lose": r"can(?:'|’)?t lose|cannot lose",
    "win back": r"win(?:ning)? (?:it |them )?back",
    "recover losses": r"recover(?:ing)? (?:your |any )?loss(?:es)?",
    "investment": r"invest(?:ment|ments|ing|or|ors)?",
    "financial freedom": r"financial (?:freedom|security)",
    "bonus": r"bonus(?:es)?",
    "free spins": r"free spins?",
    "no deposit": r"no[- ]deposit",
    "deposit match": r"deposit match(?:es)?",
    "cashback": r"cash[- ]?back",
    "promo code": r"promo(?:tional)? codes?",
    "welcome offer": r"welcome (?:offer|package)s?",
}
PHRASE_PATTERNS = {label: re.compile(rf"\b(?:{rx})\b", re.I) for label, rx in HIGH_RISK_PHRASES.items()}

OPERATOR_REQUIRED = ["name", "website", "affiliate_url", "aigc_listed", "listing_checked_on",
                     "reviewed_on", "author", "summary"]
OPERATOR_OPTIONAL = {"rating": None, "logo": None, "games": [], "payment_methods": [],
                     "withdrawal_time": None, "mobile": None, "support": None, "rg_tools": [],
                     "pros": [], "cons": []}


class BuildError(Exception):
    """A problem that makes it impossible to continue the build."""


class Report:
    """Collects problems. In a preview build, errors are downgraded to warnings."""

    def __init__(self, strict):
        self.strict = strict
        self.errors = []
        self.warnings = []

    def error(self, message):
        (self.errors if self.strict else self.warnings).append(message)

    def warn(self, message):
        self.warnings.append(message)


def rel(path):
    return Path(path).relative_to(ROOT).as_posix()


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


FRONT_MATTER = re.compile(r"\A---\n(.*?)\n---\n?(.*)\Z", re.S)


def read_markdown(path, site):
    """Return (front matter, Markdown body) with {{ site.* }} placeholders filled in."""
    text = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    match = FRONT_MATTER.match(text)
    if not match:
        raise BuildError(f"{rel(path)} must start with a front matter block between --- lines")
    try:
        meta = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as e:
        raise BuildError(f"{rel(path)}: invalid front matter (a value that contains ': ' must be "
                         f"wrapped in double quotes): {e}") from None
    for key in ("title", "seo_title", "description", "h1", "lead"):
        if isinstance(meta.get(key), str):
            meta[key] = fill_placeholders(meta[key], site, path)
    return meta, fill_placeholders(match.group(2), site, path)


def fill_placeholders(text, site, path):
    def value(match):
        key = match.group(1)
        if key not in site or isinstance(site[key], (dict, list)):
            raise BuildError(f"{rel(path)}: unknown placeholder {match.group(0)}")
        return str(site[key])
    return re.sub(r"\{\{\s*site\.(\w+)\s*\}\}", value, text)


def render_markdown(text):
    """Return (HTML, table of contents as [(id, text)] for the h2 headings)."""
    md = markdown.Markdown(extensions=["extra", "toc", "sane_lists"],
                           extension_configs={"toc": {"toc_depth": "2-2"}})
    body = md.convert(text)
    toc = [(token["id"], html.unescape(token["name"])) for token in md.toc_tokens]
    # Wide tables scroll inside a wrapper instead of breaking the layout on phones
    body = body.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")
    return mark_sponsored(body), toc


def mark_sponsored(body):
    """Affiliate links (/go/...) must be marked as paid links for search engines."""
    return re.sub(r'<a href="(/go/[^"]*)"(?![^>]*\brel=)', r'<a href="\1" rel="sponsored nofollow"', body)


def require(meta, keys, path):
    missing = [k for k in keys if not meta.get(k)]
    if missing:
        raise BuildError(f"{rel(path)}: missing {', '.join(missing)} in front matter")


def as_date(value, where, report):
    if value in (None, ""):
        return None
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    report.error(f"{where}: {value!r} is not a date (use YYYY-MM-DD)")
    return None


def check_config(site, report):
    for key in ("site_name", "tagline", "base_url", "language", "og_locale", "contact_email", "authors", "nav"):
        if not site.get(key):
            raise BuildError(f"config.yaml: {key} is required")
    if not re.fullmatch(r"https://[^/\s]+", site["base_url"]):
        raise BuildError("config.yaml: base_url must look like https://www.your-domain.com (no trailing slash)")
    placeholders = [key for key in ("base_url", "contact_email") if "example.com" in site[key]]
    for key, author in site["authors"].items():
        if author.get("name") in (None, "", "Your Name") or str(author.get("bio", "")).startswith("CHANGE"):
            placeholders.append(f"authors.{key}")
    if placeholders:
        report.error("config.yaml still has placeholder values: " + ", ".join(placeholders))
    for key in ("og_image", "aigc_logo"):
        site.setdefault(key, None)
        if site[key] and not (ROOT / "static" / site[key].lstrip("/")).is_file():
            report.error(f"config.yaml: {key} points to {site[key]}, but static{site[key]} does not exist")


def load_operators(authors, report, include_drafts):
    """Return the operators to publish. Published entries are checked strictly."""
    operators = []
    seen = set()
    for index, op in enumerate(load_yaml(ROOT / "data" / "operators.yaml").get("operators") or []):
        slug = op.get("slug")
        where = f"data/operators.yaml: {slug or f'entry {index + 1}'}"
        if not slug or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", str(slug)):
            report.error(f"{where}: slug must be lowercase words joined by hyphens, e.g. my-casino")
            continue
        if slug in seen:
            report.error(f"{where}: duplicate slug")
            continue
        seen.add(slug)
        status = op.get("status")
        if status not in ("draft", "published"):
            report.error(f"{where}: status must be draft or published")
            continue
        if status == "draft" and not include_drafts:
            continue
        # Drafts only appear in previews, so their problems are warnings
        problem = report.error if status == "published" else report.warn

        missing = [k for k in OPERATOR_REQUIRED if op.get(k) in (None, "", [])]
        if missing:
            problem(f"{where}: missing {', '.join(missing)}")
        if op.get("aigc_listed") is not True:
            problem(f"{where}: aigc_listed must be true; only brands on AiGC's approved list may be promoted")
        for key in ("listing_checked_on", "reviewed_on"):
            op[key] = as_date(op.get(key), f"{where}: {key}", report)
        if op["listing_checked_on"]:
            age = (TODAY - op["listing_checked_on"]).days
            if age > LISTING_MAX_AGE_DAYS:
                problem(f"{where}: AiGC's list was last checked {age} days ago; check it again "
                        f"and update listing_checked_on")
        for key in ("website", "affiliate_url"):
            if op.get(key) and not str(op[key]).startswith("https://"):
                problem(f"{where}: {key} must start with https://")
        if op.get("author") and op["author"] not in authors:
            problem(f"{where}: author {op['author']!r} is not defined in config.yaml")
        rating = op.get("rating")
        if rating is not None and (isinstance(rating, bool) or not isinstance(rating, (int, float))
                                   or not 0 <= rating <= 5):
            problem(f"{where}: rating must be a number from 0 to 5")
            op["rating"] = None
        if op.get("logo") and not (ROOT / "static" / str(op["logo"]).lstrip("/")).is_file():
            problem(f"{where}: logo {op['logo']} does not exist under static/")
            op["logo"] = None
        if not (ROOT / "content" / "reviews" / f"{slug}.md").is_file():
            problem(f"{where}: write the review text in content/reviews/{slug}.md")
            continue
        for key, default in OPERATOR_OPTIONAL.items():
            op.setdefault(key, default)
            if op[key] == "":
                op[key] = default
        op["draft"] = status == "draft"
        operators.append(op)
    # Best rated first; unrated operators after them, alphabetically
    operators.sort(key=lambda o: (o["rating"] is None, -(o["rating"] or 0), o["name"].lower()))
    return operators


def minify_css(css):
    css = re.sub(r"/\*.*?\*/", "", css, flags=re.S)
    css = re.sub(r"\s+", " ", css)
    return re.sub(r"\s*([{}:;,>])\s*", r"\1", css).replace(";}", "}").strip()


def visible_text(page_html):
    text = re.sub(r"(?is)<(script|style)\b.*?</\1>", " ", page_html)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", html.unescape(text))


def compliance_problems(text, allowed):
    problems = []
    for label, pattern in PHRASE_PATTERNS.items():
        if label in allowed:
            continue
        match = pattern.search(text)
        if match:
            context = text[max(match.start() - 50, 0):match.end() + 50].strip()
            problems.append(f'"{label}" in "…{context}…"')
    return problems


def json_ld(graph):
    data = {"@context": "https://schema.org", "@graph": graph}
    return json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


def build(out_dir, preview):
    report = Report(strict=not preview)
    site = load_yaml(ROOT / "config.yaml")
    check_config(site, report)
    base_url = site["base_url"].rstrip("/")
    site["base_url"] = base_url
    authors = {key: {"role": "", "bio": "", **a, "id": key, "url": a.get("url") or f"/about/#author-{key}"}
               for key, a in site["authors"].items()}
    operators = load_operators(authors, report, include_drafts=preview)
    css = minify_css((ROOT / "assets" / "site.css").read_text(encoding="utf-8"))

    env = Environment(loader=FileSystemLoader(ROOT / "templates"), autoescape=select_autoescape(["html"]),
                      undefined=StrictUndefined, trim_blocks=True, lstrip_blocks=True)
    env.filters["date"] = lambda d: f"{d:%B} {d.day}, {d.year}"
    env.filters["host"] = lambda url: re.sub(r"^https?://(www\.)?", "", url).split("/")[0]

    organization = {"@type": "Organization", "@id": f"{base_url}/#organization", "name": site["site_name"],
                    "url": f"{base_url}/", "email": site["contact_email"]}
    website = {"@type": "WebSite", "@id": f"{base_url}/#website", "url": f"{base_url}/",
               "name": site["site_name"], "description": site["tagline"], "inLanguage": site["language"],
               "publisher": {"@id": organization["@id"]}}

    def person(author_id):
        author = authors[author_id]
        return {"@type": "Person", "name": author["name"], "jobTitle": author.get("role", ""),
                "url": base_url + author["url"]}

    pages = []

    def add_page(path, template, meta, *, crumbs=(), graph=(), lastmod=None, og_type="website", source="", **extra):
        """Register a page. crumbs are the (title, path) pairs after Home."""
        title = meta["title"]
        page = {
            "path": path,
            "url": base_url + path,
            "canonical": base_url + path if path.endswith("/") else None,
            "template": template,
            "title": title,
            "seo_title": meta.get("seo_title") or f"{title} | {site['site_name']}",
            "description": meta.get("description", ""),
            "noindex": preview or meta.get("noindex", False),
            "lastmod": lastmod,
            "og_type": og_type,
            "section": "/" + path.strip("/").split("/")[0] + "/" if path != "/" else "/",
            "breadcrumbs": [("Home", "/"), *crumbs],
            "allow": set(meta.get("compliance_allow") or []),
            "source": source,
            "body": "",
            "toc": [],
            **extra,
        }
        for label in page["allow"] - set(HIGH_RISK_PHRASES):
            report.warn(f"{source}: compliance_allow has unknown label {label!r}")
        nodes = [organization, website]
        if crumbs:
            nodes.append({"@type": "BreadcrumbList", "itemListElement": [
                {"@type": "ListItem", "position": i, "name": name, "item": base_url + url}
                for i, (name, url) in enumerate(page["breadcrumbs"], start=1)]})
        page["jsonld"] = json_ld([*nodes, *graph])
        pages.append(page)
        return page

    # Guides
    guides = []
    guide_dir = ROOT / "content" / "guides"
    for path in sorted(guide_dir.glob("*.md")):
        if path.name == "_index.md":
            continue
        meta, text = read_markdown(path, site)
        require(meta, ["title", "description", "date", "author"], path)
        if meta.get("author") not in authors:
            raise BuildError(f"{rel(path)}: author {meta.get('author')!r} is not defined in config.yaml")
        meta["date"] = as_date(meta.get("date"), rel(path), report) or TODAY
        meta["updated"] = as_date(meta.get("updated"), rel(path), report) or meta["date"]
        body, toc = render_markdown(text)
        faq = meta.get("faq") or []
        for item in faq:
            if not (isinstance(item, dict) and item.get("q") and item.get("a")):
                raise BuildError(f"{rel(path)}: every faq item needs q and a")
        for item in meta.get("sources") or []:
            if not (isinstance(item, dict) and item.get("title") and str(item.get("url", "")).startswith("https://")):
                raise BuildError(f"{rel(path)}: every source needs a title and an https:// url")
        guides.append({"slug": meta.get("slug") or path.stem, "meta": meta, "body": body, "toc": toc,
                       "faq": faq, "sources": meta.get("sources") or [], "source": rel(path),
                       "author": authors[meta["author"]], "path": f"/guides/{meta.get('slug') or path.stem}/"})
    guides.sort(key=lambda g: (g["meta"].get("order", 100), g["meta"]["title"].lower()))

    # Reviews
    reviews = []
    for op in operators:
        path = ROOT / "content" / "reviews" / f"{op['slug']}.md"
        meta, text = read_markdown(path, site)
        body, toc = render_markdown(text)
        updated = as_date(meta.get("updated"), rel(path), report)
        dates = [d for d in (op["reviewed_on"], updated, op["listing_checked_on"]) if d]
        reviews.append({"op": op, "meta": meta, "body": body, "toc": toc, "source": rel(path),
                        "updated": updated, "lastmod": max(dates) if dates else None,
                        "path": f"/reviews/{op['slug']}/"})

    # Home
    home_meta, home_text = read_markdown(ROOT / "content" / "home.md", site)
    require(home_meta, ["title", "description", "h1", "lead"], ROOT / "content" / "home.md")
    home_body, _ = render_markdown(home_text)
    newest = max([g["meta"]["updated"] for g in guides] + [r["lastmod"] for r in reviews if r["lastmod"]],
                 default=None)
    add_page("/", "home.html", {**home_meta, "seo_title": home_meta.get("seo_title") or home_meta["title"]},
             lastmod=newest, source="content/home.md", body=home_body, h1=home_meta["h1"],
             lead=home_meta["lead"], actions=home_meta.get("actions") or [])

    # Review list and review pages
    list_meta, list_text = read_markdown(ROOT / "content" / "reviews" / "_index.md", site)
    require(list_meta, ["title", "description"], ROOT / "content" / "reviews" / "_index.md")
    add_page("/reviews/", "reviews.html", list_meta, crumbs=[(list_meta["title"], "/reviews/")],
             lastmod=max([r["lastmod"] for r in reviews if r["lastmod"]], default=None),
             source="content/reviews/_index.md", body=render_markdown(list_text)[0],
             graph=[{"@type": "CollectionPage", "name": list_meta["title"], "url": f"{base_url}/reviews/",
                     "mainEntity": {"@type": "ItemList", "itemListElement": [
                         {"@type": "ListItem", "position": i, "url": base_url + r["path"], "name": r["op"]["name"]}
                         for i, r in enumerate(reviews, start=1)]}}])
    for review in reviews:
        op = review["op"]
        meta = review["meta"]
        title = meta.get("title") or f"{op['name']} Review"
        year = op["reviewed_on"].year if op["reviewed_on"] else TODAY.year
        review_node = {"@type": "Review", "name": title, "url": base_url + review["path"],
                       "itemReviewed": {"@type": "Organization", "name": op["name"], "url": op["website"]},
                       "reviewBody": op["summary"], "publisher": {"@id": organization["@id"]},
                       "inLanguage": site["language"]}
        if op["author"] in authors:
            review_node["author"] = person(op["author"])
        if op["reviewed_on"]:
            review_node["datePublished"] = op["reviewed_on"].isoformat()
        if review["lastmod"]:
            review_node["dateModified"] = review["lastmod"].isoformat()
        if op["rating"] is not None:
            review_node["reviewRating"] = {"@type": "Rating", "ratingValue": op["rating"],
                                           "bestRating": 5, "worstRating": 0}
        add_page(review["path"], "review.html",
                 {**meta, "title": title,
                  "seo_title": meta.get("seo_title") or f"{title} for Alberta Players ({year})",
                  "description": meta.get("description") or op["summary"]},
                 crumbs=[(list_meta["title"], "/reviews/"), (title, review["path"])], graph=[review_node],
                 lastmod=review["lastmod"], og_type="article", source=review["source"], op=op,
                 author=authors.get(op["author"]), body=review["body"], updated=review["updated"])

    # Guide list and guide pages
    list_meta, list_text = read_markdown(guide_dir / "_index.md", site)
    require(list_meta, ["title", "description"], guide_dir / "_index.md")
    add_page("/guides/", "guides.html", list_meta, crumbs=[(list_meta["title"], "/guides/")],
             lastmod=max([g["meta"]["updated"] for g in guides], default=None),
             source="content/guides/_index.md", body=render_markdown(list_text)[0],
             graph=[{"@type": "CollectionPage", "name": list_meta["title"], "url": f"{base_url}/guides/",
                     "mainEntity": {"@type": "ItemList", "itemListElement": [
                         {"@type": "ListItem", "position": i, "url": base_url + g["path"],
                          "name": g["meta"]["title"]} for i, g in enumerate(guides, start=1)]}}])
    for guide in guides:
        meta = guide["meta"]
        article = {"@type": "Article", "headline": meta["title"][:110], "description": meta["description"],
                   "url": base_url + guide["path"], "mainEntityOfPage": base_url + guide["path"],
                   "datePublished": meta["date"].isoformat(), "dateModified": meta["updated"].isoformat(),
                   "author": person(meta["author"]), "publisher": {"@id": organization["@id"]},
                   "inLanguage": site["language"]}
        graph = [article]
        if guide["faq"]:
            graph.append({"@type": "FAQPage", "mainEntity": [
                {"@type": "Question", "name": item["q"], "acceptedAnswer": {"@type": "Answer", "text": item["a"]}}
                for item in guide["faq"]]})
        related = [g for g in guides if g is not guide][:3]
        add_page(guide["path"], "guide.html", meta, crumbs=[("Guides", "/guides/"), (meta["title"], guide["path"])],
                 graph=graph, lastmod=meta["updated"], og_type="article", source=guide["source"],
                 body=guide["body"], toc=guide["toc"], faq=guide["faq"], sources=guide["sources"],
                 author=guide["author"], date=meta["date"], updated=meta["updated"], related=related)

    # Standalone pages: about, contact, policies...
    for path in sorted((ROOT / "content" / "pages").glob("*.md")):
        meta, text = read_markdown(path, site)
        require(meta, ["title", "description"], path)
        slug = meta.get("slug") or path.stem
        updated = as_date(meta.get("updated"), rel(path), report)
        schema_type = meta.get("schema_type", "WebPage")
        add_page(f"/{slug}/", "page.html", meta, crumbs=[(meta["title"], f"/{slug}/")],
                 graph=[{"@type": schema_type, "name": meta["title"], "url": f"{base_url}/{slug}/",
                         "description": meta["description"], "inLanguage": site["language"]}],
                 lastmod=updated, source=rel(path), body=render_markdown(text)[0], updated=updated,
                 show_authors=bool(meta.get("show_authors")))

    add_page("/404.html", "404.html", {"title": "Page not found", "description": "This page does not exist.",
                                         "noindex": True}, source="templates/404.html")

    # Render
    context = {"site": site, "css": css, "preview": preview, "year": TODAY.year, "operators": operators,
               "guides": guides, "authors": authors}
    rendered = {}
    for page in pages:
        rendered[page["path"]] = env.get_template(page["template"]).render(page=page, **context)

    check_pages(pages, rendered, report)

    # Write everything into a temporary folder and swap it in only if the build succeeds
    tmp_dir = out_dir.with_name(out_dir.name + ".tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    shutil.copytree(ROOT / "static", tmp_dir)
    for page in pages:
        target = tmp_dir / (page["path"].strip("/") + ("/index.html" if page["path"].endswith("/") else ""))
        if page["path"] == "/":
            target = tmp_dir / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered[page["path"]], encoding="utf-8")
    go_template = env.get_template("go.html")
    for op in operators:
        target = tmp_dir / "go" / op["slug"] / "index.html"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(go_template.render(op=op, site=site), encoding="utf-8")
    write_sitemap(pages, base_url, tmp_dir)
    write_robots_and_headers(base_url, tmp_dir, preview)
    check_links(tmp_dir, report)

    if report.errors:
        shutil.rmtree(tmp_dir)
    else:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        tmp_dir.rename(out_dir)
    return report, pages, operators


def check_pages(pages, rendered, report):
    titles, descriptions = {}, {}
    for page in pages:
        where = page["source"] or page["path"]
        page_html = rendered[page["path"]]
        if page_html.count("<h1") != 1:
            report.error(f"{where}: pages need exactly one <h1> (found {page_html.count('<h1')}); "
                         f"start headings in Markdown at ##")
        for img in re.findall(r"<img\b[^>]*>", page_html):
            if not re.search(r'\balt="', img):
                report.error(f"{where}: image without alt text: {img}")
        text = " ".join([page["seo_title"], page["description"], visible_text(page_html)])
        for problem in compliance_problems(text, page["allow"]):
            report.error(f"{where}: high-risk advertising phrase {problem}. Rewrite it, or add the label "
                         f"to compliance_allow if the use is intentional")
        if page["path"] == "/404.html":
            continue
        if not 20 <= len(page["seo_title"]) <= 65:
            report.warn(f"{where}: <title> is {len(page['seo_title'])} characters; aim for 30–60: "
                        f"{page['seo_title']!r}")
        if not 70 <= len(page["description"]) <= 160:
            report.warn(f"{where}: meta description is {len(page['description'])} characters; aim for 70–160")
        for value, seen, label in ((page["seo_title"], titles, "title"),
                                   (page["description"], descriptions, "description")):
            if value in seen:
                report.error(f"{where}: same {label} as {seen[value]}; every page needs its own")
            seen[value] = where


def write_sitemap(pages, base_url, out_dir):
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">']
    for page in pages:
        if page["noindex"] or not page["path"].endswith("/"):
            continue
        lines.append(f"  <url><loc>{xml_escape(base_url + page['path'])}</loc>"
                     + (f"<lastmod>{page['lastmod'].isoformat()}</lastmod>" if page["lastmod"] else "")
                     + "</url>")
    lines.append("</urlset>")
    (out_dir / "sitemap.xml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_robots_and_headers(base_url, out_dir, preview):
    if preview:
        robots = "User-agent: *\nDisallow: /\n"
    else:
        robots = f"User-agent: *\nDisallow: /go/\n\nSitemap: {base_url}/sitemap.xml\n"
    (out_dir / "robots.txt").write_text(robots, encoding="utf-8")
    # Netlify and Cloudflare Pages read this file; other hosts ignore it
    headers = ["/*",
               "  X-Content-Type-Options: nosniff",
               "  Referrer-Policy: strict-origin-when-cross-origin",
               "  X-Frame-Options: DENY",
               "  Permissions-Policy: camera=(), microphone=(), geolocation=()"]
    if preview:
        headers.append("  X-Robots-Tag: noindex, nofollow")
    headers += ["/go/*", "  X-Robots-Tag: noindex, nofollow"]
    (out_dir / "_headers").write_text("\n".join(headers) + "\n", encoding="utf-8")


def check_links(out_dir, report):
    files = {p.relative_to(out_dir).as_posix() for p in out_dir.rglob("*") if p.is_file()}
    for page_file in sorted(out_dir.rglob("*.html")):
        where = page_file.relative_to(out_dir).as_posix()
        for url in re.findall(r'(?:href|src)="([^"]*)"', page_file.read_text(encoding="utf-8")):
            if url.startswith(("http://", "https://", "mailto:", "tel:", "#", "data:")):
                continue
            path = url.split("#")[0].split("?")[0]
            if not path.startswith("/"):
                report.error(f"{where}: link {url!r} must start with / (or be a full https:// address)")
                continue
            target = path.lstrip("/")
            if target == "" or target.endswith("/"):
                target += "index.html"
            if target not in files:
                report.error(f"{where}: broken internal link {url}")


def main():
    parser = argparse.ArgumentParser(description="Build the static review site.")
    parser.add_argument("--drafts", action="store_true",
                        help="preview build into dist-preview/ with draft reviews (noindex, warnings only)")
    args = parser.parse_args()
    out_dir = ROOT / ("dist-preview" if args.drafts else "dist")
    try:
        report, pages, operators = build(out_dir, preview=args.drafts)
    except BuildError as e:
        sys.exit(f"Build failed: {e}")
    for message in report.warnings:
        print(f"warning: {message}")
    if report.errors:
        for message in report.errors:
            print(f"error: {message}", file=sys.stderr)
        sys.exit(f"Build failed with {len(report.errors)} error(s); {out_dir.name}/ was not changed.")
    indexed = sum(1 for p in pages if not p["noindex"])
    print(f"Built {len(pages)} pages ({indexed} indexable, {len(operators)} reviews) into {out_dir.name}/")


if __name__ == "__main__":
    main()
