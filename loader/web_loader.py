"""
loader/web_loader.py
====================
Web loader for VinWonders product pages.

The loader is intentionally conservative: it starts from the Vinpearl Safari
Phu Quoc page, follows only a small set of relevant Vietnamese VinWonders links,
and extracts visible content into section-level LangChain Documents.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable
from urllib.parse import urljoin, urlparse, urlunparse

from langchain_core.documents import Document

from loader.base import BaseLoader, Language
from loader.utils import content_hash

logger = logging.getLogger(__name__)

DEFAULT_SEED_URL = "https://vinwonders.com/vi/vinpearl-safari-phu-quoc/"

DEFAULT_INCLUDE_KEYWORDS = (
    "vinpearl-safari",
    "safari",
    "night-safari",
    "junior-zoo-keeper",
    "phu-quoc",
    "phú quốc",
    "dong-vat",
    "động vật",
)

DEFAULT_EXCLUDE_PATTERNS = (
    "booking.vinwonders.com",
    "affiliate",
    "vinclub",
    "dang-nhap",
    "login",
    "register",
    "static.",
    "/wp-content/",
    "/cdn-cgi/",
    "mastercard",
    "mega-sale",
    "flash-sale",
    "happy-member",
    "combo",
    "voucher",
    "phu-quoc-deals",
    "daily-tour-grand-world",
    "diem-den-phu-quoc",
    "wonderpedia/phu-quoc",
    "grand-world-phu-quoc",
    "/vi/vinwonders-phu-quoc/",
    "hai-duong-hoc",
    "oceanographer",
)

ASSET_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".pdf",
    ".zip",
    ".mp4",
    ".mov",
    ".avi",
    ".webm",
)

NOISE_TEXT = {
    "đặt vé",
    "đặt vé vinwonders",
    "chọn điểm đến",
    "chọn ngày",
    "vui lòng chọn điểm đến",
    "vui lòng chọn ngày sử dụng",
    "người lớn",
    "trẻ em",
    "người cao tuổi",
    "tìm kiếm",
    "đăng nhập",
    "đăng ký",
    "xem thêm",
    "săn ngay",
    "scroll",
}

RELEVANCE_TERMS = (
    "vinpearl safari",
    "safari",
    "night safari",
    "động vật",
    "dong vat",
    "junior zoo keeper",
)


@dataclass
class CrawlResult:
    docs: list[Document] = field(default_factory=list)
    included_urls: list[str] = field(default_factory=list)
    skipped_urls: list[dict] = field(default_factory=list)
    combined_text: str = ""


def canonicalize_url(url: str, base_url: str | None = None) -> str:
    """Normalize VinWonders URLs for deduping and cache keys."""
    joined = urljoin(base_url or "", url.strip())
    parsed = urlparse(joined)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc.lower()
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    if not path.endswith("/") and "." not in path.rsplit("/", 1)[-1]:
        path += "/"
    return urlunparse((scheme, netloc, path, "", "", ""))


def normalize_text(text: str) -> str:
    """Collapse whitespace while preserving readable Vietnamese text."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def web_snapshot_hash(urls: Iterable[str], text: str) -> str:
    h = hashlib.sha256()
    for url in sorted(set(urls)):
        h.update(url.encode("utf-8"))
        h.update(b"\n")
    h.update(normalize_text(text).encode("utf-8"))
    return h.hexdigest()


class VinWondersWebLoader(BaseLoader):
    """
    Crawl and parse VinWonders Safari-related pages.

    Defaults are tuned for local MVP usage: one crawl level and at most 8 pages.
    """

    def __init__(
        self,
        language: Language = "vi",
        max_depth: int = 1,
        max_pages: int = 8,
        include_keywords: Iterable[str] | None = None,
        exclude_patterns: Iterable[str] | None = None,
        timeout: int = 20,
    ):
        super().__init__(language)
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.include_keywords = tuple(include_keywords or DEFAULT_INCLUDE_KEYWORDS)
        self.exclude_patterns = tuple(exclude_patterns or DEFAULT_EXCLUDE_PATTERNS)
        self.timeout = timeout

    def load(self, file_path: str) -> list[Document]:
        """Load from a URL. The BaseLoader arg is named file_path for contract compatibility."""
        result = self.crawl(file_path or DEFAULT_SEED_URL)
        return result.docs

    def snapshot(self, url: str) -> CrawlResult:
        """Fetch and parse pages for cache hashing without changing repo state."""
        return self.crawl(url or DEFAULT_SEED_URL)

    def crawl(self, seed_url: str) -> CrawlResult:
        from bs4 import BeautifulSoup
        import requests

        session = requests.Session()
        session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                "RAGPipelineVisualizer/1.0 SafariCrawler"
            )
        })

        seed = canonicalize_url(seed_url or DEFAULT_SEED_URL)
        queue: list[tuple[str, int, str]] = [(seed, 0, "seed")]
        visited: set[str] = set()
        seen_content: set[str] = set()
        result = CrawlResult()

        while queue and len(result.included_urls) < self.max_pages:
            url, depth, anchor = queue.pop(0)
            if url in visited:
                continue
            visited.add(url)

            ok, reason = self._is_allowed_url(url, anchor, is_seed=(url == seed))
            if not ok:
                result.skipped_urls.append({"url": url, "reason": reason, "anchor": anchor})
                continue

            try:
                response = session.get(url, timeout=self.timeout)
                response.raise_for_status()
            except Exception as exc:
                logger.warning("VinWonders fetch failed %s: %s", url, exc)
                result.skipped_urls.append({"url": url, "reason": f"fetch_error: {exc}", "anchor": anchor})
                continue

            content_type = response.headers.get("content-type", "")
            if "text/html" not in content_type and content_type:
                result.skipped_urls.append({"url": url, "reason": f"non_html: {content_type}", "anchor": anchor})
                continue

            soup = BeautifulSoup(response.text, "lxml")
            title = normalize_text(soup.title.get_text(" ")) if soup.title else ""
            page_links = self._extract_links(soup, url)
            docs = self._extract_documents(soup, url, title, depth)
            page_text = "\n\n".join(doc.page_content for doc in docs)

            if url != seed and not self._is_relevant_page(url, anchor, title, page_text):
                result.skipped_urls.append({"url": url, "reason": "content_not_relevant", "anchor": anchor})
                continue

            kept_docs: list[Document] = []
            for doc in docs:
                h = content_hash(doc.page_content)
                if h not in seen_content:
                    seen_content.add(h)
                    kept_docs.append(doc)

            if not kept_docs:
                result.skipped_urls.append({"url": url, "reason": "empty_or_duplicate", "anchor": anchor})
                continue

            result.included_urls.append(url)
            result.docs.extend(kept_docs)
            result.combined_text += "\n\n" + page_text

            if depth < self.max_depth:
                for link, link_text in page_links:
                    if link in visited or any(item[0] == link for item in queue):
                        continue
                    ok, reason = self._is_allowed_url(link, link_text, is_seed=False)
                    if ok:
                        queue.append((link, depth + 1, link_text))
                    else:
                        result.skipped_urls.append({"url": link, "reason": reason, "anchor": link_text})

        for doc in result.docs:
            doc.metadata["_crawl_included_urls"] = result.included_urls
            doc.metadata["_crawl_skipped_urls"] = result.skipped_urls[:80]

        self._stamp(result.docs)
        return result

    def _extract_links(self, soup, base_url: str) -> list[tuple[str, str]]:
        links: list[tuple[str, str]] = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "")
            text = normalize_text(a.get_text(" "))
            if not href or href.startswith(("#", "mailto:", "tel:", "javascript:")):
                continue
            links.append((canonicalize_url(href, base_url), text))
        def _score(item: tuple[str, str]) -> int:
            haystack = f"{item[0]} {item[1]}".lower()
            if "night-safari" in haystack or "night safari" in haystack:
                return 0
            if "junior-zoo-keeper" in haystack or "zoo keeper" in haystack:
                return 1
            if "vinpearl-safari" in haystack or "vinpearl safari" in haystack:
                return 2
            if "safari" in haystack:
                return 3
            return 9

        return sorted(links, key=_score)

    def _is_allowed_url(self, url: str, anchor: str = "", is_seed: bool = False) -> tuple[bool, str]:
        parsed = urlparse(url)
        haystack = f"{url} {anchor}".lower()

        if parsed.scheme not in ("http", "https"):
            return False, "unsupported_scheme"
        if parsed.netloc.lower() not in ("vinwonders.com", "www.vinwonders.com"):
            return False, "external_domain"
        if not parsed.path.startswith("/vi/"):
            return False, "non_vietnamese_path"
        if parsed.path.lower().endswith(ASSET_EXTENSIONS):
            return False, "asset"
        for pattern in self.exclude_patterns:
            if pattern.lower() in haystack:
                if is_seed:
                    break
                return False, f"exclude_pattern:{pattern}"
        if is_seed:
            return True, "seed"
        if not any(kw.lower() in haystack for kw in self.include_keywords):
            return False, "no_include_keyword"
        return True, "allowed"

    def _is_relevant_page(self, url: str, anchor: str, title: str, text: str) -> bool:
        haystack = f"{url} {anchor} {title} {text[:3000]}".lower()
        return any(term in haystack for term in RELEVANCE_TERMS)

    def _extract_documents(self, soup, url: str, title: str, depth: int) -> list[Document]:
        self._remove_noise_nodes(soup)

        body = soup.find("main") or soup.find("body") or soup
        sections: list[tuple[str, list[str]]] = []
        current_heading = title or url
        current_lines: list[str] = []
        emitted_lines: set[str] = set()

        for tag in body.find_all(["h1", "h2", "h3", "h4", "h5", "p", "li", "td", "th"], recursive=True):
            text = normalize_text(tag.get_text(" "))
            if not self._keep_text(text):
                continue

            if tag.name in {"h1", "h2", "h3", "h4", "h5"}:
                if current_lines:
                    sections.append((current_heading, current_lines))
                    current_lines = []
                current_heading = text
                emitted_lines = {text.lower()}
                continue

            key = text.lower()
            if key in emitted_lines:
                continue
            emitted_lines.add(key)
            current_lines.append(text)

        if current_lines:
            sections.append((current_heading, current_lines))

        if not sections:
            text = normalize_text(body.get_text(" "))
            if self._keep_text(text):
                sections = [(title or url, [text])]

        docs: list[Document] = []
        fetched_at = datetime.now().isoformat(timespec="seconds")
        for heading, lines in sections:
            content = "\n".join([heading, *lines]).strip()
            if len(content) < 80:
                continue
            docs.append(Document(
                page_content=content,
                metadata={
                    "source": url,
                    "source_url": url,
                    "canonical_url": url,
                    "title": title,
                    "section_heading": heading,
                    "crawl_depth": depth,
                    "content_type": "web_page",
                    "file_type": "html",
                    "fetched_at": fetched_at,
                    "language": self.language,
                },
            ))
        return docs

    def _remove_noise_nodes(self, soup) -> None:
        noisy_tags = ["script", "style", "noscript", "svg", "canvas", "iframe", "form", "input", "button"]
        for node in soup.find_all(noisy_tags):
            node.decompose()

        noisy_attr = re.compile(
            r"(header|footer|nav|menu|breadcrumb|booking|ticket|login|register|"
            r"language|modal|popup|cookie|search|social|share)",
            re.IGNORECASE,
        )
        to_remove = []
        for node in soup.find_all(True):
            if getattr(node, "attrs", None) is None:
                continue
            attrs = " ".join([
                " ".join(node.get("class", [])) if isinstance(node.get("class"), list) else str(node.get("class", "")),
                str(node.get("id", "")),
                str(node.get("role", "")),
                str(node.get("aria-label", "")),
            ])
            if noisy_attr.search(attrs):
                to_remove.append(node)
        for node in to_remove:
            if getattr(node, "parent", None) is not None:
                node.decompose()

    def _keep_text(self, text: str) -> bool:
        if not text or len(text) < 3:
            return False
        low = text.lower().strip(" :.-–")
        if low in NOISE_TEXT:
            return False
        if low.startswith(("vui lòng", "tối đa 20 khách", "đối tượng cần")):
            return False
        if len(text) < 18 and low in {"phú quốc", "vinpearl safari phú quốc", "mua sắm"}:
            return True
        return len(text) >= 12
