#!/usr/bin/env python3
"""
Offline RAG Index Builder for "Dancing with the SARs" (NASA Space Apps Challenge).

Reads `scripts/sources.yaml`, verifies every URL resolves before ingestion (reporting
any failures or guarded TODOs without guessing replacements), extracts text from HTML,
PDF, and local Markdown files, chunks text into 500-800 token windows with 100-token
overlap while preserving section headings and page numbers, embeds chunks using a
configurable model, and writes:
  - `public/rag/index.json` (chunks + base64 float16/float32 normalized embeddings)
  - `public/rag/sources.json` (source metadata and attribution links)
  - `REFERENCES.md` (competition-ready references list generated from sources.yaml)

Usage:
  python scripts/build_rag_index.py
  python scripts/build_rag_index.py --dtype float16 --max-chunks 2000
  python scripts/build_rag_index.py --provider local --model all-MiniLM-L6-v2
  python scripts/build_rag_index.py --verify-only
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import requests
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader

try:
    import tiktoken  # type: ignore

    _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
except Exception:
    _TIKTOKEN_ENC = None


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCES_YAML = PROJECT_ROOT / "scripts" / "sources.yaml"
DEFAULT_INDEX_OUT = PROJECT_ROOT / "public" / "rag" / "index.json"
DEFAULT_SOURCES_OUT = PROJECT_ROOT / "public" / "rag" / "sources.json"
DEFAULT_REFERENCES_OUT = PROJECT_ROOT / "REFERENCES.md"

USER_AGENT = (
    "DancingWithTheSARs-RAGBuilder/1.0 "
    "(NASA Space Apps Challenge; Grounded NISAR Educational Assistant)"
)


@dataclass
class RawSection:
    section: str
    page: Optional[int]
    text: str


@dataclass
class ChunkItem:
    id: str
    source_id: str
    section: str
    page: Optional[int]
    text: str
    token_count: int


def encode_tokens(text: str) -> List[int]:
    """Tokenizes text using tiktoken if available, else deterministic regex tokens."""
    if _TIKTOKEN_ENC is not None:
        return _TIKTOKEN_ENC.encode(text)
    words = re.findall(r"\S+|\n", text)
    return list(range(len(words)))


def count_tokens(text: str) -> int:
    if _TIKTOKEN_ENC is not None:
        return len(_TIKTOKEN_ENC.encode(text))
    words = re.findall(r"\S+", text)
    return int(round(len(words) * 1.3))


def chunk_section_text(
    text: str,
    min_tokens: int = 500,
    max_tokens: int = 800,
    overlap_tokens: int = 100,
) -> List[str]:
    """
    Splits section text into chunks of 500-800 tokens with a 100-token overlap.
    Uses tiktoken exact token slicing when installed, or word-window slicing otherwise.
    """
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    if not cleaned:
        return []

    target_tokens = 650  # Midpoint inside the 500-800 token specification
    step_tokens = max(1, target_tokens - overlap_tokens)

    if _TIKTOKEN_ENC is not None:
        token_ids = _TIKTOKEN_ENC.encode(cleaned)
        if len(token_ids) <= max_tokens:
            return [cleaned]

        chunks: List[str] = []
        start = 0
        while start < len(token_ids):
            end = min(start + target_tokens, len(token_ids))
            # If the remaining tail is smaller than overlap, merge it into current chunk
            if len(token_ids) - end < overlap_tokens and end < len(token_ids):
                end = len(token_ids)
            chunk_str = _TIKTOKEN_ENC.decode(token_ids[start:end]).strip()
            if chunk_str:
                chunks.append(chunk_str)
            if end >= len(token_ids):
                break
            start += step_tokens
        return chunks

    # Fallback word-based windowing (1 word ~ 1.3 tokens)
    words = cleaned.split()
    max_words = int(max_tokens / 1.3)
    target_words = int(target_tokens / 1.3)
    overlap_words = int(overlap_tokens / 1.3)
    step_words = max(1, target_words - overlap_words)

    if len(words) <= max_words:
        return [cleaned]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + target_words, len(words))
        if len(words) - end < overlap_words and end < len(words):
            end = len(words)
        chunk_str = " ".join(words[start:end]).strip()
        if chunk_str:
            chunks.append(chunk_str)
        if end >= len(words):
            break
        start += step_words
    return chunks


def verify_and_fetch_url(
    url: str, expected_type: str, timeout: int = 25
) -> Tuple[bool, str, Optional[bytes]]:
    """
    Verifies a URL resolves with HTTP 200 and matches expected Content-Type.
    Never guesses a replacement URL if a request fails.
    """
    headers: Dict[str, str] = {"User-Agent": USER_AGENT}
    earthdata_token = os.environ.get("EARTHDATA_TOKEN", "").strip()
    if earthdata_token and "earthdatacloud.nasa.gov" in url:
        headers["Authorization"] = f"Bearer {earthdata_token}"

    try:
        response = requests.get(
            url, headers=headers, timeout=timeout, allow_redirects=True
        )
        if response.status_code != 200:
            return (
                False,
                f"HTTP {response.status_code} from {url}",
                None,
            )

        content_type = response.headers.get("Content-Type", "").lower()
        if expected_type == "pdf":
            is_pdf_bytes = response.content[:4] == b"%PDF"
            if "application/pdf" not in content_type and not is_pdf_bytes:
                return (
                    False,
                    f"Expected PDF but received Content-Type '{content_type}' (possible login redirect)",
                    None,
                )

        return True, "OK", response.content
    except requests.RequestException as exc:
        return False, f"Request failed: {exc}", None


def extract_sections_from_html(html_bytes: bytes, default_title: str) -> List[RawSection]:
    """
    Extracts structured sections from HTML by walking heading (h1-h4) and prose tags,
    discarding navigation, scripts, footers, and styles.
    """
    soup = BeautifulSoup(html_bytes, "html.parser")
    for noisy_tag in soup(["script", "style", "nav", "footer", "header", "noscript", "svg"]):
        noisy_tag.decompose()

    main_container = (
        soup.find("main")
        or soup.find("article")
        or soup.find("div", {"role": "main"})
        or soup.body
        or soup
    )

    sections: List[RawSection] = []
    current_heading = default_title
    current_paragraphs: List[str] = []

    for element in main_container.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "pre"]):
        tag_name = element.name.lower() if element.name else ""
        text = re.sub(r"\s+", " ", element.get_text(" ", strip=True)).strip()
        if not text:
            continue

        if tag_name in {"h1", "h2", "h3", "h4"}:
            if current_paragraphs:
                joined = "\n\n".join(current_paragraphs).strip()
                if len(joined) >= 80:
                    sections.append(
                        RawSection(section=current_heading, page=None, text=joined)
                    )
                current_paragraphs = []
            current_heading = text[:160]
        else:
            if len(text) >= 25:
                current_paragraphs.append(text)

    if current_paragraphs:
        joined = "\n\n".join(current_paragraphs).strip()
        if len(joined) >= 80:
            sections.append(RawSection(section=current_heading, page=None, text=joined))

    return sections


def detect_pdf_page_heading(page_text: str, fallback_heading: str) -> str:
    """Heuristically finds the top section heading on a PDF page."""
    for line in page_text.splitlines()[:8]:
        clean = line.strip()
        if 4 <= len(clean) <= 110 and not clean.isdigit():
            # Avoid running headers with copyright notices
            if "california institute of technology" in clean.lower():
                continue
            return clean
    return fallback_heading


def extract_sections_from_pdf(
    pdf_bytes: bytes, default_title: str, max_pages: Optional[int] = None
) -> List[RawSection]:
    """Extracts page-indexed text from a PDF file using pypdf."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    sections: List[RawSection] = []
    total_pages = len(reader.pages)
    limit = min(total_pages, max_pages) if max_pages else total_pages

    last_heading = default_title
    for idx in range(limit):
        page_num = idx + 1
        try:
            raw_text = reader.pages[idx].extract_text() or ""
        except Exception:
            continue
        cleaned = re.sub(r"[ \t]+", " ", raw_text).strip()
        if len(cleaned) < 60:
            continue
        heading = detect_pdf_page_heading(cleaned, last_heading)
        last_heading = heading
        sections.append(RawSection(section=heading, page=page_num, text=cleaned))

    return sections


def extract_sections_from_markdown(md_text: str, default_title: str) -> List[RawSection]:
    """Splits a local Markdown document (`docs/challenge.md`, `docs/methods.md`) by `#` headings."""
    sections: List[RawSection] = []
    current_heading = default_title
    buffer: List[str] = []

    for line in md_text.splitlines():
        heading_match = re.match(r"^(#{1,4})\s+(.+)$", line.strip())
        if heading_match:
            if buffer:
                body = "\n".join(buffer).strip()
                if body:
                    sections.append(
                        RawSection(section=current_heading, page=None, text=body)
                    )
                buffer = []
            current_heading = heading_match.group(2).strip()[:160]
        else:
            buffer.append(line)

    if buffer:
        body = "\n".join(buffer).strip()
        if body:
            sections.append(RawSection(section=current_heading, page=None, text=body))

    return sections


def build_gemini_embedder(model_name: str, output_dim: int = 768) -> Callable[[List[str]], np.ndarray]:
    """
    Creates an embedding function backed by the Gemini REST API (`GEMINI_API_KEY`).
    Uses batchEmbedContents so no extra SDK version conflicts occur.
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is required when --provider=gemini. "
            "Set GEMINI_API_KEY or run with `--provider local`."
        )

    clean_model = model_name if model_name.startswith("models/") else f"models/{model_name}"
    endpoint = f"https://generativelanguage.googleapis.com/v1beta/{clean_model}:batchEmbedContents?key={api_key}"

    def embed_batch(texts: List[str]) -> np.ndarray:
        all_vectors: List[List[float]] = []
        batch_size = 50
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            requests_payload = [
                {
                    "model": clean_model,
                    "content": {"parts": [{"text": t}]},
                    "outputDimensionality": output_dim,
                }
                for t in batch
            ]
            resp = requests.post(
                endpoint,
                json={"requests": requests_payload},
                timeout=60,
            )
            if resp.status_code != 200:
                raise RuntimeError(
                    f"Gemini embedding API returned HTTP {resp.status_code}: {resp.text[:300]}"
                )
            data = resp.json()
            for item in data.get("embeddings", []):
                all_vectors.append(item["values"])
            time.sleep(0.15)  # Respect rate limits
        matrix = np.asarray(all_vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return matrix / norms

    return embed_batch


def build_local_embedder(model_name: str) -> Callable[[List[str]], np.ndarray]:
    """Creates an offline SentenceTransformers embedder."""
    from sentence_transformers import SentenceTransformer  # type: ignore

    model = SentenceTransformer(model_name)

    def embed_batch(texts: List[str]) -> np.ndarray:
        vecs = model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
        )
        return np.asarray(vecs, dtype=np.float32)

    return embed_batch


def encode_vector_base64(vec: np.ndarray, dtype: str) -> str:
    """Serializes a 1-D numpy vector into base64 float16 or float32 bytes (little-endian)."""
    target_dtype = np.dtype("<f2") if dtype == "float16" else np.dtype("<f4")
    casted = vec.astype(target_dtype)
    return base64.b64encode(casted.tobytes()).decode("ascii")


def write_references_md(sources_cfg: List[Dict[str, Any]], out_path: Path) -> None:
    """Generates REFERENCES.md from sources.yaml for the Space Apps References field."""
    lines: List[str] = [
        "# External References & Official Data Sources",
        "",
        "Generated automatically from `scripts/sources.yaml` for the **Dancing with the SARs** NASA Space Apps Challenge submission.",
        "",
        "## Ingested Official NISAR & SAR Documentation",
        "",
    ]

    for idx, src in enumerate(sources_cfg, start=1):
        status = "Ingested / Enabled" if src.get("enabled", True) else "TODO (Unverified Direct URL)"
        lines.append(f"{idx}. **{src['title']}** (`{src['id']}`)")
        lines.append(f"   - **URL / Path**: {src['url']}")
        lines.append(f"   - **Attribution / License**: {src.get('attribution', 'NASA / JPL / ASF')}")
        lines.append(f"   - **Status**: {status}")
        if src.get("notes"):
            lines.append(f"   - **Notes**: {src['notes']}")
        lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build RAG index and REFERENCES.md from scripts/sources.yaml"
    )
    parser.add_argument(
        "--sources",
        type=Path,
        default=DEFAULT_SOURCES_YAML,
        help="Path to sources.yaml",
    )
    parser.add_argument(
        "--index-out",
        type=Path,
        default=DEFAULT_INDEX_OUT,
        help="Output path for public/rag/index.json",
    )
    parser.add_argument(
        "--sources-out",
        type=Path,
        default=DEFAULT_SOURCES_OUT,
        help="Output path for public/rag/sources.json",
    )
    parser.add_argument(
        "--references-out",
        type=Path,
        default=DEFAULT_REFERENCES_OUT,
        help="Output path for REFERENCES.md",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "local"],
        default=os.environ.get("RAG_EMBEDDING_PROVIDER", "gemini"),
        help="Embedding provider ('gemini' uses GEMINI_API_KEY; 'local' uses sentence-transformers)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("GEMINI_EMBEDDING_MODEL", "text-embedding-004"),
        help="Embedding model name (e.g., text-embedding-004, gemini-embedding-001, or all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=768,
        help="Output embedding dimension for Gemini embeddings (default: 768)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float16",
        help="Vector serialization format (float16 halves index file size)",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=2500,
        help="Maximum chunks to retain so serverless cold start stays fast",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify URLs and generate REFERENCES.md without running embeddings",
    )
    args = parser.parse_args()

    if not args.sources.exists():
        print(f"[ERROR] sources.yaml not found at {args.sources}", file=sys.stderr)
        return 1

    cfg = yaml.safe_load(args.sources.read_text(encoding="utf-8"))
    sources_list: List[Dict[str, Any]] = cfg.get("sources", [])

    # Always generate REFERENCES.md from sources.yaml
    write_references_md(sources_list, args.references_out)
    print(f"[INFO] Generated {args.references_out}")

    all_chunks: List[ChunkItem] = []
    ingested_sources: List[Dict[str, str]] = []
    verification_failures: List[Tuple[str, str, str]] = []
    todo_skipped: List[Tuple[str, str]] = []

    for src in sources_list:
        src_id = str(src["id"])
        title = str(src["title"])
        url = str(src["url"])
        src_type = str(src.get("type", "html")).lower()
        enabled = bool(src.get("enabled", True))
        attribution = str(src.get("attribution", "NASA / JPL / ASF"))
        max_pages = src.get("max_pages")

        if not enabled or src.get("todo") or url.startswith("TODO"):
            todo_skipped.append((src_id, url))
            print(f"[TODO/SKIP] {src_id}: Marked as unverified TODO ({url})")
            continue

        raw_sections: List[RawSection] = []

        if src_type == "markdown" or not url.startswith(("http://", "https://")):
            local_path = PROJECT_ROOT / url
            if not local_path.exists():
                verification_failures.append(
                    (src_id, url, f"Local file does not exist at {local_path}")
                )
                print(f"[WARN] Local file missing for {src_id}: {local_path}")
                continue
            md_text = local_path.read_text(encoding="utf-8")
            if not args.verify_only:
                raw_sections = extract_sections_from_markdown(md_text, title)
            print(f"[OK] Verified local source {src_id} ({local_path})")
        else:
            ok, reason, content_bytes = verify_and_fetch_url(url, src_type)
            if not ok or content_bytes is None:
                verification_failures.append((src_id, url, reason))
                print(f"[FAIL] {src_id} ({url}) -> {reason}")
                continue
            print(f"[OK] Verified URL {src_id} ({len(content_bytes) / 1024:.1f} KB)")

            if not args.verify_only:
                if src_type == "pdf":
                    raw_sections = extract_sections_from_pdf(
                        content_bytes, title, max_pages=max_pages
                    )
                else:
                    raw_sections = extract_sections_from_html(content_bytes, title)

        if args.verify_only:
            continue

        source_chunk_count = 0
        for sec in raw_sections:
            pieces = chunk_section_text(
                sec.text, min_tokens=500, max_tokens=800, overlap_tokens=100
            )
            for piece in pieces:
                if len(all_chunks) >= args.max_chunks:
                    break
                source_chunk_count += 1
                chunk_id = f"{src_id}#c{source_chunk_count:04d}"
                all_chunks.append(
                    ChunkItem(
                        id=chunk_id,
                        source_id=src_id,
                        section=sec.section,
                        page=sec.page,
                        text=piece,
                        token_count=count_tokens(piece),
                    )
                )
            if len(all_chunks) >= args.max_chunks:
                break

        if source_chunk_count > 0:
            ingested_sources.append(
                {
                    "id": src_id,
                    "title": title,
                    "url": url,
                    "attribution": attribution,
                }
            )
            print(f"     -> Extracted {source_chunk_count} chunks from {src_id}")

    print("\n=== URL & Source Verification Summary ===")
    print(f"Verified & Ingested Sources : {len(ingested_sources)}")
    print(f"Guarded TODO Rows Skipped   : {len(todo_skipped)}")
    print(f"Failed Resolutions          : {len(verification_failures)}")
    for fid, furl, freason in verification_failures:
        print(f"  - [FAILED] {fid}: {furl} ({freason})")

    if args.verify_only:
        return 0

    if not all_chunks:
        print("[ERROR] No chunks were extracted. Check network or local docs.", file=sys.stderr)
        return 1

    print(f"\n[INFO] Embedding {len(all_chunks)} chunks with provider='{args.provider}', model='{args.model}'...")
    if args.provider == "local":
        embed_fn = build_local_embedder(args.model)
    else:
        embed_fn = build_gemini_embedder(args.model, output_dim=args.dim)

    texts_to_embed = [
        f"{c.section}\n\n{c.text}" for c in all_chunks
    ]
    vectors = embed_fn(texts_to_embed)
    actual_dim = int(vectors.shape[1])

    index_records: List[Dict[str, Any]] = []
    for chunk, vec in zip(all_chunks, vectors):
        index_records.append(
            {
                "id": chunk.id,
                "source_id": chunk.source_id,
                "section": chunk.section,
                "page": chunk.page,
                "text": chunk.text,
                "embedding": encode_vector_base64(vec, args.dtype),
                "dtype": args.dtype,
            }
        )

    args.index_out.parent.mkdir(parents=True, exist_ok=True)
    args.sources_out.parent.mkdir(parents=True, exist_ok=True)

    index_payload = {
        "version": 1,
        "model": args.model,
        "provider": args.provider,
        "dimension": actual_dim,
        "dtype": args.dtype,
        "chunk_count": len(index_records),
        "chunks": index_records,
    }

    args.index_out.write_text(
        json.dumps(index_payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    args.sources_out.write_text(
        json.dumps(ingested_sources, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    index_size_kb = args.index_out.stat().st_size / 1024.0
    sources_size_kb = args.sources_out.stat().st_size / 1024.0
    print("\n=== RAG Index Build Complete ===")
    print(f"Total Chunks Written : {len(index_records)}")
    print(f"Embedding Dimension  : {actual_dim} ({args.dtype})")
    print(f"Index File Size      : {index_size_kb:.1f} KB ({index_size_kb / 1024.0:.2f} MB) -> {args.index_out}")
    print(f"Sources File Size    : {sources_size_kb:.1f} KB -> {args.sources_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())