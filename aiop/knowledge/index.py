"""Lightweight, dependency-free knowledge retrieval (BM25 + tag boosts).

The knowledge base must be usable without embedding services or network access,
so chunks are tokenized (English words + Chinese bigrams) and ranked with BM25.
Indexes are rebuilt automatically when source files change.
"""

from __future__ import annotations

import json
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from ..config import PROJECT_ROOT

TOKEN_RE = re.compile(r"[a-z0-9_./:-]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]")

STOPWORDS = {
    "the", "a", "an", "of", "to", "in", "on", "for", "and", "or", "is", "are",
    "be", "with", "by", "as", "at", "it", "this", "that", "from", "we", "you",
    "can", "not", "no", "if", "then", "so", "but", "all", "any", "use", "using",
}


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    tokens = [t for t in TOKEN_RE.findall(lowered) if t not in STOPWORDS]
    cjk = CJK_RE.findall(lowered)
    tokens.extend(cjk)
    for i in range(len(cjk) - 1):
        tokens.append(cjk[i] + cjk[i + 1])
    return tokens


@dataclass
class Chunk:
    doc: str
    title: str
    heading: str
    text: str
    tags: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc": self.doc,
            "title": self.title,
            "heading": self.heading,
            "text": self.text,
            "tags": self.tags,
        }


def _split_markdown(path: Path, rel: str) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    title = path.stem
    if lines and lines[0].startswith("# "):
        title = lines[0][2:].strip()
    chunks: list[Chunk] = []
    h2 = ""
    h3 = ""
    heading_path = [title]
    buffer: list[str] = []
    tags = [p.name for p in path.parents[:2]]
    for line in lines:
        if line.startswith("## ") or line.startswith("### "):
            if buffer:
                text = "\n".join(buffer).strip()
                if text:
                    chunks.append(Chunk(rel, title, " / ".join(heading_path), text, tags))
                buffer = []
            if line.startswith("## "):
                h2 = line[3:].strip()
                h3 = ""
            else:
                h3 = line[4:].strip()
            heading_path = [title, h2] + ([h3] if h3 else [])
        else:
            buffer.append(line)
    if buffer:
        text = "\n".join(buffer).strip()
        if text:
            chunks.append(Chunk(rel, title, " / ".join(heading_path), text, tags))
    return chunks


def _sources(root: Path) -> Iterable[Path]:
    patterns = [
        "knowledge/*.md",
        ".agent/skills/*/SKILL.md",
        ".agent/skills/*/references/*.md",
        ".opencode/agents/*.md",
        "samples/*.md",
    ]
    seen: set[Path] = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file() and path.suffix == ".md" and path.stat().st_size < 400_000:
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    yield path


class KnowledgeIndex:
    def __init__(self, root: Path | None = None, index_path: Path | None = None) -> None:
        self.root = Path(root or PROJECT_ROOT)
        self.index_path = Path(index_path or self.root / ".index" / "knowledge.json")
        self.chunks: list[dict[str, Any]] = []
        self.df: dict[str, int] = {}
        self.avg_len = 0.0
        self.built_at = 0.0
        self._load_or_build()

    def _fingerprint(self) -> float:
        stamps = [p.stat().st_mtime for p in _sources(self.root)]
        return max(stamps) if stamps else 0.0

    def _load_or_build(self) -> None:
        fp = self._fingerprint()
        if self.index_path.is_file():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if data.get("fingerprint") == fp:
                    self.chunks = data["chunks"]
                    self.df = {k: int(v) for k, v in data["df"].items()}
                    self.avg_len = float(data["avg_len"])
                    self.built_at = float(data.get("built_at", 0))
                    return
            except (json.JSONDecodeError, KeyError, ValueError):
                pass
        self.build()

    def build(self) -> None:
        chunks: list[dict[str, Any]] = []
        for path in _sources(self.root):
            rel = str(path.relative_to(self.root))
            for chunk in _split_markdown(path, rel):
                tokens = tokenize(chunk.text + " " + chunk.heading)
                if not tokens:
                    continue
                item = chunk.to_dict()
                item["tokens"] = tokens
                item["len"] = len(tokens)
                chunks.append(item)
        df: dict[str, int] = {}
        for item in chunks:
            for token in set(item["tokens"]):
                df[token] = df.get(token, 0) + 1
        avg_len = sum(c["len"] for c in chunks) / max(1, len(chunks))
        self.chunks = chunks
        self.df = df
        self.avg_len = avg_len
        self.built_at = time.time()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.index_path.write_text(
            json.dumps(
                {
                    "fingerprint": self._fingerprint(),
                    "built_at": self.built_at,
                    "avg_len": avg_len,
                    "df": df,
                    "chunks": chunks,
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def search(self, query: str, k: int = 5, tags: list[str] | None = None) -> list[dict[str, Any]]:
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        n = max(1, len(self.chunks))
        k1, b = 1.5, 0.75
        scored: list[tuple[float, dict[str, Any]]] = []
        for item in self.chunks:
            if tags and not set(tags).intersection(item["tags"]):
                continue
            tf: dict[str, int] = {}
            for token in item["tokens"]:
                tf[token] = tf.get(token, 0) + 1
            score = 0.0
            for token in q_tokens:
                freq = tf.get(token, 0)
                if not freq:
                    continue
                idf = math.log(1 + (n - self.df.get(token, 0) + 0.5) / (self.df.get(token, 0) + 0.5))
                score += idf * freq * (k1 + 1) / (freq + k1 * (1 - b + b * item["len"] / max(self.avg_len, 1e-6)))
            if score <= 0:
                continue
            if any(token in item["heading"].lower() for token in q_tokens if len(token) > 2):
                score *= 1.15
            scored.append((score, item))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        results = []
        for score, item in scored[:k]:
            results.append(
                {
                    "score": round(score, 4),
                    "doc": item["doc"],
                    "title": item["title"],
                    "heading": item["heading"],
                    "text": item["text"],
                }
            )
        return results

    def context(self, query: str, k: int = 5, max_chars: int = 6000) -> str:
        parts = []
        used = 0
        for hit in self.search(query, k=k):
            block = f"[{hit['doc']} › {hit['heading']}]\n{hit['text']}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)
        return "\n\n---\n\n".join(parts)


_DEFAULT: KnowledgeIndex | None = None


def default_index() -> KnowledgeIndex:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = KnowledgeIndex()
    return _DEFAULT
