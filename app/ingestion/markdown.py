"""Turn a handbook Markdown page into clean, heading-aware chunks. Pure functions: no DB, no model.

Why split by headings: a policy page is organized by sections ("Travel > Airfare > Advance
Purchase"). A chunk that stays inside one section is about one thing, and its heading path tells
the retriever (and later the LLM) what the passage is about even when the text says "Limit is set
at $60/year" without naming the subject.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import yaml

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.S)
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")

# Hugo shortcodes: {{% name args %}} / {{< name args >}}, and closing tags {{% /name %}}.
_SHORTCODE = re.compile(r"\{\{[<%]\s*(/?)([\w-]+)([^}]*?)\s*[>%]\}\}")
_ATTR = re.compile(r'(\w[\w-]*)="([^"]*)"')
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)
_MERMAID = re.compile(r"^```mermaid\n.*?^```\s*$", re.S | re.M)  # diagram source, not prose
_SUMMARY = re.compile(r"<summary>(.*?)</summary>", re.S | re.I)
_BR = re.compile(r"<br\s*/?>", re.I)
_DROP_TAGS = re.compile(r"<iframe\b.*?</iframe>", re.S | re.I)
_INLINE_TAGS = re.compile(r"</?(details|summary|strong|b|i|em|span|p|div|sup|sub|a|u)\b[^>]*>", re.I)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\((?:[^()]|\([^)]*\))*\)")  # [text](url), url may hold one (...) pair
_BLANKS = re.compile(r"\n{3,}")


@dataclass(frozen=True)
class ChunkDraft:
    seq: int
    heading_path: str
    text: str


def split_front_matter(raw: str) -> tuple[dict, str]:
    """YAML front matter (title, description...) and the Markdown body."""
    m = _FRONT_MATTER.match(raw)
    if not m:
        return {}, raw
    try:
        meta = yaml.safe_load(m.group(1)) or {}
    except yaml.YAMLError:
        meta = {}
    return (meta if isinstance(meta, dict) else {}), raw[m.end():]


def _shortcode(m: re.Match) -> str:
    closing, name, args = m.group(1), m.group(2).lower(), m.group(3)
    if closing:
        return ""
    attrs = dict(_ATTR.findall(args))
    # Keep the label a reader would see: a details summary ("GitLab LTD (UK)") or a panel header
    # ("Coding") is context for the text inside it.
    label = attrs.get("summary") or attrs.get("header") or attrs.get("name")
    if name in {"details", "panel", "card", "alert", "label"} and label:
        return f"\n**{label.strip('* ')}**\n"
    return ""  # youtube, cardpane, misused-terms...: nothing readable


def clean_markdown(body: str) -> str:
    """Remove Hugo/HTML noise, keep the readable text. Markdown emphasis and tables stay."""
    text = _HTML_COMMENT.sub("", body)
    text = _MERMAID.sub("", text)
    text = _SHORTCODE.sub(_shortcode, text)
    text = _DROP_TAGS.sub("", text)
    text = _SUMMARY.sub(lambda m: f"\n**{m.group(1).strip()}**\n", text)
    text = _BR.sub(" ", text)
    text = _INLINE_TAGS.sub("", text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = "\n".join(line.rstrip() for line in text.splitlines())
    return _BLANKS.sub("\n\n", text).strip()


def _plain_heading(text: str) -> str:
    return _LINK.sub(r"\1", text).replace("**", "").replace("`", "").strip()


def _sections(text: str, title: str) -> list[tuple[str, str]]:
    """(heading path, body) for each heading section, in order."""
    stack: list[tuple[int, str]] = []
    sections: list[tuple[str, list[str]]] = [(title, [])]
    in_fence = False
    for line in text.splitlines():
        if _FENCE.match(line):
            in_fence = not in_fence
        m = None if in_fence else _HEADING.match(line)
        if m:
            level, heading = len(m.group(1)), _plain_heading(m.group(2))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, heading))
            path = " > ".join([title] + [h for _, h in stack if h and h != title])
            sections.append((path, []))
        elif line.strip() not in {"---", "***", "___"}:  # horizontal rules carry no text
            sections[-1][1].append(line)
    return [(path, "\n".join(lines).strip()) for path, lines in sections]


def _split_long(block: str, max_chars: int) -> list[str]:
    """A block bigger than max_chars: split on lines, then on sentences, then hard."""
    pieces: list[str] = []
    for line in block.splitlines():
        if len(line) <= max_chars:
            pieces.append(line)
            continue
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for s in sentences:
            while len(s) > max_chars:
                cut = s.rfind(" ", 0, max_chars)
                cut = cut if cut > max_chars // 2 else max_chars
                pieces.append(s[:cut])
                s = s[cut:].lstrip()
            pieces.append(s)
    return _pack(pieces, max_chars, sep="\n")


def _pack(blocks: list[str], max_chars: int, sep: str = "\n\n") -> list[str]:
    """Greedily join consecutive blocks while they fit in max_chars."""
    out: list[str] = []
    current = ""
    for b in blocks:
        if not b.strip():
            continue
        candidate = f"{current}{sep}{b}" if current else b
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.append(current)
            current = b
    if current:
        out.append(current)
    return out


def chunk_markdown(raw: str, *, fallback_title: str, max_chars: int = 1200) -> tuple[str, list[ChunkDraft]]:
    """Clean a Markdown page and cut it into chunks no longer than max_chars.

    Returns (title, chunks). Blocks (paragraphs, lists, tables) are never split unless a single
    block is longer than max_chars on its own.
    """
    meta, body = split_front_matter(raw)
    title = str(meta.get("title") or fallback_title).strip()
    chunks: list[ChunkDraft] = []
    for path, section in _sections(clean_markdown(body), title):
        if not section:
            continue
        blocks: list[str] = []
        for block in re.split(r"\n\s*\n", section):
            blocks.extend(_split_long(block, max_chars) if len(block) > max_chars else [block])
        for text in _pack(blocks, max_chars):
            chunks.append(ChunkDraft(seq=len(chunks), heading_path=path, text=text.strip()))
    return title, chunks


def embedding_input(heading_path: str, text: str) -> str:
    """What we embed: the heading path gives a bare passage its subject."""
    return f"{heading_path}\n\n{text}"
