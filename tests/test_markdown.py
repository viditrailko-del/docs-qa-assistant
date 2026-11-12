"""Cleaning and chunking: unit cases plus invariants over the whole real corpus."""

import json
import re

import pytest

from app.ingestion.markdown import chunk_markdown, clean_markdown, embedding_input, split_front_matter
from scripts.check_questions import normalize
from scripts.snapshot_corpus import DOCS, MANIFEST

PAGE = """---
title: "Travel Policy"
description: test
---

Intro line.

## Airfare

### Advance Purchase

- Domestic: At least 14 days in advance
- See [the policy](https://example.com/a_(b)) for details.

{{% panel header="**Coding**" header-bg="green" %}}
**Expense Type:** Airfare
{{% /panel %}}

<details markdown="1">
<summary>Key Details</summary>

Up to $300 per night.<br>Tax included.
</details>

<!-- editor note: hidden -->
{{< youtube "abc123" >}}

## Hotels

---

Hotel text.
"""


def test_front_matter_is_parsed_and_removed():
    meta, body = split_front_matter(PAGE)
    assert meta["title"] == "Travel Policy"
    assert not body.startswith("---")


def test_front_matter_missing_or_broken():
    assert split_front_matter("# Hi\ntext") == ({}, "# Hi\ntext")
    meta, body = split_front_matter("---\n: [broken\n---\nbody\n")
    assert meta == {} and body == "body\n"


def test_clean_removes_noise_and_keeps_labels():
    text = clean_markdown(split_front_matter(PAGE)[1])
    for noise in ("{{", "<details", "<summary", "<br", "<!--", "editor note", "youtube", "https://"):
        assert noise not in text
    assert "See the policy for details." in text  # link text kept, URL (with parentheses) dropped
    assert "**Coding**" in text and "**Key Details**" in text  # panel header / summary kept as context
    assert "Up to $300 per night. Tax included." in text


def test_chunks_follow_heading_paths():
    title, chunks = chunk_markdown(PAGE, fallback_title="x")
    assert title == "Travel Policy"
    paths = [c.heading_path for c in chunks]
    assert paths == ["Travel Policy", "Travel Policy > Airfare > Advance Purchase", "Travel Policy > Hotels"]
    assert [c.seq for c in chunks] == [0, 1, 2]
    assert "At least 14 days" in chunks[1].text
    assert chunks[2].text == "Hotel text."  # horizontal rule dropped


def test_heading_inside_code_fence_is_not_a_section():
    raw = "# Page\n\n```\n# not a heading\n```\n\nafter"
    _, chunks = chunk_markdown(raw, fallback_title="x")
    assert len(chunks) == 1 and "# not a heading" in chunks[0].text


def test_fallback_title_and_sibling_headings_reset_path():
    raw = "## A\n\n### A1\n\none\n\n## B\n\ntwo"
    title, chunks = chunk_markdown(raw, fallback_title="file")
    assert title == "file"
    assert [c.heading_path for c in chunks] == ["file > A > A1", "file > B"]


def test_long_text_is_split_within_limit():
    para = " ".join(f"Sentence number {i} is here." for i in range(200))
    _, chunks = chunk_markdown(f"# T\n\n{para}\n\nsecond paragraph", fallback_title="x", max_chars=300)
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 for c in chunks)
    assert chunks[-1].text.endswith("second paragraph")


def test_word_longer_than_limit_is_hard_cut():
    _, chunks = chunk_markdown("x" * 1000, fallback_title="x", max_chars=300)
    assert all(len(c.text) <= 300 for c in chunks) and sum(len(c.text) for c in chunks) == 1000


def test_empty_page_has_no_chunks():
    assert chunk_markdown("---\ntitle: t\n---\n\n<!-- only a comment -->\n", fallback_title="x")[1] == []


def test_embedding_input_puts_heading_first():
    assert embedding_input("A > B", "text") == "A > B\n\ntext"


# --- invariants over the real corpus ---------------------------------------------------------

def _corpus():
    for f in json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]:
        yield f["path"], chunk_markdown((DOCS / f["path"]).read_text(encoding="utf-8"), fallback_title=f["path"])


@pytest.fixture(scope="module")
def corpus_chunks():
    return dict(_corpus())


def test_corpus_chunks_are_clean_and_bounded(corpus_chunks):
    for path, (title, chunks) in corpus_chunks.items():
        assert title, path
        for c in chunks:
            assert c.text.strip(), path
            assert len(c.text) <= 1200, path
            assert not re.search(r"\{\{|<details|</?summary|<!--|\]\(https?://", c.text), (path, c.text[:100])


def test_every_eval_quote_lands_inside_one_chunk(corpus_chunks):
    """A quote split across two chunks could never be retrieved or cited as one passage."""
    questions = json.loads((DOCS.parents[1] / "eval/questions/questions.json").read_text(encoding="utf-8"))
    by_path = {p: [normalize(c.text) for c in chunks] for p, (_, chunks) in corpus_chunks.items()}
    for q in questions["questions"]:
        for s in q.get("sources", []) + q.get("near_miss", []):
            needle = normalize(clean_markdown(s["quote"]))
            assert any(needle in c for c in by_path[s["path"]]), (q["id"], s["quote"][:60])
