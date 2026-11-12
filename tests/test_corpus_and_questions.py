"""The frozen corpus matches its manifest, and every question label checks out against it."""

from scripts.check_questions import check
from scripts.snapshot_corpus import verify


def test_corpus_matches_manifest():
    verify()  # exits (SystemExit) on any changed, missing or extra file


def test_question_labels_are_valid():
    assert check() == []
