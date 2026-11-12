"""Scope check: a general statement backed only by country/entity-specific sources.

Found in the Phase 3 smoke test: asked "how many weeks of parental leave does GitLab give?",
the model answered "16 weeks" citing only the Canada and UK pages, because the company-wide
passage was not retrieved. The quotes were exact, so the citation checks passed, but the
evidence doesn't support a company-wide claim. Here that happened to be true; for another
benefit the same move would state one country's rule as everyone's.

A source's scope comes from its title, section path and file path (e.g. "GitLab Canada Corp",
"Global Expansion > Kenya", "pty-benefits-australia"), never from its body text, which mentions
countries in passing. Flag (warning) when the question and the statement name no country but
every cited source belongs to one.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from app.checks.base import Flag, Severity, SourceText
from app.schemas.answer import LLMAnswer

# country -> pattern for names/adjectives in prose. "US" is case-sensitive: "us" is a word.
_COUNTRIES: dict[str, str] = {
    "Australia": r"(?i:\baustralia\w*)",
    "New Zealand": r"(?i:\bnew zealand\b|\bNZ\b)",
    "Austria": r"(?i:\baustria\w*)",
    "Belgium": r"(?i:\bbelgi\w+)",
    "Brazil": r"(?i:\bbrazil\w*)",
    "Canada": r"(?i:\bcanad\w+)",
    "Denmark": r"(?i:\bdenmark\b|\bdanish\b)",
    "France": r"(?i:\bfrance\b|\bfrench\b)",
    "Germany": r"(?i:\bgerman\w*|\bgmbh\b)",
    "Hungary": r"(?i:\bhungar\w+)",
    "India": r"(?i:\bindia\w*)",
    "Ireland": r"(?i:\bireland\b|\birish\b)",
    "Israel": r"(?i:\bisrael\w*)",
    "Italy": r"(?i:\bital(?:y|ian)\w*)",
    "Japan": r"(?i:\bjapan\w*)",
    "Kenya": r"(?i:\bkenya\w*)",
    "Korea": r"(?i:\bkorea\w*)",
    "Luxembourg": r"(?i:\bluxembourg\w*)",
    "Mexico": r"(?i:\bmexic\w+)",
    "Netherlands": r"(?i:\bnetherlands\b|\bdutch\b)",
    "Philippines": r"(?i:\bphilippin\w+|\bfilipino\b)",
    "Poland": r"(?i:\bpoland\b|\bpolish\b)",
    "Singapore": r"(?i:\bsingapor\w+)",
    "South Africa": r"(?i:\bsouth africa\w*)",
    "Spain": r"(?i:\bspain\b|\bspanish\b|\biberia\b)",
    "Sweden": r"(?i:\bswed\w+)",
    "Switzerland": r"(?i:\bswitzerland\b|\bswiss\b)",
    "UAE": r"(?i:\bunited arab emirates\b)|\bUAE\b",
    "UK": r"\bUK\b|\bU\.K\.|(?i:\bunited kingdom\b|\bbritain\b|\bbritish\b|\bengland\b)",
    "US": r"\bUSA?\b|\bU\.S\.(?:A\.)?|(?i:\bunited states\b|\bamerican\b)",
}
_PATTERNS = {name: re.compile(p) for name, p in _COUNTRIES.items()}

# file-path tokens (lowercase, split on - _ / .) -> country
_PATH_TOKENS = {
    "australia": "Australia", "zealand": "New Zealand", "belgium": "Belgium", "netherlands": "Netherlands",
    "canada": "Canada", "france": "France", "gk": "Japan", "ireland": "Ireland", "korea": "Korea",
    "uk": "UK", "us": "US", "usa": "US", "singapore": "Singapore", "india": "India", "italy": "Italy",
}


def countries_in(text: str) -> set[str]:
    return {name for name, pat in _PATTERNS.items() if pat.search(text)}


def source_scope(src: SourceText) -> set[str]:
    tokens = set(re.split(r"[-_/.]", src.path.lower()))
    return countries_in(f"{src.title or ''} > {src.heading_path}") | {
        _PATH_TOKENS[t] for t in tokens if t in _PATH_TOKENS
    }


def check_scope(question: str, answer: LLMAnswer, sources: Sequence[SourceText]) -> list[Flag]:
    if countries_in(question):
        return []  # the question is about a specific country: country sources are the right ones
    by_label = {s.label: s for s in sources}
    flags: list[Flag] = []
    for i, statement in enumerate(answer.statements):
        if countries_in(statement.text):
            continue  # the statement itself says which country it's about
        cited = [by_label[c.source.strip()] for c in statement.citations if c.source.strip() in by_label]
        scopes = [source_scope(s) for s in cited]
        if cited and all(scopes):
            names = sorted(set().union(*scopes))
            flags.append(Flag("scope_mismatch", Severity.warning,
                              f"states a general rule, but its sources are specific to {', '.join(names)}", i))
    return flags
