"""Verification checks: each one gets a passing case, a failing case and edge cases."""

from decimal import Decimal

import pytest

from app.checks.base import SourceText, normalize
from app.checks.citations import check_citations
from app.checks.numbers import check_numbers, numbers_in
from app.checks.runner import run_checks
from app.checks.scope import check_scope, countries_in, source_scope
from app.schemas.answer import LLMAnswer


def src(label="S1", text="Limit is set to $60/year.", heading="Expenses > Books", title="Expenses",
        path="finance/expenses.md", similarity=0.8) -> SourceText:
    return SourceText(label=label, text=text, heading_path=heading, title=title, path=path, similarity=similarity)


def answer(*statements: tuple[str, list[tuple[str, str]]], disagree=False) -> LLMAnswer:
    return LLMAnswer(answerable=True, sources_disagree=disagree, statements=[
        {"text": t, "citations": [{"source": s, "quote": q} for s, q in cits]} for t, cits in statements])


# --- normalize ------------------------------------------------------------------------------

def test_normalize_forgives_only_formatting():
    assert normalize("“Dynamic  Policy” – **bold** `code`\n") == normalize('"dynamic policy" - bold code')
    assert normalize("16 weeks") != normalize("15 weeks")


# --- citations -----------------------------------------------------------------------------

def test_exact_quote_passes():
    a = answer(("Books: $60 a year.", [("S1", "Limit is set to $60/year.")]))
    assert check_citations(a, [src()]) == []


def test_quote_with_formatting_differences_passes():
    s = src(text='Uses a **“Dynamic Policy”** which\ncalculates a price.')
    a = answer(("x", [("S1", 'uses a "Dynamic Policy" which calculates')]))
    assert check_citations(a, [s]) == []


def test_invented_quote_is_flagged():
    a = answer(("Books: $80 a year.", [("S1", "Limit is set to $80/year.")]))
    [f] = check_citations(a, [src()])
    assert (f.code, f.severity, f.statement, f.citation) == ("quote_not_in_source", "error", 0, 0)


def test_quote_from_the_wrong_source_is_flagged():
    a = answer(("x", [("S2", "Limit is set to $60/year.")]))
    [f] = check_citations(a, [src("S1"), src("S2", text="Hotel stays up to $300.")])
    assert f.code == "quote_not_in_source" and "S2" in f.reason


def test_joined_passages_get_a_hint():
    a = answer(("x", [("S1", "Limit is set ... per year")]))
    assert "joined" in check_citations(a, [src()])[0].reason


def test_unknown_label_is_flagged():
    [f] = check_citations(answer(("x", [("S9", "anything")])), [src("S1"), src("S2")])
    assert f.code == "unknown_source" and "S1, S2" in f.reason


# --- numbers -------------------------------------------------------------------------------

@pytest.mark.parametrize("text,expected", [
    ("$1,500 base bonus", {Decimal(1500)}),
    ("1500 and 1,500.00", {Decimal(1500)}),
    ("2-3 months, 6'5\" tall, 20%", {Decimal(2), Decimal(3), Decimal(6), Decimal(5), Decimal(20)}),
    ("see S1 and v2", set()),  # glued to a letter: a label, not a quantity
    ("4.1 NON-TRAVEL and $0.50/mile.", {Decimal("4.1"), Decimal("0.5")}),
    ("no numbers here", set()),
])
def test_numbers_in(text, expected):
    assert numbers_in(text) == expected


def test_supported_numbers_pass_in_any_format():
    s = src(text="Bonus of [$1,500](/link) after 3 months.")
    a = answer(("The bonus is 1500 USD, paid after 3 months.", [("S1", "after 3 months")]))
    assert check_numbers(a, [s]) == []


def test_number_in_heading_counts():
    s = src(text="Company match: 50% of contributions.", heading="US Benefits > 401(k) Plan")
    assert check_numbers(answer(("The 401(k) match is 50%.", [("S1", "50% of contributions")])), [s]) == []


def test_unsupported_number_is_flagged():
    a = answer(("Books are limited to $80 a year.", [("S1", "Limit is set to $60/year.")]))
    [f] = check_numbers(a, [src()])
    assert (f.code, f.severity, f.statement) == ("number_unsupported", "error", 0) and "80" in f.reason


def test_computed_number_is_flagged():
    a = answer(("Together that is $380.", [("S1", "x"), ("S2", "y")]))
    srcs = [src("S1", text="Hotel $300."), src("S2", text="Meals $80.")]
    assert check_numbers(a, srcs)[0].reason.startswith("380 not found")


def test_number_only_in_an_uncited_source_is_flagged():
    a = answer(("Hotels are $300.", [("S1", "Limit is set to $60/year.")]))
    assert check_numbers(a, [src("S1"), src("S2", text="Hotel stays up to $300.")])[0].code == "number_unsupported"


# --- scope ---------------------------------------------------------------------------------

CANADA = src("S1", text="GitLab's Parental Leave supplements government benefits to provide 100% salary for up to 16 weeks.",
             heading="GitLab Canada Corp > Parental Leave", title="GitLab Canada Corp",
             path="total-rewards/benefits/general-and-entity-benefits/canada-corp-benefits.md")
UK = src("S2", text="GitLab's enhanced policy provides up to 16 weeks at 100% salary",
         heading="GitLab LTD (UK) Benefits > Maternity Leave", title="GitLab LTD (UK) Benefits",
         path="total-rewards/benefits/general-and-entity-benefits/ltd-benefits-uk.md")
GLOBAL = src("S3", text="you'll have 16 weeks of paid leave", title="General & Entity Specific Benefits & Information",
             heading="General & Entity Specific Benefits & Information > Parental Leave",
             path="total-rewards/benefits/general-and-entity-benefits/_index.md")


def test_general_claim_from_country_sources_is_flagged():
    """The Phase 3 smoke-test case: right number, wrong evidence."""
    a = answer(("GitLab's parental leave generally provides up to 16 weeks at 100% salary.",
                [("S1", "provide 100% salary for up to 16 weeks"), ("S2", "up to 16 weeks at 100% salary")]))
    [f] = check_scope("How many weeks of paid parental leave does GitLab give?", a, [CANADA, UK])
    assert (f.code, f.severity) == ("scope_mismatch", "warning") and "Canada, UK" in f.reason


def test_general_claim_with_a_company_wide_source_passes():
    a = answer(("Parental leave is 16 weeks.", [("S1", "16 weeks"), ("S3", "16 weeks of paid leave")]))
    assert check_scope("How many weeks of parental leave?", a, [CANADA, UK, GLOBAL]) == []


def test_country_named_in_question_or_statement_passes():
    a = answer(("Parental leave is 16 weeks.", [("S1", "16 weeks")]))
    assert check_scope("How much parental leave in Canada?", a, [CANADA]) == []
    b = answer(("In Canada, parental leave is topped up to 16 weeks.", [("S1", "16 weeks")]))
    assert check_scope("How much parental leave?", b, [CANADA]) == []


def test_scope_from_path_when_title_is_generic():
    au = src(title="PTY LTD Benefits", heading="PTY LTD Benefits > Leave",
             path="total-rewards/benefits/general-and-entity-benefits/pty-benefits-australia.md")
    assert source_scope(au) == {"Australia"}
    assert source_scope(GLOBAL) == set()
    assert source_scope(src(heading="Global Expansion > Kenya 🇰🇪", title="Global Expansion",
                            path="total-rewards/benefits/general-and-entity-benefits/global-expansion.md")) == {"Kenya"}


def test_country_words_need_care_with_us():
    assert countries_in("Tell us about leave") == set()
    assert countries_in("US team members") == {"US"} and countries_in("the U.S. policy") == {"US"}
    assert countries_in("Irish and British rules") == {"Ireland", "UK"}


# --- runner --------------------------------------------------------------------------------

def test_refusal_has_nothing_to_check():
    refusal = LLMAnswer(answerable=False, statements=[], refusal_reason="not covered")
    assert run_checks("q", refusal, [src()]) == []


def test_runner_combines_checks_and_weak_retrieval_is_opt_in():
    a = answer(("Books are limited to $80 a year.", [("S1", "Limit is set to $80/year.")]))
    codes = {f.code for f in run_checks("books?", a, [src(similarity=0.5)])}
    assert codes == {"quote_not_in_source", "number_unsupported"}
    codes = {f.code for f in run_checks("books?", a, [src(similarity=0.5)], min_similarity=0.6)}
    assert "weak_retrieval" in codes


def test_clean_answer_has_no_flags():
    a = answer(("Books are limited to $60 a year.", [("S1", "Limit is set to $60/year.")]))
    assert run_checks("How much for books?", a, [src()]) == []


def test_flag_serializes_to_plain_json():
    [f] = check_citations(answer(("x", [("S9", "q")])), [src()])
    assert f.to_dict() == {"code": "unknown_source", "severity": "error", "reason": f.reason,
                           "statement": 0, "citation": 0}
