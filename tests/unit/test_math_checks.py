"""Deterministic math checks from question + answer only (no answer key)."""

import pytest

from app.verification.math_checks import check_math


def outcome(query, answer):
    result = check_math(query, answer)
    return None if result is None else (result.status, result.evidence)


# ---- the three required cases ----

def test_correct_math_answer_is_independently_verified():
    assert outcome("What is 17 × 23?", "17 × 23 = 391. Final answer: 391") == ("pass", "passed")


def test_wrong_math_answer_is_independently_contradicted():
    result = check_math("What is 17 × 23?", "Final answer: 381")
    assert (result.status, result.evidence) == ("fail", "math_mismatch")
    assert "391" in result.detail and "381" in result.detail


def test_ambiguous_math_answer_is_not_checkable():
    assert outcome("What is 17 × 23?", "It is either 391 or 381, depending on rounding.") == ("not_checkable", "not_checkable")


# ---- supported shapes ----

@pytest.mark.parametrize("query,answer", [
    ("Compute 2^10 + 3.", "1027"),
    ("Calculate (12 - 4) / 2.", "The result is 4."),
    ("Evaluate 7 mod 3", "Final answer: 1"),
    ("What is the remainder when 2^10 is divided by 7?", "1024 = 7 * 146 + 2. Final answer: 2"),
    ("What is the remainder when 123456789^987 is divided by 11?", "Final answer: 3"),  # fast modular power
    ("What is 10 / 3?", "About 3.33"),  # half a unit of the last shown digit
])
def test_supported_arithmetic_passes(query, answer):
    assert outcome(query, answer) == ("pass", "passed")


def test_integer_answer_to_non_integer_result_fails():
    assert outcome("What is 10 / 4?", "Final answer: 2") == ("fail", "math_mismatch")
    assert outcome("What is 10 / 4, rounded?", "Final answer: 3") is None  # extra words: shape not recognized


@pytest.mark.parametrize("query,answer,expected", [
    ("Solve for x: 3x + 7 = 22.", "3x = 15, so x = 5.", "pass"),
    ("Solve for x: 3x + 7 = 22.", "x = 6", "fail"),
    ("Solve: 2(x + 1) = 10", "x = 4", "pass"),
    ("Solve for x: 3x = 7", "x = 2.33", "pass"),  # rounded root: exact root lies within +-0.005
    ("Solve for x: 3x = 7", "x = 2.4", "fail"),
    ("Solve for x: x^2 = 9", "x = -3", "pass"),
])
def test_equation_by_substitution(query, answer, expected):
    assert check_math(query, answer).status == expected


# ---- deliberately NOT interpreted ----

@pytest.mark.parametrize("query", [
    "What is 15% of 240?",                        # percent vs modulo ambiguity
    "What is 6/2(1+2)?",                          # implicit multiplication ambiguity
    "How many ordered triples satisfy x + y + z = 20 with each at most 8?",  # word problem
    "What is the capital of France?",
    "What is 2 + 2 in binary?",                   # extra words change the meaning
    "Solve for x and y: x + y = 3",               # more than one unknown
    "What is 12?",                                # no operation
    "What is 2^99999999?",                        # too large to evaluate safely
    "What is 1/0?",
])
def test_unrecognized_or_unsafe_questions_return_none(query):
    assert check_math(query, "Final answer: 4") is None


def test_fraction_or_scientific_answers_are_not_checkable():
    assert outcome("What is 1 / 3?", "Final answer: 1/3") == ("not_checkable", "not_checkable")
    assert outcome("What is 1000 * 1000?", "Final answer: 1e6") == ("not_checkable", "not_checkable")


def test_equation_answer_without_a_value_is_not_checkable():
    assert outcome("Solve for x: 3x + 7 = 22.", "There is no real solution.") == ("not_checkable", "not_checkable")


def test_several_numbers_in_the_final_sentence_without_a_marker_is_not_checkable():
    """Conservative by design: '1024 = 7 * 146 + 2, so the remainder is 2.' has five numbers and no marker."""
    assert outcome("What is the remainder when 2^10 is divided by 7?", "1024 = 7 * 146 + 2, so the remainder is 2.") == (
        "not_checkable", "not_checkable")
