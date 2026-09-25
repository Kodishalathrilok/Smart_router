import pytest

from app.core.analyzer import HeuristicAnalyzer

analyze = HeuristicAnalyzer().analyze


@pytest.mark.parametrize("query,category", [
    ("What is the capital of France?", "general_qa"),
    ("Write a Python function to reverse a linked list.", "coding"),
    ("Calculate 17 * 23 + 4.", "math"),
    ("Explain why the sky is blue and compare it to sunsets.", "reasoning"),
    ("Summarize the following text in 2 sentences: ...", "summarization"),
])
def test_category(query, category):
    assert analyze(query).category == category


def test_short_factual_is_easy_and_hard_reasoning_is_hard():
    easy = analyze("Who wrote Hamlet?")
    hard = analyze("Prove step by step why the halting problem is undecidable, analyze the implications and justify each step rigorously.")
    assert easy.difficulty < 0.35
    assert hard.difficulty >= 0.70
    assert hard.requires_reasoning and "hard_markers:2" in hard.signals


def test_flags_and_long_context():
    a = analyze("Implement a function that computes 2 + 2 " + "word " * 1000)
    assert a.requires_code and a.requires_math and a.requires_long_context
    assert 0.0 <= a.difficulty <= 1.0


def test_deterministic():
    q = "Explain the tradeoffs of optimistic vs pessimistic locking."
    assert analyze(q) == analyze(q)
