"""Query analysis: estimate category and difficulty before routing.

V1 is a deterministic keyword/regex heuristic. It is deliberately simple and
replaceable: anything implementing `QueryAnalyzer.analyze` (e.g. a learned
classifier trained on logged query records) can be swapped in.
"""

import re
from typing import Protocol

from app.router.schemas import Category, QueryAnalysis

CODE = re.compile(
    r"```|\b(python|javascript|typescript|java|rust|golang|sql|regex|function|method|class|compile|bug|debug|"
    r"implement|refactor|algorithm|recursion|array|linked list|hash ?map|big-?o|api|code|script|unit test)\b",
    re.I,
)
MATH = re.compile(
    r"\d+\s*[-+*/^=%]\s*\d+|\b(calculate|compute|solve|equation|integral|derivative|probability|percent(age)?|"
    r"how many|how much|sum of|product of|average of|remainder|prime|factorial|interest)\b",
    re.I,
)
REASONING = re.compile(
    r"\b(why|explain|compare|contrast|prove|step[- ]by[- ]step|reason|trade-?offs?|analy[sz]e|deduce|infer|"
    r"puzzle|evaluate|justify|what would happen|implications?)\b",
    re.I,
)
SUMMARY = re.compile(r"\b(summari[sz]e|summary|tl;?dr|condense|key points|main points|in \d+ (sentences?|words|bullets?))\b", re.I)
HARD = re.compile(
    r"\b(prove|rigorous(ly)?|optimi[sz]e|edge cases?|concurren(t|cy)|thread-?safe|design a|architecture|complexity|"
    r"counterexample|derive|multi-?step|trick|paradox)\b",
    re.I,
)
KNOWLEDGE = re.compile(r"\b(who|when|where|which|what is|what are|capital|history|define|definition)\b", re.I)

BASE_DIFFICULTY: dict[Category, float] = {
    "general_qa": 0.15,
    "summarization": 0.30,
    "math": 0.40,
    "coding": 0.45,
    "reasoning": 0.45,
}
LONG_CONTEXT_CHARS = 4000
REASONING_STEP_BONUS = 0.08
MAX_REASONING_BONUS = 0.24
HARD_BONUS = 0.15
LENGTH_BONUS = ((200, 0.15), (60, 0.08))  # (min words, bonus), checked in order
SHORT_QUERY_WORDS = 12
SHORT_QUERY_DISCOUNT = 0.05


class QueryAnalyzer(Protocol):
    def analyze(self, query: str) -> QueryAnalysis: ...


class HeuristicAnalyzer:
    """Keyword/regex analyzer. Deterministic and explainable via `signals`."""

    def analyze(self, query: str) -> QueryAnalysis:
        words = len(query.split())
        code, math, summary = bool(CODE.search(query)), bool(MATH.search(query)), bool(SUMMARY.search(query))
        reasoning_hits = len(REASONING.findall(query))
        hard_hits = len(HARD.findall(query))
        category = _category(summary=summary, code=code, math=math, reasoning=reasoning_hits > 0)

        signals = [f"category:{category}"]
        difficulty = BASE_DIFFICULTY[category]
        if reasoning_hits:
            bonus = min(MAX_REASONING_BONUS, REASONING_STEP_BONUS * reasoning_hits)
            difficulty += bonus
            signals.append(f"reasoning_markers:{reasoning_hits}")
        if hard_hits:
            difficulty += HARD_BONUS
            signals.append(f"hard_markers:{hard_hits}")
        if code and math:
            difficulty += REASONING_STEP_BONUS
            signals.append("code+math")
        for min_words, bonus in LENGTH_BONUS:
            if words >= min_words and category != "summarization":  # summaries are long by nature
                difficulty += bonus
                signals.append(f"length>={min_words}w")
                break
        if words < SHORT_QUERY_WORDS and category == "general_qa":
            difficulty -= SHORT_QUERY_DISCOUNT
            signals.append("short_factual")

        return QueryAnalysis(
            category=category,
            difficulty=round(min(1.0, max(0.0, difficulty)), 3),
            requires_reasoning=reasoning_hits > 0,
            requires_code=code,
            requires_math=math,
            requires_knowledge=bool(KNOWLEDGE.search(query)),
            requires_long_context=len(query) > LONG_CONTEXT_CHARS,
            word_count=words,
            signals=signals,
        )


def _category(*, summary: bool, code: bool, math: bool, reasoning: bool) -> Category:
    if summary:
        return "summarization"
    if code:
        return "coding"
    if math:
        return "math"
    if reasoning:
        return "reasoning"
    return "general_qa"
