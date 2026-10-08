"""Reading an answer's final number. Pure text parsing: no answer key involved.

Shared by runtime verification (math checks) and the benchmark's exact checks.
"""

import re
from typing import Literal

FINAL_ANSWER = re.compile(r"final answer\s*[:=]?\s*\**\s*\$?\s*(-?[\d,]*\.?\d+)", re.I)
THOUSANDS_SEP = re.compile(r"(?<=\d)[, \u00a0\u2009\u202f](?=\d{3}(?!\d))")  # comma, space, nbsp, thin spaces
NUMBER = re.compile(r"-?\d[\d,]*\.?\d*|-?\.\d+")
BOXED = re.compile(r"\\boxed\{\s*\$?\s*(-?[\d,]*\.?\d+)")
SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
SEGMENT_TAIL = 20  # characters after a marked number, to see what follows it (e.g. "/3")

Status = Literal["found", "none", "ambiguous"]


def final_number_token(text: str) -> tuple[Status, str | None, str]:
    """(status, the final number as written without separators, the text segment it was read from).

    An explicit 'Final answer: X' or \\boxed{X} wins; otherwise the last sentence that contains numbers
    must contain exactly one distinct number. Anything else is "ambiguous" (e.g. "204 squares on an
    8 x 8 board") and must be treated as not checkable rather than guessed.
    """
    text = THOUSANDS_SEP.sub("", text)  # "34 650" / "34,650" -> "34650"
    for pattern in (FINAL_ANSWER, BOXED):
        marked = list(pattern.finditer(text))
        if marked:
            last = marked[-1]
            return "found", _clean(last.group(1)), text[last.start(1) : last.end() + SEGMENT_TAIL]
    if not NUMBER.search(text):
        return "none", None, ""
    sentence = [part for part in SENTENCE_BREAK.split(text.strip()) if NUMBER.search(part)][-1]
    tokens = [_clean(t) for t in NUMBER.findall(sentence)]
    values = {_to_float(t) for t in tokens} - {None}
    if len(values) != 1:
        return "ambiguous", None, sentence
    value = values.pop()
    return "found", next(t for t in tokens if _to_float(t) == value), sentence


def read_final_number(text: str) -> tuple[Status, float | None]:
    status, token, _ = final_number_token(text)
    return status, (_to_float(token) if token is not None else None)


def extract_number(text: str) -> float | None:
    return read_final_number(text)[1]


def _clean(raw: str) -> str:
    return raw.replace(",", "").rstrip(".")


def _to_float(raw: str) -> float | None:
    try:
        return float(raw)
    except ValueError:
        return None
