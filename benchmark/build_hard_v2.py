"""Build benchmark/datasets/hard_v2.jsonl: python -m benchmark.build_hard_v2

Every expected value is COMPUTED here (brute force or exact formula), never typed in, so the answer key
is reproducible and auditable. Coding items ship a reference solution that must pass its own tests
(enforced by tests/unit/test_benchmark_evaluators.py).

Design intent: the earlier sets were saturated (the cheap model scored ~94% on `hard`). These items need
long exact multi-step work (counting with constraints, exact modular/combinatorial arithmetic) or code
with subtle edge cases and a performance test. Whether they actually separate the model tiers is NOT
known until a real run measures it; treat the set as a hypothesis.
"""

import json
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import comb, factorial, gcd
from pathlib import Path

OUT = Path(__file__).parent / "datasets" / "hard_v2.jsonl"


def phi(n: int) -> int:
    return sum(1 for k in range(1, n + 1) if gcd(k, n) == 1)


def trailing_zeros(n: int) -> int:
    z, p = 0, 5
    while p <= n:
        z += n // p
        p *= 5
    return z


def legendre(n: int, p: int) -> int:
    e, q = 0, p
    while q <= n:
        e += n // q
        q *= p
    return e


def stirling2(n: int, k: int) -> int:
    @lru_cache(None)
    def s(n, k):
        if n == k:
            return 1
        if k == 0 or k > n:
            return 0
        return k * s(n - 1, k) + s(n - 1, k - 1)

    return s(n, k)


def derangements(n: int) -> int:
    d = [1, 0]
    for i in range(2, n + 1):
        d.append((i - 1) * (d[-1] + d[-2]))
    return d[n]


def coin_ways(total: int, coins: list[int]) -> int:
    ways = [1] + [0] * total
    for c in coins:
        for v in range(c, total + 1):
            ways[v] += ways[v - c]
    return ways[total]


def binary_strings(n: int) -> int:
    """Length-n binary strings with no '11' and no '000'."""
    return sum(1 for bits in product("01", repeat=n) if "11" not in (s := "".join(bits)) and "000" not in s)


def bounded_compositions(total: int, parts: int, lo: int, hi: int) -> int:
    @lru_cache(None)
    def f(t, k):
        if k == 0:
            return int(t == 0)
        return sum(f(t - v, k - 1) for v in range(lo, hi + 1) if v <= t)

    return f(total, parts)


def smallest_n_with_zeros(z: int) -> int:
    n = 0
    while trailing_zeros(n) < z:
        n += 5
    return n


def four_digit_digit_sum(target: int) -> int:
    return sum(1 for n in range(1000, 10000) if sum(map(int, str(n))) == target)


def numeric(i, category, question, value, reference, tolerance=None):
    item = {"id": f"{category}_v2_{i:02d}", "category": category, "difficulty": "hard", "question": question,
            "reference_answer": f"{reference} Final answer: {value}", "answer_type": "numeric", "expected": value}
    if tolerance is not None:
        item["tolerance"] = tolerance
    return item


def math_items() -> list[dict]:
    necklaces = sum(phi(d) * 3 ** (8 // d) for d in (1, 2, 4, 8)) // 8
    circle = sum(1 for x in range(-10, 11) for y in range(-10, 11) if x * x + y * y < 100)
    v2, v3 = legendre(2026, 2), legendre(2026, 3)
    base12 = min(v2 // 2, v3)
    die8 = float(round(8 * sum(Fraction(1, k) for k in range(1, 9)), 4))
    rows = [
        ("How many surjective (onto) functions are there from a 7-element set onto a 4-element set?", factorial(4) * stirling2(7, 4), "4!·S(7,4)"),
        ("How many distinct necklaces of 8 beads can be made with beads of 3 colours, where two necklaces are the same if one is a rotation of the other (reflections count as different)?", necklaces, "Burnside over rotations"),
        ("How many points (x, y) with integer coordinates satisfy x^2 + y^2 < 100?", circle, "count lattice points"),
        ("What is the remainder when 2^2026 + 3^2026 is divided by 13?", (pow(2, 2026, 13) + pow(3, 2026, 13)) % 13, "Fermat, orders mod 13"),
        ("How many trailing zeros does 2026! have when written in base 12?", base12, f"min(v2/2, v3) = min({v2 // 2}, {v3})"),
        ("What is the smallest positive integer n such that n! ends in at least 1000 zeros (in base 10)?", smallest_n_with_zeros(1000), "Legendre"),
        ("How many ordered 4-tuples of integers (a, b, c, d) with 1 <= a, b, c, d <= 10 satisfy a + b + c + d = 30?", bounded_compositions(30, 4, 1, 10), "inclusion-exclusion"),
        ("How many 4-digit positive integers have digits that sum to 20?", four_digit_digit_sum(20), "count"),
        ("What is the sum of all positive divisors of 2^10 · 3^5?", (2**11 - 1) * (3**6 - 1) // 2, "sigma is multiplicative"),
        ("In how many ways can 8 non-attacking rooks be placed on an 8 × 8 board so that no rook is on the main diagonal (top-left to bottom-right)?", derangements(8), "derangements of 8"),
        ("A fair 8-sided die is rolled until every face has appeared at least once. What is the expected number of rolls? Give a decimal to 4 places.", die8, "8·H(8)", 0.00051),
        ("How many binary strings of length 20 contain neither two consecutive 1s nor three consecutive 0s?", binary_strings(20), "DP over suffixes"),
    ]
    return [numeric(i + 1, "math", *row) for i, row in enumerate(rows)]


def reasoning_items() -> list[dict]:
    rows = [
        ("There are 5 doors; one hides a car. You pick a door. The host, who knows where the car is, opens 2 of the other doors, both with goats, and offers you a switch to one of the 2 remaining closed doors chosen at random. What is the probability that switching wins the car? Give a decimal.",
         0.4, "P(first pick wrong)·1/2 = 4/5·1/2", 0.0005),
        ("Using coins worth 1, 2, 5, 10, 20, 50 and 100 cents, in how many different ways can you make exactly 250 cents? The order of coins does not matter.",
         coin_ways(250, [1, 2, 5, 10, 20, 50, 100]), "coin-change DP"),
        ("A clock's hour and minute hands overlap at 12:00. How many times do they overlap in the 24 hours starting at 00:00 and ending just before the next 00:00 (count 00:00 itself once)?",
         22, "every 12/11 hours"),
        ("A bag holds 4 red, 5 blue and 6 green marbles. Three are drawn without replacement. What is the probability that all three have different colours? Give a decimal to 4 places.",
         float(round(Fraction(4 * 5 * 6, comb(15, 3)), 4)), "4·5·6 / C(15,3)", 0.00051),
        ("Each of 6 people writes their name on a card; the cards are shuffled and dealt back one per person. What is the probability that exactly 2 people get their own card? Give a decimal to 4 places.",
         float(round(Fraction(comb(6, 2) * derangements(4), factorial(6)), 4)), "C(6,2)·D(4)/6!", 0.00051),
    ]
    return [numeric(i + 1, "reasoning", *row) for i, row in enumerate(rows)]


CODING = [
    ("count_inversions", "Write a Python function `count_inversions(nums: list[int]) -> int` returning the number of pairs i < j with nums[i] > nums[j]. It must handle 200,000 elements in well under a second (O(n log n)).",
     '''def count_inversions(nums):
    def sort(a):
        if len(a) <= 1:
            return a, 0
        mid = len(a) // 2
        left, x = sort(a[:mid])
        right, y = sort(a[mid:])
        merged, inv, i, j = [], x + y, 0, 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                merged.append(left[i]); i += 1
            else:
                merged.append(right[j]); j += 1; inv += len(left) - i
        merged += left[i:] + right[j:]
        return merged, inv
    return sort(list(nums))[1]''',
     "assert count_inversions([]) == 0\nassert count_inversions([1, 2, 3]) == 0\nassert count_inversions([3, 2, 1]) == 3\n"
     "assert count_inversions([2, 2, 1]) == 2\nassert count_inversions([1, 3, 2, 3, 1]) == 4\n"
     "import time\n_n = 200000\n_t = time.perf_counter()\nassert count_inversions(list(range(_n, 0, -1))) == _n * (_n - 1) // 2\nassert time.perf_counter() - _t < 5"),
    ("num_decodings", "Write a Python function `num_decodings(s: str) -> int`: letters A-Z are encoded as '1'..'26'; return how many ways the digit string s can be decoded. Return 0 when no decoding exists (for example '0', '06', '100'), and 1 for the empty string.",
     '''def num_decodings(s):
    prev, cur = 1, 1
    for i in range(len(s)):
        nxt = cur if s[i] != "0" else 0
        if i and s[i - 1] != "0" and 10 <= int(s[i - 1:i + 1]) <= 26:
            nxt += prev
        prev, cur = cur, nxt
    return cur''',
     "assert num_decodings('') == 1\nassert num_decodings('12') == 2\nassert num_decodings('226') == 3\nassert num_decodings('0') == 0\n"
     "assert num_decodings('06') == 0\nassert num_decodings('100') == 0\nassert num_decodings('2101') == 1\nassert num_decodings('11106') == 2\n"
     "assert num_decodings('1' * 40) == 165580141"),
    ("min_window", "Write a Python function `min_window(s: str, t: str) -> str` returning the shortest substring of s that contains every character of t including duplicates (case-sensitive). If several are shortest, return the leftmost; return '' if none exists or t is empty.",
     '''def min_window(s, t):
    from collections import Counter
    if not t:
        return ""
    need, missing = Counter(t), len(t)
    best, left = (0, float("inf")), 0
    for right, c in enumerate(s, 1):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        if missing == 0:
            while need[s[left]] < 0:
                need[s[left]] += 1
                left += 1
            if right - left < best[1] - best[0]:
                best = (left, right)
            need[s[left]] += 1
            missing += 1
            left += 1
    return "" if best[1] == float("inf") else s[best[0]:best[1]]''',
     "assert min_window('ADOBECODEBANC', 'ABC') == 'BANC'\nassert min_window('a', 'aa') == ''\nassert min_window('aa', 'aa') == 'aa'\n"
     "assert min_window('abc', '') == ''\nassert min_window('abcabc', 'cb') == 'bc'\nassert min_window('aAbB', 'AB') == 'AbB'"),
    ("merge_intervals", "Write a Python function `merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]` that merges overlapping closed intervals and also merges intervals that merely touch (e.g. (1, 3) and (3, 5)). Input may be unsorted; return tuples sorted by start.",
     '''def merge_intervals(intervals):
    out = []
    for a, b in sorted(intervals):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out''',
     "assert merge_intervals([]) == []\nassert merge_intervals([(1, 3), (2, 6), (8, 10)]) == [(1, 6), (8, 10)]\n"
     "assert merge_intervals([(3, 5), (1, 3)]) == [(1, 5)]\nassert merge_intervals([(1, 10), (2, 3)]) == [(1, 10)]\n"
     "assert merge_intervals([(5, 5), (1, 2)]) == [(1, 2), (5, 5)]"),
    ("lis_length", "Write a Python function `lis_length(nums: list[int]) -> int` returning the length of the longest STRICTLY increasing subsequence. It must handle 100,000 elements quickly (O(n log n)).",
     '''def lis_length(nums):
    import bisect
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)''',
     "assert lis_length([]) == 0\nassert lis_length([10, 9, 2, 5, 3, 7, 101, 18]) == 4\nassert lis_length([7, 7, 7]) == 1\n"
     "assert lis_length([0, 1, 0, 3, 2, 3]) == 4\nimport time\n_t = time.perf_counter()\nassert lis_length(list(range(100000))) == 100000\nassert time.perf_counter() - _t < 5"),
]


def coding_items() -> list[dict]:
    return [{"id": f"coding_v2_{i + 1:02d}", "category": "coding", "difficulty": "hard", "question": q,
             "reference_answer": f"```python\n{src}\n```", "answer_type": "code", "tests": tests}
            for i, (_, q, src, tests) in enumerate(CODING)]


def build() -> list[dict]:
    items = math_items() + reasoning_items() + coding_items()
    for item in items:  # integer answers must match exactly (see check_numeric); keep them float-exact
        if item["answer_type"] == "numeric" and float(item["expected"]).is_integer():
            assert abs(item["expected"]) < 2**53, item["id"]
    return items


def main() -> None:
    OUT.write_text("".join(json.dumps(i, ensure_ascii=False) + "\n" for i in build()), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
