# SmartRouter Benchmark Report

Run `20260929T131419Z` · generated 2026-09-29T13:52:33+00:00

> **PROVIDER FAILURES.** Generation calls failed after retries (Always Strong: 1, Always Cheap: 1, Static Router: 1). A failed query has no answer and counts as wrong, so a rate-limited or timed-out tier looks less accurate than it is. Check `provider_error` events before trusting the comparison.

> **Cost is estimated** for these older records: they did not store per-role billed calls, so SmartRouter's judge cost is a lower bound (parse retries and fallback-judge calls are not counted).

> Re-scored from the raw per-item results by `benchmark.reanalyze` with the current methodology (no new model calls).

## Executive Summary

Claim under test: *SmartRouter reduces cost without reducing answer quality.*

**Verdict: NOT SUPPORTED (Always Cheap dominates).** SmartRouter is cheaper than Always Strong without a measurable accuracy loss, but Always Cheap matches it for less money, so routing adds cost without a measurable benefit on this dataset (strict accuracy SmartRouter 96.7%, Always Cheap 93.3%, Always Strong 93.3%; cost incl. judge 6.57 vs 0.97 vs 9.67; paired vs cheap 2 wins / 1 losses (p=1.0)).

- Always Strong: strict accuracy 93.3% (95% CI 78.7%-98.2%, n=30); cost incl. judge 9.67; median latency 37268 ms.
- Always Cheap: strict accuracy 93.3% (95% CI 78.7%-98.2%, n=30); cost incl. judge 0.97; median latency 2179 ms.
- SmartRouter: strict accuracy 96.7% (95% CI 83.3%-99.4%, n=30); cost incl. judge 6.57; median latency 2934 ms.
- Static Router: strict accuracy 93.3% (95% CI 78.7%-98.2%, n=30); cost incl. judge 2.63; median latency 3045 ms.

Quality here means exact-check accuracy (numeric answers and unit tests). LLM-judge scores are secondary: the same judge family drives SmartRouter's escalations, so judging the systems with it would be circular.

## Dataset

30 fixed items from `benchmark/datasets/hard.jsonl`.

| Category | Items |
|---|---|
| coding | 9 |
| math | 12 |
| reasoning | 9 |

| Difficulty label | Items |
|---|---|
| hard | 30 |

## Methodology

1. Every item is sent through the production pipeline (`SmartRouterPipeline.run`) once per system; systems differ only in pipeline mode.
2. The in-loop judge (used for SmartRouter's escalation decisions) sees only the query and candidate answer, never the reference answer.
3. Final answers of all systems are then graded by the same judge model **with** the reference answer (reference-graded quality) — *skipped in this run (--no-reference-eval)*.
4. Numeric items are also checked deterministically against the expected value; code items are executed against unit tests (enabled). Deterministic checks are skipped in mock mode.
5. Quality threshold 0.78, max escalations 2, routing thresholds cheap < 0.35 <= mid < 0.7 <= strong, concurrency 1.

| Role | Model | Provider | Normalized cost |
|---|---|---|---|
| cheap | `openai/gpt-oss-20b` | groq | 1.0 |
| mid | `openai/gpt-oss-120b` | groq | 3.0 |
| strong | `nvidia/nemotron-3-ultra-550b-a55b` | nvidia | 10.0 |
| judge | `gemini-3.5-flash-lite` | gemini | 3.0 |
| judge_fallback | `z-ai/glm-5.3-flash` | nvidia | 3.0 |
| second_opinion | `qwen/qwen3.8-27b` | groq | 3.0 |

## Baselines

| System | Mode | Description |
|---|---|---|
| Always Strong | `always_strong` | Every query goes to the strong tier (quality ceiling / cost reference). |
| Always Cheap | `always_cheap` | Every query goes to the cheap tier; no judge needed in production. |
| SmartRouter | `smart` | Heuristic routing + answer checks + judge-driven escalation. |
| Static Router | `static` | Heuristic difficulty routing; no judge-driven escalation. |

## Metrics

- **Strict acc.** (primary): correct / all deterministically checkable items (numeric answers, unit tests). An answer with no readable final number counts as wrong (*Unparseable* column). 95% Wilson interval.
- **Ref. judge q.** / **In-loop judge q.**: LLM-judge scores with / without the reference answer (secondary; same judge family as the escalation loop).
- **Calls/query**: generation calls plus serving-time judge calls (what production would pay for), excluding transport retries.
- **Gen cost**: mean normalized cost per query (sum of per-call tier weights; experiment weights, not money).
- **Cost incl. judge**: gen cost plus serving-time judge calls x judge weight (only SmartRouter needs the judge when serving).
- **Latency**: end-to-end pipeline latency including the in-loop judge call for every system; `gen p50` is generation-only.
- **Below thr.**: share of queries whose final answer did not meet the quality threshold (best effort or failed).
- `quality_score` is an LLM evaluation signal, **not** a probability of correctness.

## Results

| System | Strict acc. | 95% CI | Unparseable | Gen cost | Cost incl. judge | vs cheap | vs strong (incl. judge) | Calls/query | p50 ms | p95 ms | Escalation | Failed | Ref. judge q. | In-loop judge q. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Always Strong | 93.3% | 78.7%-98.2% | 1 | 9.67 | 9.67 | 10.00x | 0.0% lower | 0.97 | 37268 | 122402 | 0.0% | 3.3% | n/a | 0.972 |
| Always Cheap | 93.3% | 78.7%-98.2% | 1 | 0.97 | 0.97 | 1.00x | 90.0% lower | 0.97 | 2179 | 4490 | 0.0% | 3.3% | n/a | 0.974 |
| SmartRouter | 96.7% | 83.3%-99.4% | 0 | 3.07 | 6.57 | 6.79x | 32.1% lower | 2.20 | 2934 | 10613 | 3.3% | 0.0% | n/a | 0.928 |
| Static Router | 93.3% | 78.7%-98.2% | 0 | 2.63 | 2.63 | 2.72x | 72.8% lower | 0.97 | 3045 | 10447 | 0.0% | 3.3% | n/a | 0.943 |

### Paired comparison (same items, strict exact checks)

*Wins*: the first system was right and the second wrong. The sign test asks whether wins and losses differ by more than chance; p >= 0.05 means no detectable difference on this many items.

| Comparison | Items | Wins | Losses | Both right | Both wrong | Sign-test p |
|---|---|---|---|---|---|---|
| SmartRouter vs Always Cheap | 30 | 2 | 1 | 27 | 0 | 1.000 |
| SmartRouter vs Always Strong | 30 | 2 | 1 | 27 | 0 | 1.000 |
| Static Router vs Always Cheap | 30 | 2 | 2 | 26 | 0 | 1.000 |
| Always Strong vs Always Cheap | 30 | 1 | 1 | 27 | 1 | 1.000 |

### Router validation (no model calls)

Heuristic category matches the dataset label on 66.7% of 30 items. Initial tier by the dataset's difficulty label (a useful difficulty estimate sends *hard* items up more often):

| Label | Cheap | Mid | Strong |
|---|---|---|---|
| hard | 4 | 26 | 0 |

### Model utilization (share of generation calls)

| System | Cheap | Mid | Strong |
|---|---|---|---|
| Always Strong | 0.0% | 0.0% | 100.0% |
| Always Cheap | 100.0% | 0.0% | 0.0% |
| SmartRouter | 12.9% | 83.9% | 3.2% |
| Static Router | 13.3% | 86.7% | 0.0% |

### Accuracy by answering tier (exact checks, all systems pooled)

| Tier | Correct | Checked | Accuracy |
|---|---|---|---|
| Cheap | 36 | 37 | 97.3% |
| Mid | 48 | 51 | 94.1% |
| Strong | 29 | 30 | 96.7% |

> **Tier order check failed:** a higher tier was less accurate than a lower one on exact checks. Routing harder queries upward may be making answers worse; review the model lineup.

### Judge usage

Which judge graded each attempt. More than one judge means scores mix evaluators.

| System | Graded by | Judge failures |
|---|---|---|
| Always Strong | `gemini-3.5-flash-lite` 29 | 0 |
| Always Cheap | `gemini-3.5-flash-lite` 29 | 0 |
| SmartRouter | `gemini-3.5-flash-lite` 31 | 0 |
| Static Router | `gemini-3.5-flash-lite` 29 | 0 |

### Totals

| System | Total gen cost | Strong calls | Judge calls | Judge failures | Escalations (necessary / unnecessary / unknown) |
|---|---|---|---|---|---|
| Always Strong | 290.0 | 30 | 29 | 0 | 0 (0 / 0 / 0) |
| Always Cheap | 29.0 | 0 | 29 | 0 | 0 (0 / 0 / 0) |
| SmartRouter | 92.0 | 1 | 35 | 0 | 1 (0 / 1 / 0) |
| Static Router | 79.0 | 0 | 29 | 0 | 0 (0 / 0 / 0) |

An escalation is *unnecessary* when the answer it replaced already passed the exact check.

### Judge accuracy vs exact checks (every judged attempt)

| System | Scored | Checked | Agree | False accepts | False rejects |
|---|---|---|---|---|---|
| Always Strong | primary judge | 28 | 28/28 | 0 | 0 |
| Always Cheap | primary judge | 28 | 27/28 | 0 | 1 |
| SmartRouter | primary judge | 31 | 26/31 | 1 | 4 |
| SmartRouter | final decision | 31 | 28/31 | 1 | 2 |
| Static Router | primary judge | 29 | 26/29 | 1 | 2 |

*Primary judge* is the first judge's verdict alone; *final decision* adds answer checks and second opinions.

Second opinions (SmartRouter): 4 requested, 4 succeeded, 0 failed (success rate 100.0%); latency median 946 ms, max 1448 ms, including failures.

### Per-item trace (SmartRouter)

| Item | Initial model | Judge result | Judge conf. | Second opinion | Exact (initial) | Escalated | Escalation reason | Final model | Final correct | Cost | Judge calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| math_hard_01 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_02 | `openai/gpt-oss-120b` | incorrect 0.25 | 1.00 | good 1.00 (0.4s) | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 2 |
| math_hard_03 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_04 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_05 | `openai/gpt-oss-120b` | incorrect 0.25 | 1.00 | incorrect 0.25 (1.4s) | yes | yes | second judge also rejected (qwen/qwen3.8-27b: incorrect, score 0.25, confidence 1.00) afte | `nvidia/nemotron-3-ultra-550b-a55b` | yes | 13 | 4 |
| math_hard_06 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_07 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_08 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_09 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_10 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_11 | `openai/gpt-oss-20b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-20b` | yes | 1 | 1 |
| math_hard_12 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| reasoning_hard_01 | `openai/gpt-oss-20b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-20b` | yes | 1 | 1 |
| reasoning_hard_02 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| reasoning_hard_03 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| reasoning_hard_04 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| reasoning_hard_05 | `openai/gpt-oss-20b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-20b` | yes | 1 | 1 |
| reasoning_hard_06 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| reasoning_hard_07 | `openai/gpt-oss-120b` | incorrect 0.10 | 1.00 | good 1.00 (0.4s) | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 2 |
| reasoning_hard_08 | `openai/gpt-oss-20b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-20b` | yes | 1 | 1 |
| reasoning_hard_09 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_01 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_02 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_03 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | no | no |  | `openai/gpt-oss-120b` | no | 3 | 1 |
| coding_hard_04 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_05 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_06 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_07 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_08 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_09 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |


## Per-category Results

Reference-graded quality (mean) / normalized cost (mean):

| System | general_qa | reasoning | coding | math | summarization |
|---|---|---|---|---|---|
| Always Strong | n/a | n/a / 10.00 | n/a / 10.00 | n/a / 9.17 | n/a |
| Always Cheap | n/a | n/a / 1.00 | n/a / 1.00 | n/a / 0.92 | n/a |
| SmartRouter | n/a | n/a / 2.33 | n/a / 3.00 | n/a / 3.67 | n/a |
| Static Router | n/a | n/a / 2.00 | n/a / 3.00 | n/a / 2.83 | n/a |

## Failure Analysis

Counts of non-accepted attempts by cause (one attempt can have several judge-reported reasons):

| System | Incorrect | Incomplete | Reasoning weakness | Provider error | Below threshold (no reason given) |
|---|---|---|---|---|---|
| Always Strong | 0 | 1 | 0 | 1 | 0 |
| Always Cheap | 1 | 0 | 0 | 1 | 1 |
| SmartRouter | 2 | 0 | 0 | 0 | 0 |
| Static Router | 2 | 0 | 1 | 1 | 1 |

SmartRouter routing flow (edge counts):

| From | To | Count |
|---|---|---|
| cheap | accepted | 4 |
| mid | accepted | 25 |
| mid | strong | 1 |
| strong | best_effort | 1 |

## Tradeoffs

- Escalation versus static routing: SmartRouter mean gen cost 3.07 vs Static Router 2.63; median latency 2934 ms vs 3045 ms; reference-graded quality n/a vs n/a.
- SmartRouter needed 35 serving-time judge calls for 30 queries; baselines need none in production.
- Mean escalations per query: 0.03.

## Limitations

- The judge is an LLM: it has evaluator, verbosity and self-preference biases, can share blind spots with the models it grades, and scores inconsistently across calls. quality_score is not ground truth.
- Reference answers are short and hand-written; open-ended items (reasoning, summarization) admit many valid answers.
- 30 items is small: the confidence intervals above are wide, and a sign test on a handful of discordant items cannot detect small differences.
- Normalized cost uses configured tier weights, not provider billing. It measures relative model usage, not money.
- Latency depends on provider load, rate limits and retries at run time and is not reproducible across runs.
- Provider model versions change over time; results apply only to the model IDs listed above on the run date.
- The query analyzer is a keyword heuristic; see *Router validation* for how its estimates line up with the dataset labels.
- Re-running the same items gives repeated measurements of the same questions, not new independent samples.
