# SmartRouter Benchmark Report

Run `20261111T021401Z` · generated 2026-11-11T02:49:02+00:00

> **PROVIDER FAILURES.** Generation calls failed after retries (Always Cheap: 1). A failed query has no answer and counts as wrong, so a rate-limited or timed-out tier looks less accurate than it is. Check `provider_error` events before trusting the comparison.

> **Cost is estimated** for these older records: they did not store per-role billed calls, so SmartRouter's judge cost is a lower bound (parse retries and fallback-judge calls are not counted).

> Re-scored from the raw per-item results by `benchmark.reanalyze` with the current methodology (no new model calls).

## Executive Summary

Claim under test: *SmartRouter reduces cost without reducing answer quality.*

**Verdict: NOT SUPPORTED (Always Cheap dominates).** SmartRouter is cheaper than Always Strong without a measurable accuracy loss, but Always Cheap matches it for less money, so routing adds cost without a measurable benefit on this dataset (strict accuracy SmartRouter 96.7%, Always Cheap 90.0%, Always Strong 96.7%; cost incl. judge 5.93 vs 0.97 vs 10.00; paired vs cheap 3 wins / 1 losses (p=0.625)).

- Always Mid: strict accuracy 93.3% (95% CI 78.7%-98.2%, n=30); cost incl. judge 3.00; median latency 9575 ms.
- Always Strong: strict accuracy 96.7% (95% CI 83.3%-99.4%, n=30); cost incl. judge 10.00; median latency 16555 ms.
- Always Cheap: strict accuracy 90.0% (95% CI 74.4%-96.5%, n=30); cost incl. judge 0.97; median latency 9058 ms.
- SmartRouter: strict accuracy 96.7% (95% CI 83.3%-99.4%, n=30); cost incl. judge 5.93; median latency 8440 ms.
- Static Router: strict accuracy 96.7% (95% CI 83.3%-99.4%, n=30); cost incl. judge 2.73; median latency 7808 ms.

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

| Role | Model | Provider | Normalized cost | USD per 1M tokens (in / out) |
|---|---|---|---|---|
| cheap | `openai/gpt-oss-20b` | groq | 1.0 | $0.075 / $0.3 |
| mid | `openai/gpt-oss-120b` | groq | 3.0 | $0.15 / $0.6 |
| strong | `nvidia/nemotron-3-ultra-550b-a55b` | nvidia | 10.0 | no public price |
| judge | `gemini-3.5-flash-lite` | gemini | 3.0 | $0.3 / $2.5 |
| judge_fallback | `z-ai/glm-5.3-flash` | nvidia | 3.0 | no public price |
| second_opinion | `qwen/qwen3.8-27b` | groq | 3.0 | $0.8 / $4.0 |

## Baselines

| System | Mode | Description |
|---|---|---|
| Always Mid | `always_mid` | Every query goes to the mid tier. |
| Always Strong | `always_strong` | Every query goes to the strong tier (quality ceiling / cost reference). |
| Always Cheap | `always_cheap` | Every query goes to the cheap tier; no judge needed in production. |
| SmartRouter | `smart` | Heuristic routing + answer checks + judge-driven escalation. |
| Static Router | `static` | Heuristic difficulty routing; no judge-driven escalation. |

## Metrics

- **Strict acc.** (primary): correct / all deterministically checkable items (numeric answers, unit tests). An answer with no readable final number counts as wrong. 95% Wilson interval.
- **Plumbing failures**: checkable items missed for a reason that says nothing about the model's ability: an empty reply, an answer cut off by the output limit, a final number that cannot be read, or a provider error. Judged on the final answer.
- **Acc. among readable** (secondary): correct / checkable items minus plumbing failures. Shows capability once plumbing is set aside; it is not the headline, because a system that fails to deliver an answer has still failed the user.
- **Ref. judge q.** / **In-loop judge q.**: LLM-judge scores with / without the reference answer (secondary; same judge family as the escalation loop).
- **Calls/query**: generation calls plus serving-time judge calls (what production would pay for), excluding transport retries.
- **Gen cost**: mean normalized cost per query (sum of per-call tier weights; experiment weights, not money).
- **Cost incl. judge**: gen cost plus serving-time judge calls x judge weight (only SmartRouter needs the judge when serving).
- **$ / 1k queries**: US dollars per 1,000 queries at list prices (`config/models.json`), from the tokens stored for every call, billed failures included, with the same serving convention as *Cost incl. judge*. `unknown` when any model that ran has no public price or the records predate token storage; never a partial sum. Kept separate from normalized cost: *vs cheap* / *vs strong* use normalized units.
- **Latency**: end-to-end pipeline latency including the in-loop judge call for every system; `gen p50` is generation-only.
- **Below thr.**: share of queries whose final answer did not meet the quality threshold (best effort or failed).
- `quality_score` is an LLM evaluation signal, **not** a probability of correctness.

## Results

| System | Strict acc. | 95% CI | Acc. among readable | Plumbing failures | Gen cost | Cost incl. judge | $ / 1k queries | vs cheap | vs strong (incl. judge) | Calls/query | p50 ms | p95 ms | Escalation | Failed | Ref. judge q. | In-loop judge q. |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Always Mid | 93.3% | 78.7%-98.2% | 93.3% | 0.0% (0/30) | 3.00 | 3.00 | $0.34 | 3.10x | 70.0% lower | 1.00 | 9575 | 14283 | 0.0% | 0.0% | n/a | 0.975 |
| Always Strong | 96.7% | 83.3%-99.4% | 100.0% | 3.3% (1/30) | 10.00 | 10.00 | unknown | 10.34x | 0.0% lower | 1.00 | 16555 | 84943 | 0.0% | 0.0% | n/a | 0.988 |
| Always Cheap | 90.0% | 74.4%-96.5% | 96.4% | 6.7% (2/30) | 0.97 | 0.97 | $0.20 | 1.00x | 90.3% lower | 0.97 | 9058 | 14691 | 0.0% | 3.3% | n/a | 0.974 |
| SmartRouter | 96.7% | 83.3%-99.4% | 100.0% | 3.3% (1/30) | 2.73 | 5.93 | $1.15 | 6.14x | 40.7% lower | 2.07 | 8440 | 15845 | 0.0% | 0.0% | n/a | 0.950 |
| Static Router | 96.7% | 83.3%-99.4% | 100.0% | 3.3% (1/30) | 2.73 | 2.73 | $0.35 | 2.83x | 72.7% lower | 1.00 | 7808 | 11257 | 0.0% | 0.0% | n/a | 0.945 |

### Paired comparison (same items, strict exact checks)

*Wins*: the first system was right and the second wrong. The sign test asks whether wins and losses differ by more than chance; p >= 0.05 means no detectable difference on this many items.

| Comparison | Items | Wins | Losses | Both right | Both wrong | Sign-test p |
|---|---|---|---|---|---|---|
| SmartRouter vs Always Cheap | 30 | 3 | 1 | 26 | 0 | 0.625 |
| SmartRouter vs Always Mid | 30 | 2 | 1 | 27 | 0 | 1.000 |
| SmartRouter vs Always Strong | 30 | 1 | 1 | 28 | 0 | 1.000 |
| Static Router vs Always Cheap | 30 | 3 | 1 | 26 | 0 | 0.625 |
| Always Strong vs Always Cheap | 30 | 3 | 1 | 26 | 0 | 0.625 |

### Router validation (no model calls)

Heuristic category matches the dataset label on 66.7% of 30 items. Initial tier by the dataset's difficulty label (a useful difficulty estimate sends *hard* items up more often):

| Label | Cheap | Mid | Strong |
|---|---|---|---|
| hard | 4 | 26 | 0 |

### Model utilization (share of generation calls)

| System | Cheap | Mid | Strong |
|---|---|---|---|
| Always Mid | 0.0% | 100.0% | 0.0% |
| Always Strong | 0.0% | 0.0% | 100.0% |
| Always Cheap | 100.0% | 0.0% | 0.0% |
| SmartRouter | 13.3% | 86.7% | 0.0% |
| Static Router | 13.3% | 86.7% | 0.0% |

### Accuracy by answering tier (exact checks, all systems pooled)

| Tier | Correct | Checked | Accuracy |
|---|---|---|---|
| Cheap | 35 | 37 | 94.6% |
| Mid | 78 | 81 | 96.3% |
| Strong | 29 | 30 | 96.7% |

### Judge usage

Which judge graded each attempt. More than one judge means scores mix evaluators.

| System | Graded by | Judge failures |
|---|---|---|
| Always Mid | `gemini-3.5-flash-lite` 30 | 0 |
| Always Strong | `gemini-3.5-flash-lite` 30 | 0 |
| Always Cheap | `gemini-3.5-flash-lite` 29 | 0 |
| SmartRouter | `gemini-3.5-flash-lite` 30 | 0 |
| Static Router | `gemini-3.5-flash-lite` 30 | 0 |

### Totals

| System | Total gen cost | Strong calls | Judge calls | Judge failures | Escalations (necessary / unnecessary / unknown) |
|---|---|---|---|---|---|
| Always Mid | 90.0 | 0 | 30 | 0 | 0 (0 / 0 / 0) |
| Always Strong | 300.0 | 30 | 30 | 0 | 0 (0 / 0 / 0) |
| Always Cheap | 29.0 | 0 | 29 | 0 | 0 (0 / 0 / 0) |
| SmartRouter | 82.0 | 0 | 32 | 0 | 0 (0 / 0 / 0) |
| Static Router | 82.0 | 0 | 30 | 0 | 0 (0 / 0 / 0) |

An escalation is *unnecessary* when the answer it replaced already passed the exact check.

### Judge accuracy vs exact checks (every judged attempt)

| System | Scored | Checked | Agree | False accepts | False rejects |
|---|---|---|---|---|---|
| Always Mid | primary judge | 30 | 27/30 | 2 | 1 |
| Always Strong | primary judge | 30 | 30/30 | 0 | 0 |
| Always Cheap | primary judge | 28 | 26/28 | 1 | 1 |
| SmartRouter | primary judge | 30 | 27/30 | 1 | 2 |
| SmartRouter | final decision | 30 | 29/30 | 1 | 0 |
| Static Router | primary judge | 29 | 27/29 | 0 | 2 |

*Primary judge* is the first judge's verdict alone; *final decision* adds answer checks and second opinions.

Second opinions (SmartRouter): 2 requested, 2 succeeded, 0 failed (success rate 100.0%); latency median 381 ms, max 381 ms, including failures.

### Per-item trace (SmartRouter)

| Item | Initial model | Judge result | Judge conf. | Second opinion | Exact (initial) | Escalated | Escalation reason | Final model | Final correct | Cost | Judge calls |
|---|---|---|---|---|---|---|---|---|---|---|---|
| math_hard_01 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_02 | `openai/gpt-oss-120b` | incorrect 0.25 | 1.00 | good 1.00 (0.4s) | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 2 |
| math_hard_03 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_04 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| math_hard_05 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
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
| reasoning_hard_07 | `openai/gpt-oss-120b` | incorrect 0.25 | 1.00 | good 1.00 (0.4s) | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 2 |
| reasoning_hard_08 | `openai/gpt-oss-20b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-20b` | yes | 1 | 1 |
| reasoning_hard_09 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_01 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_02 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_03 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_04 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_05 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_06 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_07 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | no | no |  | `openai/gpt-oss-120b` | no | 3 | 1 |
| coding_hard_08 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |
| coding_hard_09 | `openai/gpt-oss-120b` | good 1.00 | 1.00 |  | yes | no |  | `openai/gpt-oss-120b` | yes | 3 | 1 |


## Per-category Results

Reference-graded quality (mean) / normalized cost (mean):

| System | general_qa | reasoning | coding | math | summarization |
|---|---|---|---|---|---|
| Always Mid | n/a | n/a / 3.00 | n/a / 3.00 | n/a / 3.00 | n/a |
| Always Strong | n/a | n/a / 10.00 | n/a / 10.00 | n/a / 10.00 | n/a |
| Always Cheap | n/a | n/a / 1.00 | n/a / 1.00 | n/a / 0.92 | n/a |
| SmartRouter | n/a | n/a / 2.33 | n/a / 3.00 | n/a / 2.83 | n/a |
| Static Router | n/a | n/a / 2.33 | n/a / 3.00 | n/a / 2.83 | n/a |

## Failure Analysis

Counts of non-accepted attempts by cause (one attempt can have several judge-reported reasons):

| System | Incorrect | Incomplete | Reasoning weakness | Provider error |
|---|---|---|---|---|
| Always Mid | 1 | 0 | 1 | 0 |
| Always Strong | 0 | 1 | 0 | 0 |
| Always Cheap | 1 | 0 | 0 | 1 |
| SmartRouter | 0 | 0 | 0 | 0 |
| Static Router | 2 | 0 | 1 | 0 |

SmartRouter routing flow (edge counts):

| From | To | Count |
|---|---|---|
| cheap | accepted | 4 |
| mid | accepted | 26 |

## Tradeoffs

- Escalation versus static routing: SmartRouter mean gen cost 2.73 vs Static Router 2.73; median latency 8440 ms vs 7808 ms; reference-graded quality n/a vs n/a.
- SmartRouter needed 32 serving-time judge calls for 30 queries; baselines need none in production.
- Mean escalations per query: 0.00.

## Limitations

- The judge is an LLM: it has evaluator, verbosity and self-preference biases, can share blind spots with the models it grades, and scores inconsistently across calls. quality_score is not ground truth.
- Reference answers are short and hand-written; open-ended items (reasoning, summarization) admit many valid answers.
- 30 items is small: the confidence intervals above are wide, and a sign test on a handful of discordant items cannot detect small differences.
- Normalized cost uses configured tier weights, not provider billing. It measures relative model usage, not money. Dollar figures use list prices on the date recorded in `config/models.json`, not an invoice.
- Latency depends on provider load, rate limits and retries at run time and is not reproducible across runs.
- Provider model versions change over time; results apply only to the model IDs listed above on the run date.
- The query analyzer is a keyword heuristic; see *Router validation* for how its estimates line up with the dataset labels.
- Re-running the same items gives repeated measurements of the same questions, not new independent samples.
