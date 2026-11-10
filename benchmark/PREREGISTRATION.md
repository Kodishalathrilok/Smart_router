# Pre-registration: hard-set re-run after the measurement fixes

Written and committed on 2026-11-10, **before** any of the runs it describes. The analysis code is the code at the
commit that adds this file. Anything done differently afterwards is listed under "Deviations" at the bottom, with the
reason, and is never applied silently.

## Why this run exists

On the three pooled hard-set runs (20260929T050756Z, 20260929T111739Z, 20260929T131419Z), 11 of the 15 misses were
plumbing rather than wrong answers: 4 empty replies with `finish_reason=length`, 3 answers cut off by the output limit,
3 final numbers the checker could not read, and 1 provider error. Always Mid was not run. Costs were in normalized units
only. This run repeats the experiment with truncated empty replies retried, all five systems, and dollar costs.

## Hypothesis

*SmartRouter reduces cost without reducing answer quality.*

## Design

| | |
|---|---|
| Dataset | `benchmark/datasets/hard.jsonl`, all 30 items (12 math, 9 reasoning, 9 coding), every item exactly checkable |
| Systems | All five: Always Cheap (`always_weak`), Always Mid, Always Strong, Static Router, SmartRouter |
| Runs | 3 complete runs, made one after another |
| Command | `python -m benchmark.run --dataset hard --exec-code --no-reference-eval --delay 2` (concurrency 1) |
| Lineup | Same as the pooled runs: cheap `openai/gpt-oss-20b`, mid `openai/gpt-oss-120b` (both Groq), strong `nvidia/nemotron-3-ultra-550b-a55b`, judge `gemini-3.5-flash-lite`, fallback judge `z-ai/glm-5.3-flash`, second opinion `qwen/qwen3.8-27b` |
| Settings | As in the pooled runs: `QUALITY_THRESHOLD=0.78`, `MAX_ESCALATIONS=2`, routing cut-offs 0.35 / 0.70, `GENERATION_MAX_TOKENS=4096`, `JUDGE_MAX_TOKENS=1024`, `REQUEST_TIMEOUT_S=180`. New: `TRUNCATION_RETRY_MAX_TOKENS=8192` |

`TRUNCATION_RETRY_MAX_TOKENS` must be above `GENERATION_MAX_TOKENS` or the retry never fires; the pooled runs already
used 4096 for generation, so the default retry limit of 4096 would change nothing. The value used is recorded here
before the first run; if the repository owner chooses a different one, this file is updated in its own commit first.

Item-runs: 5 systems x 30 items x 3 runs = 450.

## Which runs are pooled

The first three runs started after this commit that complete (all five per-system result files present, 30 items
each). All three are pooled, whatever they show. A run that is interrupted (crash, quota exhausted, machine asleep)
is discarded whole and repeated; its run ID and the reason are listed under Deviations. No run is dropped or repeated
because of its results. The earlier runs are not pooled with the new ones (they have four systems and predate the
truncation retry); they stay in the repository for comparison.

## Metrics

- **Primary: strict exact accuracy** = correct / every checkable item. An answer that is missing, cut off or
  unreadable counts as wrong.
- Secondary, reported beside it and never in place of it: accuracy among readable answers, and the plumbing failure
  rate (empty reply, truncation, unreadable final number, provider error), per system.
- Cost: normalized units including SmartRouter's serving-time judge calls (as before), and US dollars per 1,000 queries
  from stored tokens at the list prices in `config/models.json`. The two are never mixed. A system that used a model
  with no public price (the NVIDIA-hosted strong model and fallback judge) has dollar cost "unknown".
- LLM-judge scores are not used for any conclusion.

## Analysis: by question, not by repeat

Three runs of 30 questions are 30 questions measured three times, not 90 independent samples. So:

- Each system gets **one result per question**: correct if it was strictly correct in at least 2 of the 3 runs.
- Systems are compared with a two-sided exact sign test on the **30 questions** (`Paired by question` in
  `benchmark/results/reanalysis.md`), alpha = 0.05.
- Item-run counts (x/90), their Wilson intervals and the item-run paired table are descriptive only and are labelled so.
- No other tests, subsets or per-category breakdowns are used for the verdict. Anything else looked at afterwards is
  labelled exploratory.

## What counts as supported / not supported

**Supported** only if all three hold on the pooled new runs:

1. SmartRouter beats Always Cheap in the by-question sign test (p < 0.05, more wins than losses).
2. SmartRouter is not significantly worse than Always Strong in the by-question sign test.
3. SmartRouter costs less than Always Strong in normalized cost including its judge calls. (The dollar comparison with
   Always Strong cannot be made: the strong model has no public price.)

**Not supported** if any of the three fails. In particular, if (1) fails and Always Cheap costs less than SmartRouter,
the result is "not supported: Always Cheap dominates". With about 94% cheap-model accuracy in the earlier runs there
are few questions on which the systems can differ, so a null result is the expected outcome, and it will be reported as
"not supported", not as "inconclusive, needs more data".

Also reported, each as a plain yes/no:

- **Did plumbing change the picture?** Compare each system's strict accuracy with its accuracy among readable answers,
  and the earlier pooled plumbing rate (11/360 item-runs) with the new one.
- **Does "the judge costs more than the tier gap" hold in dollars?** SmartRouter's judge-side dollars per query (judge,
  fallback judge and second-opinion tokens) versus the dollar difference per query between Always Mid and Always Cheap.
  It holds if the first is larger. If SmartRouter's judge-side cost is "unknown" (the unpriced fallback judge was
  used), the answer is "cannot be determined", not an estimate.

## Validity condition

The measurement counts as valid if plumbing failures are at most 1 of the 450 item-runs. If there are more, the numbers
are still reported in full, and the verdict stands only if it is the same whether the plumbing failures are counted as
wrong (strict) or left out (readable only); otherwise the outcome is reported as "measurement not valid", with the
failures listed by kind. The truncation retry addresses empty replies only: answers cut off mid-text, unreadable final
numbers and provider errors can still occur, so this condition may fail.

## Deviations

None yet.
