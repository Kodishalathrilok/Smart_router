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
used 4096 for generation, so the default retry limit of 4096 would change nothing. Confirmed 2026-11-10, before any
run: one test call to each Groq model (`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `qwen/qwen3.8-27b`) with a limit
of 8192 was accepted, so 8192 is used (the agreed fallback, had Groq rejected it as too large, was 6000).

Item-runs: 5 systems x 30 items x 3 runs = 450.

## Which runs are pooled

The first three runs started after this commit that complete (all five per-system result files present, 30 items
each). All three are pooled, whatever they show. No run is dropped or repeated because of its results. The earlier
runs are not pooled with the new ones (they have four systems and predate the truncation retry); they stay in the
repository for comparison.

Amended 2026-11-10, before any run (the first version discarded an interrupted run whole): runs are resumable. If a
provider is still rate limiting after every retry (in practice an exhausted daily quota), the run stops; the item
that was being answered is not kept, because a quota error says nothing about the model, and
`python -m benchmark.run ... --resume RUN_ID` later continues the same run from the next unsaved item. Every item
that finished is kept exactly as it was first recorded and is never re-run. Each stop is listed under Deviations with
the run ID, the system and the item count reached. Provider errors other than rate limits (for example a 503) are not
retried this way: they stay in the results and count as wrong.

Amended 2026-11-11, after the discarded run described under Deviations and before any run that is pooled: a
**connection failure** that survives every retry (`NetworkError`: the request never reached the provider) is treated
exactly like a rate limit. The run stops, the item is not kept, and the run is resumed later. A request that reached
the provider and failed there (a 5xx, a timeout, an empty or cut-off reply) is still recorded and still counts as
wrong.

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

## Target

Amended 2026-11-10, before any run (the first version of this section required at most 1 plumbing failure in 450
item-runs; the repository owner replaced it because the truncation retry can only fix empty replies).

**Zero empty-reply misses**: no item-run in the pooled new runs may be missed because the model returned an empty reply,
whether or not it was cut off by the output limit (an empty reply with `finish_reason=length` that is still empty after
the retry counts). The earlier pooled runs had 4.

The other plumbing failures (an answer cut off mid-text, an answer without a readable final number, a provider error)
are **reported, not targeted**: they are counted per system and by kind, and they count as wrong in the primary metric.
The answer prompt, the number reader and `GENERATION_MAX_TOKENS` are deliberately left unchanged, so these can recur.

If the target is missed, the numbers are still reported in full and the miss is stated next to the verdict.

## Deviations

1. **Run `20261110T164607Z` discarded whole (decided 2026-11-11 by the repository owner, before any other run
   existed).** The local network connection dropped about 8 minutes into the run (around 16:54 UTC). Always Cheap had
   finished (28/30 correct); from the middle of Always Mid onward every call failed with `ConnectError`, so 136 of
   the 150 item-runs contain a connection failure and say nothing about any model. The run still ran to the end
   because only rate limits stopped a run at that point. No result from it was used for any decision other than this
   one, and its files are not committed. The connection-failure rule above was added in response. The three pooled
   runs are the first three complete runs started after this entry was committed.
2. **Run `20261111T024904Z` (the third pooled run) stopped by hand and resumed (2026-11-11).** Runs
   `20261111T013331Z` and `20261111T021401Z` completed without any stop. In the third run, Always Cheap, Always Mid
   and Always Strong completed; during Static Router the Gemini judge returned HTTP 429 through all four tries on two
   items (`math_hard_06`, `math_hard_08`). This was the per-minute limit, not the daily quota: a single call a few
   minutes later succeeded. The run did not stop by itself, because the fallback judge took over and the stop rule only
   looked at the error left on the attempt; each such item then waited about 12 minutes on fallback-judge timeouts. The
   process was killed at Static Router 7/30. What was done, following the rule already written above ("if a provider is
   still rate limiting after every retry, the run stops; the item is not kept"):
   - the code now applies that rule also when a fallback model answers after the rate-limited one gave up;
   - `math_hard_06`, which had been saved with the judge in that state, was removed from the partial file and is
     answered again on resume (`math_hard_08` was in progress and had not been saved). The other six saved Static
     Router items are kept as first recorded;
   - the rest of the run uses `--delay 5` instead of `--delay 2`, to stay under the judge's per-minute limit, and is
     resumed automatically after a 90-second wait when it stops. This changes pacing only; no prompt, model, limit or
     scoring rule changed.
   No accuracy figure had been computed or looked at for any new run when this was decided.
