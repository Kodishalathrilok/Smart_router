# SmartRouter production-readiness audit (2026-10-07, finalized 2026-10-08)

Every finding below was reproduced with a probe or test before it was fixed, and each fix has a regression test
(mostly `tests/integration/test_audit_regressions.py`, `tests/unit/test_benchmark_stats.py`).

| ID | Sev | Problem (evidence) | Status |
|---|---|---|---|
| A1 | P1 | Judge score `8` silently read as `0.8` (0-10 "scale guessing"); also `"0.9"` strings accepted | Fixed: finite JSON numbers in [0,1] only, else parse error -> retry -> fallback judge |
| A2 | P1 | One query could make 66 provider calls (3 attempts x retries x parse retries x fallback judge); probe reproduced 42 | Fixed: per-query call budget (`MAX_PROVIDER_CALLS_PER_QUERY`, default 16), `provider_calls` recorded |
| A3 | P1 | One truncated JSONL line made /api/stats, /routes, /failures return 500 permanently; every call re-parsed whole files | Fixed: SQLite store (stdlib), tolerant reads, indexed lookups, stats window |
| A4 | P1 | Benchmark headline "68% cheaper than always-strong" while Always Cheap was as accurate at 1/7 the cost | Fixed: strict exact accuracy, Wilson CIs, paired sign tests, claim verdict vs Always Cheap; Always Mid added |
| A5 | P1 | "What is 1+1+...+1?" (3000 terms) -> RecursionError -> HTTP 500 | Fixed: expression size cap + verifier guard |
| A6 | P2 | Model-written code could forge/break the execution report -> verifier crash -> 500 | Fixed: report parsed as untrusted |
| A7 | P2 | Any verifier exception discarded an answer already paid for | Fixed: `checks_error` event, answer kept |
| A8 | P2 | Integer answers passed within +-0.01% (76120 accepted for 76127) | Fixed: exact match for integers |
| A9 | P2 | Unparseable answers dropped from the accuracy denominator (systems scored on different items) | Fixed: counted wrong, reported separately |
| A10 | P2 | Best effort could return an answer that failed its own assertion over a clean lower-scored one | Fixed |
| A11 | P2 | 50 MB body fully read and parsed before the length check | Fixed: streaming 413 (Content-Length and chunked) |
| A12 | P2 | DNS rebinding could reach the unauthenticated localhost API | Fixed: Host allow-list (`ALLOWED_HOSTS`) |
| A13 | P2 | Disk/log failure turned a finished answer into HTTP 500 | Fixed: best-effort logging, `/health` reports `degraded` |
| A14 | P2 | Deadline (504) cancellations left no trace of paid calls | Fixed: `pipeline_failed` with `Cancelled` |
| A15 | P2 | Code execution: no memory/file/process limits; children could outlive timeout (latter not reproduced) | Mitigated: rlimits + process-group kill on Linux. Still NOT a sandbox (accepted, documented) |
| A16 | P2 | Analyzer: 76% category accuracy, routes no item to strong, 18/66 "hard" items to cheap | Documented + router validation in every report; not rewritten (no data showing a better router pays off) |
| A17 | P2 | CI would fail: unpinned ruff 0.16 changed default rules (143 errors) | Fixed: `ruff.toml`, pinned ruff, CI matrix 3.10/3.13 + mock benchmark smoke test |
| A18 | P3 | `except TimeoutError` misses `asyncio.TimeoutError` on Python 3.10 (4 failing tests) | Fixed |
| A19 | P3 | `< /candidate>` (spaced) bypassed delimiter neutralization | Fixed |
| A20 | P3 | Dashboard headline compared only against all-strong; misleading escalation reason text | Fixed |
| A21 | P2 | Benchmark undercounted SmartRouter's judge cost: `judge_calls` counted 1 per attempt even when a parse retry or fallback judge made more billed calls | Fixed: per-role `billed_calls` on every record; older runs labeled "estimated lower bound" |
| A22 | P3 | Corrupt log database crashed startup with a raw traceback | Fixed: clear configuration error |
| A23 | P2 | Benchmark could silently score rate-limited/timed-out tiers as "wrong" | Fixed: warning + report banner on any generation failure |

## Benchmark verdict

`hard` set, 3 complete real runs with one lineup, pooled (90 item-runs per system), strict exact accuracy:

| System | Accuracy | 95% CI | Cost incl. judge | vs cheap |
|---|---|---|---|---|
| Always Cheap | 94.4% | 87.6-97.6% | 0.97 | 1.0x |
| Static Router | 95.6% | 89.1-98.3% | 2.70 | 2.8x |
| Always Strong | 95.6% | 89.1-98.3% | 9.89 | 10.2x |
| SmartRouter | 97.8% | 92.3-99.4% | 6.94 | 7.2x |

SmartRouter vs Always Cheap: 5 wins / 2 losses, sign test p = 0.45. 6 of 7 escalations were unnecessary.
**The cost-without-quality-loss claim is not supported**: Always Cheap dominates on this dataset.
Next experiment: `python -m benchmark.run --dataset hard_v2 --exec-code --no-reference-eval --delay 2`.

## Remaining accepted risks

- No authentication (local single-user boundary); judge can be persuaded by answer text; code execution is not a sandbox;
  SQLite store is single-process; no automatic log retention; heuristic router; benchmark small and saturated.

## Router finding (finalization)

On all 152 dataset questions the difficulty score never reaches 0.70, so the router never selects strong; the "hard"
keyword list fired once. Routing is effectively category -> tier (general knowledge -> cheap, everything else -> mid).
On the recorded `hard` runs the cheap model was right on 74/78 questions the router sent to mid. Kept as is (deterministic,
explainable, too little failure data to learn from); documented in README.
