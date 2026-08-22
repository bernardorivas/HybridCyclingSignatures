# Refined Compass probability analysis (v3, C=0.75)

> **Artifact availability.** Generated Fourier streams, plans, signatures,
> summaries, and figures under `experiments_planned/outputs/` are intentionally
> not stored in Git. Completed-run validation requires the local artifacts or
> a fresh rebuild using the versioned builder and commands below.

This directory is an isolated successor to `shared_probability_v2`. It binds
the refined continuous-period Compass bundle and calls the unchanged shared
Julia probability kernel. It never reads or modifies the frozen v2 plan,
signatures, summaries, or figure.

The frozen pilot protocol is `dt=.00125`, segment lengths `160:80:9600`
(durations `.2:.1:12`), 20 independently sampled starts per duration, seed
`20260820`, infinity-normalized tangents, cover `.75 Q_1 x Q_2`, dynamic
metric `C=.75`, radii `0:.002:.5`, and coefficients in `F_43`. Starts are
paired across the five equal-length streams. The statistic remains
`P(rank > 0)`.

Periodic guides are the OLS continuous periods in the refined certificates,
not the older rounded stored-row values. The driver requires `beta1(Y)=1`,
rejects exact full-period lifted recurrences inside any planned 12-unit
window, records the `h<.18` target, and hard-rejects any consecutive-sample
bound `h>=.5`. It does not claim global noncommensurability of the 469-unit
analysis stream.

The default action is read-only:

```bash
PYTHONDONTWRITEBYTECODE=1 code/venv/bin/python \
  code/period_doubling/shared_probability_v3/compass_probability_v3.py check
```

`materialize` creates an immutable `materialized_not_executed` plan beneath
`experiments_planned/outputs/shared_coauthor_protocol/` after all Python and
Julia preflights pass. `execute` is the only action that computes signatures;
the author approved that action and all five jobs completed on 2026-08-22.
`validate` is read-only. `summarize` writes a new compact summary only after
all five results validate. Every write refuses an existing target and result
bindings are published atomically.

## Completed `C=0.75` pilot

The validated output root is
`experiments_planned/outputs/shared_coauthor_protocol/`
`compass_refined_v3_probability_linf_C0p75/`. It contains 119 durations and
20 trials for each of five cases, or 11,900 trials in all. The paired start
tables are byte-identical. Every case has `beta1(Y)=1`, no trial has a zero or
near-zero birth at the `1e-12` audit threshold, and the entire `r=0` row has
`P(rank > 0)=0`. Thus the refined cadence and continuous periods remove the
earlier exact sample-grid lock.

In period-1/2/4/8/chaos order, the measured bounds `h` are
`0.0685227462000182/0.0757526962952281/0.109046270801063/`
`0.0959224712507916/0.155813652930367`; the first common
strictly curve-resolved grid row is `.156`. No common curve-resolved radius
has strictly ordered period-1/2/4/8 pooled P25, P50, or P75 first or sustained
onsets. At `.156`, the pooled sustained-P50 onsets are
`.9/1.1/1.2/1.2`. The tuning and validation ordered P75 bands do not overlap.
The numerical repair therefore passes, but the preregistered period-
discrimination acceptance test fails.

`compact_summary_v2/` is a validated read-only postprocessing of the stored
birth vectors. Its low-radius matrices use exact thresholding on
`r=0:.0001:.02`; that whole grid is below every `h`, with no interpolation,
smoothing, or signature rerun. The full-grid `compassgait_C0p75.pdf` and the
low-radius `compassgait_C0p75_lowr.pdf` are code-root diagnostics only. Neither
has been copied or wired into the paper, which retains the existing
`compassgait_C5p0.pdf` pending an author decision.

Immutable SHA-256 anchors are:

- bundle manifest:
  `371cfb4bf751ab6f4b04226dece7e1049fceb516db638fee44f111ada352b442`;
- plan:
  `3245bbd941debf5eebda6ba0ddf2ae618154095757563d644166d914f8eab45a`;
- paired starts:
  `b7d1f2ca415a96e655d841761f6f62562f685e3f003296fa9ac1b55a349a9d92`;
- period-1/2/4/8/chaos result bindings:
  `958196b9a422e6b43e3de6eb544525e6d3cad96d7dc53821a047d40ba1b962d4`,
  `8f2e4c8251d47c4c62182c25605f338f925e4b55b4e60407e1c5581b3e680eaf`,
  `5af6a683acebb59cd958deb1cb3d1283ba9cba20b3550d5c1a6bd3a91f3d12ed`,
  `ef6f1117005f8da7e2510029edf38ec498564f0bfbe81f38824a05f85b566e6e`,
  and `6d509cefde929aa7016fb2631ef9bced6a21f5d351d4e7f1d8d807a1441defc3`.

The plan, execution orchestration, validation, summaries, and render sidecars
record Python `3.13.15` with NumPy `2.4.2` from the isolated project
environment. The Homebrew base interpreter now imports NumPy `2.5.2`; the
isolated environment still imports `2.4.2`. Raw Julia outputs, hashes, and
result bindings are unchanged, so this is recorded as environment provenance
drift rather than a result change.
