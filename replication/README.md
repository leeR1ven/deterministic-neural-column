# Replication of `deterministic-neural-column` from its written description

This directory is an independent, from-scratch rebuild. Nothing under the
original repository's `code/` was read, listed, grepped or imported.

Start with **[REPLICATION_REPORT.md](REPLICATION_REPORT.md)** — it contains the
mental model, everything I had to guess, all measured numbers, the full
ambiguity list, and the judgement.

## Quick facts

- Python 3.13 + numpy only, CPU, deterministic.
- All headline behaviours reproduce qualitatively: sparse word codes
  (Jaccard 0.011 for 12 words, 0.034 vs floor 0.033 for 160 words), 1 s working
  memory (12/12 identities), a 12/12-step chain at 12/36 words, locally-grown
  concept cells (34–36/256 vs 0–5 control), an 8/8 two-way muscle loop over
  3 seeds, and the four think/act/instinct/suppress scenarios.
- The scaling laws (`m·f ≲ 2`, "code length 40–70") and the
  concept-overlap-0.000 claim did **not** reproduce.

## Layout

| path | contents |
|---|---|
| `dnc/core.py` | n-gram encoder, ±1 complement, mean subtraction, sparse projection, per-cell threshold, per-item gain, metrics |
| `dnc/loop.py` | slow-variable working memory, inhibitory pool, L5 readout |
| `dnc/readout.py` | shared/per-transition motor maps, template matching, fidelity definitions |
| `dnc/concepts.py` | local concept layer (top-3 Hebbian + competition) |
| `dnc/pathways.py` | think / suppress / act circuit |
| `01_codes.py` … `10_concept_sensitivity.py` | experiments, in report order |
| `out/*.log` | raw outputs |

## Run

```
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe 01_codes.py
```

(and likewise for `02` … `10`). Each script is self-contained and takes seconds
to a couple of minutes.
