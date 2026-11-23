# Independent replication report

**What this is.** I was asked to rebuild the model in
`repo_publish/deterministic-neural-column` from the three written documents only
(`SPEC.md`, `README.md`, `RESULTS.md`), then report what a stranger can and
cannot infer.

**Rule compliance.** I did not open, list, grep, import or execute anything
under `code/`. The three documents themselves mention filenames such as
`gpu/local_adapt2.py`; I read those names but never touched the directory.
Nothing under `code/` was seen.

**Environment.** Python 3.13.7, numpy 2.4.6, CPU only, Windows. Everything is
in `D:\deepseekherness\replication\`; nothing outside it was modified.

---

## TL;DR

- **A stranger can rebuild the *architecture*.** I did, from scratch, in one
  sitting; every experiment script runs in seconds to a couple of minutes on
  CPU: sparse word codes, a slow-variable working memory, a trained motor map,
  a chain, a locally-learned concept layer, a muscle language, a two-way
  hear→answer loop, and the three output pathways.
- **A stranger cannot reproduce the *numbers*.** Almost every quantitative
  claim is underdetermined: the documents name mechanisms but not their update
  equations, do not define the evaluation metrics, and contradict each other
  across four or five model generations.
- Headline behaviours reproduced: sparse low-overlap code (Jaccard 0.011 for
  12 words; 0.034 vs the theoretical floor 0.033 for 160 words), 1-second
  working memory (12/12 identities, self-Jaccard 0.58), 12/12-step chain over
  12/36/80 words, concept cells emerging locally (34–36/256 vs 0–5 control),
  8/8 recognition and 8/8 two-way replies over 3 seeds, and the four
  think/act/instinct/suppress scenarios.
- **Not reproduced:** the capacity laws (`m·f ≲ 2`, "code length 40–70"),
  the concept-pairwise-overlap-0.000 claim, the muscle-overlap-0.050 claim,
  and the exact 37/256 count (I get 34–36 only if I choose a response
  threshold the documents never define).
- **Verdict:** rebuildable in kind, not in detail. Three missing pieces would
  fix most of it (section 7).

---

## 1. The model as I understood it

The final-generation description (README section 2, SPEC 19/22/41, RESULTS 55)
is:

```
word -> 280-channel character n-gram pattern
     -> +-1 complement (560 channels)
     -> subtract the mean input over the vocabulary        (redundancy removal)
     -> sparse signed local projection to L2/3 (1024 cells)
     -> per-cell threshold theta_i + per-item gain (code length ~70)
     -> L2/3 held by a slow variable; inhibitory pool (256) reads L2/3
     -> L5 (512) single feed-forward readout
     -> shared Hebbian map W_m (row budget 0.05) to the next word's code
     -> unthresholded cosine template matching -> next word
     -> re-inject that word's code (the "hear yourself" big loop)
```

Layered on top: a 256-cell concept layer (local Hebbian + competition) whose
category cells are measured after training, and a motor/muscle layer
(16 muscles × 5 time steps = 80 channels) used for speaking and for a two-way
hearing loop. A separate three-pathway circuit (think / suppress / act) uses a
tonic brake, a basal-ganglia inhibitory group, and a subcortical instinct
path.

Two rules are declared fundamental:

- **R1 soft-boundary Hebbian**: `Δw = η·(1 − w/W)·(x_post − θ)` when the
  presynaptic cell fires.
- **R2 per-cell multiplicative conservation**: all excitatory input to a cell is
  multiplicatively rescaled to a budget (no category distinction).

The documents also say, explicitly, that the successful final model is *not*
the spiking LIF model of RESULTS rounds 1–31 (which they declare failed); it is
a deterministic threshold network. I rebuilt the deterministic one.

---

## 2. What I implemented

| File | Contents |
|---|---|
| `dnc/core.py` | deterministic FNV n-gram encoder, ±1 complement, mean subtraction, sparse signed projection, per-cell threshold, per-item gain, Jaccard/cosine/overlap metrics |
| `dnc/loop.py` | L2/3 slow variable, inhibitory pool (256, heterogeneous thresholds), L5 readout with activity normalisation |
| `dnc/readout.py` | shared and per-transition motor maps, template matching, three candidate fidelity metrics |
| `dnc/concepts.py` | 256-cell concept layer, top-3 winner Hebbian + decay, selectivity measurement |
| `dnc/pathways.py` | three output pathways with the documents' own constants |
| `01_codes.py` | sparse-code density/overlap |
| `02_hold.py` | E1/E2 hold test and tau sensitivity |
| `03_chain.py` | fidelity, shared vs per-transition map, chain |
| `04_concepts.py` | class structure and concept-cell emergence |
| `05_language.py` | muscle patterns, listener, two-way loop, 3 seeds |
| `06_capacity.py` | capacity-law sweep |
| `07_scale.py` | chain at 12/36/80 words, 6 seeds |
| `08_pathways.py` | think / act / instinct / suppress |
| `09_context.py` | two-word context prediction, 3 seeds |
| `10_concept_sensitivity.py` | concept count vs the undefined response threshold |
| `out/*.log` | raw outputs of all of the above |

Run: `python <script>.py` with the Python 3.13 interpreter. Two runs of
`03_chain.py` are byte-identical (SHA256
`FC3ED4E2…A8AC2F`), satisfying the determinism claim.

---

## 3. Claimed vs measured

### 3.1 Sparse word code (headline #1)

Claimed: 1024 cells, ~50 cells/word (README), ~70 cells/word and pairwise
overlap 0.051–0.065 (SPEC 19/41); 40-word structured vocabulary has
within-class 0.201 / between-class 0.019 (ratio 10.5, SPEC 28); at scale the
overlap reaches the random-code floor `f/(2−f)` (SPEC 24/37).

Measured (seed 0):

| vocabulary | code length | density | pairwise Jaccard | floor `f/(2−f)` | ratio |
|---|---|---|---|---|---|
| 12 Greek words | 69.2 (56–72) | 0.0675 | **0.0112** (max 0.050) | 0.0350 | 0.32 |
| 40 structured (8×5) | 65.0 (47–71) | 0.0635 | 0.0286 | 0.0328 | 0.87 |
| 160 pseudo-words | 66.0 | 0.0644 | **0.0341** (max 0.246) | 0.0333 | **1.02** |

- Low-overlap sparse codes: **yes**. The 160-word result sits on the random
  floor, matching SPEC 24/37. The 12-word code is *better* than the floor
  (0.32×) because with 12 items the per-cell 90th-percentile threshold makes
  cells nearly single-item.
- Structured vocabulary: within-class 0.1182, between-class 0.0183,
  **ratio 6.45×** (claimed 10.5×). Same qualitative structure, weaker.
- Concept = union of the 5 class codes: sizes 199–229 cells (19.4–22.4% of
  1024), pairwise Jaccard **0.0923** (max 0.179). The documents claim **0.000**
  (SPEC 28). Under the stated construction (union of five 70-cell codes whose
  within-class overlap is 0.2) 0.000 is arithmetically impossible: two
  220-cell unions in 1024 cells overlap by chance at ~0.09. It could only be
  true if "concept code" meant the small set of strictly selective cells,
  which is a different object. I treat 0.000 as an error or a different
  measurement, not as a target.
- Side note: the concept code is 21.6% dense, which violates the documents'
  own S5 ("active fraction < 20%") if concept codes are network states.

### 3.2 Working memory (E1/E2)

Claimed: after input removal, still active after 1 s; held states are
content-specific, overlap 0.051–0.065, self-similarity 0.61–0.66 (SPEC 19/41);
E2 = different inputs leave states with correlation/Jaccard < 0.9.

Measured, 12 words, 1000 steps (1 s) after input offset:

| tau (steps) | cells alive after 1 s | self-Jaccard | identity (held most like its own code) | pairwise Jaccard |
|---|---|---|---|---|
| 500 | 0–1 | 0.00 | 3/12 | 0.0000 |
| 1000 | 0–9 | 0.06 | 11/12 | 0.0000 |
| 2000 | 3–29 | 0.28 | 12/12 | 0.0000 |
| **4000** | **18–50** | **0.58** | **12/12** | **0.0000** |
| 8000 | 19–67 | 0.71 | 12/12 | 0.0000 |

- E1: **yes**, at tau ≥ 2000. This is the single most sensitive number in the
  whole model: the documents say "tau 400" (SPEC 38) with no units; at 400
  steps the state is dead well before 1 s. I had to use 4000 steps.
- E2: **yes** — the held state is 12/12 most similar to its own input code,
  and held states are nearly disjoint (0.0000 pairwise Jaccard). The absolute
  overlap is lower than the claimed 0.05 because the held code is a shrinking
  subset of the presented code.
- The state is a **decaying trace**, not a self-sustaining attractor. That is
  consistent with the documents' "old information disappears within seconds",
  but not with the earlier spiking model's "sustained 56 Hz for 3 s".

### 3.3 Chain (headline #3)

Claimed: 7/7 or 16/15 or 60/59-step chains, 4/4–6/6 seeds, a single shared map
(SPEC 19/41/45–48).

Measured:

| setting | chain | held identity | L5 identity |
|---|---|---|---|
| 12 words, 12 steps, raw 280-bit target | **12/12** | 12/12 | 12/12 |
| 12 words, 12 steps, 1024-cell target | **12/12** | 12/12 | 12/12 |
| 36 words, 12 steps, raw target | **12/12** | 36/36 | 36/36 |
| 80 words, 12 steps, raw target | 11/12 | 80/80 | 80/80 |
| 80 words, 12 steps, 1024-cell target | **12/12** | 80/80 | 80/80 |
| seeds 0–5, 12 words | **12/12 every seed** | 12/12 | 12/12 |

- A shared map, no per-transition blocks, walks the chain. My step count is
  12, not the documents' 7/16/60, so this is a qualitative reproduction.
- The chain is easy once the readout is right: output → recognised word → its
  code re-injected. That is exactly what SPEC 41 says the model does.
- Ambiguity: the documents never say whether the slow variable is reset
  between chain steps. I reset it. With accumulation the state becomes a
  mixture and the chain degrades.

### 3.4 Motor map and "fidelity" (headline #4a)

Claimed: shared map fidelity 0.58 (their "random level"), per-transition blocks
1.00; motor output reproduces the next word; readout by unthresholded cosine
template matching (SPEC 13/19/41).

Measured, 12 words, drive vs the next word's 280-bit code:

| map | mass precision | cosine | top-Jaccard |
|---|---|---|---|
| shared 280-row map | 1.000 | **0.824** | 1.000 |
| per-transition blocks | 1.000 | **1.000** | 1.000 |
| mismatched state (control) | 0.149 | 0.120 | 0.136 |

- Both maps read out the next word **12/12** by cosine template matching.
- The documents never define "overlap/fidelity". The 0.58 "random level" is
  *not* Jaccard at 13% density (that would be 0.0695); it is not cosine
  (0.13); it is not mass precision. I report three metrics and cannot say
  which one they meant. **This is a serious reproducibility gap**: the
  headline numbers 0.58 and 1.00 are not tied to an operational definition.
- The per-transition blocks are trivially 1.000 because each block's rows are
  zero outside its target; there is no interference by construction.

### 3.5 Concept layer (headline #4b)

Claimed: 256 cells, 100 inputs, top-3 winners, local Hebbian + decay, 300
epochs; after training 37/256 strict category cells vs a shuffled-label
control of 0.1; 40/40 class and word decoding; concept cells are "grown, not
assigned" (SPEC 30, RESULTS 54).

Measured after 300 epochs, 40 structured words, 8 classes × 5:

| seed | selective cells / 256 | shuffled-label control | LOO class decode |
|---|---|---|---|
| 0 | 20 (at rel. threshold 0.5) / **36** (at 0.3) | 2 | 38/40 |
| 1 | 27 / **34** (at 0.3) | 5 | 32/40 |
| 2 | 23 / **35** (at 0.3) | 1 | 33/40 |

- Concept cells do emerge from local rules only: 34–36 vs control 0–5 at a
  response threshold of 30% of each cell's max, 0 before training.
- The count depends entirely on the undefined definition of "responds". The
  documents' 37/256 is reproduced only by choosing 0.3×max; at 0.5×max it is
  20–27. The control cell count is 1–5, not the claimed 0.1.
- The claimed "word decode 40/40" is trivial (nearest template includes the
  word itself); the documents never describe a held-out word protocol. I
  report leave-one-out *class* decode (32–38/40), which is the meaningful
  number.
- Concept code pairwise overlap is 0.092, not 0.000 (see 3.1).

### 3.6 Muscle language and two-way loop (headline #4c)

Claimed: 16 muscles × 5 time steps = 80 channels; each concept = 3 muscles at
specific steps; concept-to-concept overlap 0.050 mean / 0.200 max; speaking is
understood 7/7 and the reply is correct 7/7 (SPEC 28/29/52/53).

Measured, 3 seeds:

| quantity | measured |
|---|---|
| muscle pattern overlap | mean 0.0143, max 0.2000 |
| heard code | 152–160 of 1024 cells, pairwise 0.051–0.053 |
| recognition of a heard utterance | **8/8**, all 3 seeds |
| two-way: hear c → answer with next concept's muscle pattern | **8/8**, all 3 seeds |
| internal concept chain via the same muscle map | 8/8, all 3 seeds |

- The loop works: external muscle pattern → sparse heard code → loop → L5 →
  shared map → next concept's muscle pattern. The reply was
  `[1,2,3,4,5,6,7,0]` in every seed.
- The mean muscle overlap is 0.014, not 0.050. The max 0.200 matches. The
  exact pattern-generation rule ("3 muscles at specific time steps") is not
  given; I chose three distinct random (muscle, time) units.
- "Speaking is understood" is close to tautological here because the listener
  uses the same encoder that defined the codes. The documents do the same
  (template matching against the same heard codes), so I report it but do not
  treat it as evidence of a learned language.

### 3.7 Context prediction (optional)

Claimed: two preceding words determine the next word; the same second word in
different contexts predicts different successors; the model beats a
last-word-only lookup (SPEC 21/26/43).

Measured, 6 words, 6 context pairs, 3 seeds:

| state | next-word readout | same second word → different prediction |
|---|---|---|
| last-word only | 3/6 | **0/3** |
| two-word context | **6/6** | **3/3** |

- Reproduced. The context information is carried by the residual of the first
  word in the slow variable (the held L2/3 states for pairs sharing the second
  word overlap only 0.33).
- I needed a different L5 threshold quantile for this experiment (0.85 vs
  0.90) — another undocumented tuning knob.

### 3.8 Three output pathways (optional)

Claimed: think (no go, no speaking block) → motor 0; act (go + speaking item) →
match > 0.5 (36–101 motor cells); instinct alone → 840; public suppress →
motor 0 and basal ganglia 128 (SPEC 16/33).

Measured, with the documents' constants (threshold 2.6, brake 4.5, instinct
6.0, 128 BG cells, 3 × 280 motor cells):

| scenario | motor cells | match | BG |
|---|---|---|---|
| THINK (no go, no block) | 0 | — | 0 |
| THINK with item (no go) | 0 | 0.000 | 0 |
| ACT (go + item 0) | 27 | **1.000** (precision/recall 1.0) | 0 |
| ACT (go + item 2) | 27 | **1.000** | 0 |
| INSTINCT alone (itch) | **840** | — | 0 |
| PUBLIC SUPPRESS (itch + hold) | 0 | — | **128** |
| HOLD (go + item + hold) | 0 | — | 128 |

- All four scenarios reproduced exactly, including the 840 figure.
- **But this is not evidence for the architecture**, and README says so
  itself: three of the four scenarios are guaranteed by the constants, and
  ACT is exact because each speaking item has its own 280-cell block whose map
  rows are zero outside the target. I confirm README's confession.
- The 2.6 / 4.5 / 6.0 numbers are in motor units that are not commensurate
  with the map's 0.05 row budget; I had to insert a gain of 120 (not given).

### 3.9 Capacity laws (NOT reproduced)

Claimed (SPEC 24/45–48): transitions ≲ 2 × output channels / code length;
code length 40–70; 36 words fail on a 280-channel output and work on 1024;
120 words need L5 2048 because words × L5 density ≲ 1.

Measured, identical pipeline, random sparse targets, cyclic transitions:

| words | output width | code len | m·f | readout | cosine | top-10 precision |
|---|---|---|---|---|---|---|
| 12 | 280 | 70 | 3.0 | 12/12 | 0.858 | 1.00 |
| 36 | 280 | 70 | 9.0 | 36/36 | 0.777 | 1.00 |
| 80 | 280 | 70 | 20.0 | **80/80** | 0.657 | 1.00 |
| 80 | 1024 | 70 | 5.5 | 80/80 | 0.598 | 1.00 |
| 80 | 2048 | 70 | 2.7 | 80/80 | 0.617 | 1.00 |
| 80 | 2048 | 40 | 1.6 | 80/80 | 0.651 | 1.00 |
| 80 | 2048 | 20 | 0.8 | 80/80 | 0.693 | 1.00 |

- **I could not reproduce the capacity law at all.** With nearly orthogonal
  L5 states (pairwise 0.043–0.045; nearest-neighbour identity 80/80) the
  Hebbian outer-product map separates 80 transitions even at m·f = 20.
- The claimed discrete failure ("the 10 strongest driven cells belong to the
  target only 4/10 of the time at 36 words") does not occur: top-10 precision
  is 1.00 at every setting.
- The "code length 40–70" law does not occur either; 20-cell codes read out
  80/80 in my reconstruction.
- Interpretation: those laws may be real for the authors' particular L5
  thresholding / delta-rule / block-selection choices, but they are **not
  consequences of the mechanisms as written**. A stranger cannot derive them.

### 3.10 S5 and determinism

- S5 (active fraction < 20%): word codes 6.3–6.9%, held states 3.8%, L5 8–16%,
  heard codes 15–16% — all fine. **Concept unions are 21.6%**, which violates
  S5 if they count as network states.
- Determinism: two runs of the chain script are byte-identical (SHA256 match),
  matching the "two runs identical" claim.
- Seed stability: 6/6 seeds give a 12/12 chain; 3/3 seeds give 8/8 two-way
  replies; 3/3 seeds give 6/6 context and 3/3 context contrast. The pipeline
  is more seed-stable than several stages of the original log, but I did not
  try to reproduce the seeds where the original failed.

---

## 4. Every place the description was ambiguous, underspecified, or missing a number

Grouped by subsystem. "Docs" = what is written; "I chose" = what I had to
invent. These are the things a stranger cannot infer.

### A. Input encoding and the A level

1. **Character n-gram parameters.** The docs say "character n-gram" and
   "±1 complement" but give no n, no hash, no channels per gram, no channel
   count per word. I chose 2- and 3-grams with boundary markers, 3 channels
   per gram, 280 channels. Input density came out 0.10–0.12, near their 12%.
2. **Complement definition.** "±1 补码" is not defined. I used
   `[2x−1 ; −(2x−1)]` (560 channels). A different complement changes all
   overlap numbers.
3. **Which mean is subtracted, over what set.** "减去平均输入" — over the
   vocabulary at fit time? a running average? per item? I used the mean over
   the fitted vocabulary. This is the single most important step for
   orthogonality; a running mean would change the code.
4. **"Local 3-D stack / topological projection" has no geometry.** SPEC 10
   describes a 3-D volume, 4 inputs per neuron, `dx,dy∈{−1,0,1}`, `dz∈{−2..2}`,
   fixed weights, threshold ≈ 0.98. The coordinates, the mapping from 280
   channels to the volume, and the layer sizes are not given. I replaced it
   with a random sparse signed projection (K = 80 of 560), which is what the
   documents elsewhere call the wrong thing ("random projection averages the
   pattern into uniform drive"). My codes still work, but this is a
   **different encoder** from the one they describe.
5. **Projection weights.** Sign distribution, magnitude, normalization
   (`1/√K`? `1/K`? row-sum 1?) are not given. I used ±`1/√K`, K = 80.
6. **Per-cell threshold.** "90th percentile" of what distribution — the
   current item, the vocabulary, a running history? I used the 90th percentile
   over the fitted vocabulary. With 12 items this makes cells nearly
   single-item (overlap below the random floor); with 160 items it reproduces
   the floor.
7. **Per-item gain calibration.** "每项增益标定，把码长统一到 70" is stated,
   but the search range, stopping rule and whether codes must be exactly 70
   are not. My gains ranged 0.61–1.03; 12-word code lengths 56–72, 160-word
   lengths 20–72 (some words cannot reach 70 at any gain because their score
   distribution is too flat). The documents' "all codes ~70" is not
   achievable with a single threshold rule for all vocabularies.
8. **What the A level outputs.** Binary code, graded drive, or the
   normalized drive `a = gain·score/θ`? The loop needs the graded version;
   the sparse-code measurements need the binary one. I keep both.
9. **Vocabulary and chain order.** The docs never list the 12 words or say
   that the chain is Greek alphabetical order. I used `alpha..mu`.
   `chain_words(n)` for n > 24 is my own pseudo-word generator, so the
   80/160-word results are not the authors' words.

### B. The PFC loop, slow variable, inhibition, L5

10. **"tau 400" has no units and no update equation.** SPEC 38 says "tau 400
    steps" (in one table) and "tau 400" (elsewhere); the spiking model used ms.
    I needed tau = 4000 steps to pass E1 at 1 s; at 400 the state is dead.
    **This one number changes E1 from fail to pass.**
11. **The slow variable's update rule.** "Input integration" could be
    `m += (a−m)/τ`, `m = m·e^{−dt/τ} + input·dt/τ`, or `m = max(m·decay, a)`.
    I used jump-then-decay (`max`) so a 20-step presentation can be held; the
    documents' figures imply presentations of 200–300 ms. With a pure leaky
    integrator and the same tau, a 300 ms presentation stores only ~7% of the
    drive and the state does not reach threshold.
12. **Activity threshold.** Is a cell active when `m > θ_i`, `m > 1`, or
    `m > 1 + inhibition`? I used `m > 1 + inhibition` with `m` in units of the
    per-cell threshold. The absolute scale is never stated.
13. **Reset policy between items/steps.** Never stated. I reset between chain
    steps and do not reset for context. This choice decides whether the chain
    is clean or becomes a mixture.
14. **Inhibitory pool.** Sizes and constants are given (256 cells, 100 inputs,
    row sum 0.65, E→I 0.34, I→E 0.52, thresholds 0.05–0.60), but not how
    I→E inhibition is applied (per-cell? global? divisive?), nor how it scales.
    I implemented a global term; under sparse codes it is **silent**
    (0–1 cells active), so it does no work in my model. The documents say
    inhibition is what prevents whole-layer ignition, but in the final sparse
    regime it never engages — another inconsistency.
15. **L5 threshold.** 0.17, 0.30 and 0.24 appear in different sections, but
    they are absolute numbers relative to a budget that is not defined. I
    calibrated a per-cell quantile (0.90, sometimes 0.85) instead.
16. **L5 fan-in K.** Never given for the final readout. K ≈ 20 appears for a
    different stage. I used K = 80 of 1024.
17. **"Drive divided by activity" (L5).** The docs say L5 input is divided
    normalized, but not by what (sum? mean? expected count?) nor with what
    floor. I divided by the current L2/3 activity and multiplied by code
    length. This makes L5 density stable as the held state shrinks; without
    it L5 dies (the documents report exactly that failure).
18. **L5 recurrence contradiction.** README says the PFC is an "L2/3 ⇄ L5
    small loop"; SPEC 19 says "L5: single feed-forward readout, does not
    participate in storage". I implemented read-only L5. There is no stated
    L5→L2/3 weight, so the "⇄" cannot be built as written.
19. **Readout delay.** How long after input offset the state is read is never
    given. I used 300 steps for readout and 1000 for E1. The measured overlap
    and identity depend on this.

### C. The motor map W_m

20. **The learning rule is named but never written.** "Hebbian", "120 epochs",
    "eta = 3e-2", "K = 300", "row budget 0.05", "anti-Hebbian suppression of
    other blocks" — no update equation. I used the fixed point of
    `C += target ⊗ state`, then row-normalized to 0.05. There is no η, no
    epoch count, and no anti-Hebbian term in my implementation. A different
    rule (delta rule, Oja, anti-Hebbian) would change the fidelity numbers.
21. **What the map reproduces.** Across sections it is the 280-bit raw input
    code (§12/13), the 1024-cell L2/3 code (§23), the 70-cell concept code
    (§45), and the 80-channel muscle pattern (§52). These are different
    models. I implemented the 280-bit and the 1024-cell versions.
22. **How `W_m` is trained.** With the word's own code as target (teacher
    forced)? With the next word's code after a hold? On presentation states or
    held states? I trained on readout-time held states, which is what
    inference sees. The documents never say.
23. **Row budget semantics.** "row budget 0.05" could mean row sum, row L1,
    row max, or mean. I used row sum. The documents elsewhere complain that
    row normalization "spreads a row that serves two tasks", implying row sum;
    but this is inference, not statement.
24. **"Overlap / fidelity" is undefined.** As shown in 3.4, the quoted
    random level 0.58 cannot be any standard overlap of 13%-dense patterns.
    Three plausible metrics give 0.15 / 0.12 / 0.14 for a mismatched state in
    my run. The headline "0.58 → 1.00" cannot be checked without the metric.
25. **Per-transition blocks vs shared map.** Both are claimed at different
    times. The block-selection rule ("block energy + previous-item match",
    SPEC 13) is not written as an algorithm; my max-drive selection happened
    to work, but the documents report that exact criterion failing.
26. **Threshold vs no threshold at readout.** The final text says
    unthresholded cosine template matching; earlier texts use absolute 0.02 or
    0.4×budget. The chain result depends on which; I used template matching.

### D. Concept layer

27. **Learning-rule constants.** Top-3 winners, "move inputs toward the
    pattern", "slight decay", 300 epochs are given; learning rate, decay
    magnitude, weight normalization, and whether losers change are not. I used
    lr 0.15, decay 3e-4, L1 renormalization after each update, winners only.
28. **What "responds" means.** The strict concept criterion ("≥2 words of one
    class, 0 outside") needs a response threshold. None is given. My count
    goes from 20 to 36 cells depending on whether the threshold is 0.5 or 0.3
    of each cell's max. **The 37/256 claim is not a prediction without this
    number.**
29. **Word decode 40/40 is trivial** as described (nearest template over the
    training words includes the word itself). No held-out word protocol is
    described. I report leave-one-out class decode (32–38/40).
30. **Concept code construction is inconsistent.** §28: union of the 5 word
    codes. §30: average response of the 5 words in layer 2. §29: "strict
    concept cells (96)". These give different densities (my union is 21.6%,
    violating S5) and different overlap (mine 0.092, claimed 0.000).
31. **Number of concepts.** 37 < 40 is claimed, but 37 is below the number of
    words only because there are 8 classes; it is not a general law.

### E. Muscle language / two-way

32. **Muscle pattern generation.** "3 muscles at specific time steps" —
    which muscles, whether a muscle may appear at two steps, and whether
    patterns are hand-assigned or learned are not specified. I used 3 distinct
    random (muscle, time) units; mean overlap 0.014 vs claimed 0.050.
33. **Listener calibration.** "Calibrate the heard code to 200 cells" — with
    what threshold rule, gain, and fan-in? I used code_len 200, θ quantile
    0.80, K = 40.
34. **Speaking is a lookup.** The concept→muscle mapping is a table in my
    implementation, as the documents describe it; there is no stated learning
    rule for it, so "speaking is understood" is not evidence of a learned
    language.
35. **What the reply is.** "Predict the next concept and answer with its
    muscle pattern" is clear enough, but whether the answer is the *current*
    or *next* concept is only implied by the chain examples. I used next.
36. **Channel counts are incompatible across sections**: 280 motor cells per
    item, 840 total, 80 muscle channels, 1024/2048/4096 output layers,
    2-D hand position in `motor_dynamics2.py`. No canonical motor interface.

### F. Three pathways

37. **Units.** Motor threshold 2.6, brake 4.5 and instinct 6.0 are not
    commensurate with a map whose row budget is 0.05. I inserted a gain of 120
    (not given) so target cells reach 6.0. Without a stated gain, the circuit
    cannot be assembled.
38. **Basal-ganglia application.** "128 inhibitory cells" and "subtract 6.0"
    are given, but not the projection, the per-cell scaling, or whether BG
    also suppresses the cortical pathway. I subtracted 6.0 globally from the
    motor drive.
39. **Instinct lights all 840 cells.** The 840 figure implies all three
    280-cell blocks are driven; why the itch is not block-specific is not
    stated.
40. **Three of four scenarios are tautological**, as README admits. I
    reproduced them exactly; they are not evidence for the architecture.

### G. Scale, evaluation, protocol

41. **The capacity law is not derivable.** `m·f ≲ 2` and "code length 40–70"
    are presented as laws, but I get perfect readout at m·f = 20 and
    code length 20. Either the laws depend on undocumented choices (L5
    thresholding, delta rule, block selection) or they are artifacts of the
    authors' implementation. A stranger cannot know which.
42. **The L5 sizing rule** (`words × L5 density ≲ 1`) is contradicted by my
    run: 80 words × 51/512 ≈ 8, yet L5 nearest-neighbour identity is 80/80
    and the chain is 12/12. The rule is not a consequence of the described
    mechanism.
43. **"Correlation < 0.9" (E2)** does not say which correlation (Pearson on
    rates? pattern correlation over cells? Jaccard?). I measured Jaccard and
    cosine; both pass.
44. **"Permutation test p < 0.01" (E3)** is stated without a test statistic,
    number of permutations, or what is permuted. I did not attempt a formal
    permutation test; the context contrast is a direct control.
45. **Presentation protocol.** Durations (200/300 ms), inter-item gaps, number
    of steps per word, and whether the network is reset between trials are
    scattered and inconsistent. I used 20 steps of injection (the jump-then-
    decay form makes the exact duration irrelevant) and 300 steps of hold.
46. **Seed protocol.** "6 seeds" is claimed in several places, "3 seeds" in
    others. I used 3–6 depending on cost, and I did not reproduce the seeds
    where the original failed.
47. **Train/test split.** README says rows 1/3/4/6 are training-set results;
    the only held-out test (two_stage6) is "memorisation only". I trained and
    evaluated on the same transitions, like most of their results, and I did
    not implement a held-out-transition test. This is a fundamental limitation
    of the whole claim set, not just of the description.
48. **The "three-dimensional stack" reference implementation is not vendored**
    (README says no fixed commit, no copy), so "matches the reference rule"
    cannot be verified by anyone.

---

## 5. What I deliberately did not attempt

- The spiking LIF five-area model of RESULTS rounds 1–31 (declared failed by
  the authors; thousands of lines of mechanism).
- The 10^5-neuron throughput benchmark and CUDA work.
- STDP / eligibility-trace / dendritic-branch variants from the middle rounds.
- A held-out-transition generalisation test; the documents already report it
  fails ("memorisation only").
- Exact reproduction of the authors' seeds and their reported failures.

These are not needed to answer the question: the final model is the
deterministic threshold network, and I rebuilt that.

---

## 6. Judgement: is the description sufficient?

**No — not to the standard the documents claim.** They are an experiment log,
not a specification, and they say so in places (README: "this diagram is a
concept diagram, not any single script's implementation"; "no script implements
the whole diagram"). A competent stranger can rebuild *a* model with the same
qualitative behaviours — I did — but cannot know whether it is *the* model,
cannot reproduce the reported numbers, and cannot even evaluate several of the
claims because the metrics are undefined.

The strongest evidence for "insufficient":
- The same phrase ("L5 threshold 0.17") is repeated, but the budget it is
  relative to is never defined; across my runs the L5 readout density moved
  between 42 and 111 cells (8–22% of 512) purely with that threshold choice.
- "tau 400" is stated twice with different implied units; at face value E1
  fails, and passing it required a 10× change.
- "Overlap" means at least four different things in different sections
  (Jaccard of codes, cosine of states, drive mass, template-match score), and
  the headline random level 0.58 matches none of them at the stated density.
- Four incompatible model generations coexist in the same three files, with
  different output widths, cell counts, code lengths, motor interfaces and
  even different definitions of "concept code".
- The capacity laws — presented as the main scaling result — do not follow
  from the stated mechanisms in my rebuild.

**What is genuinely well described**: the *idea* of the deterministic
threshold rule, the per-cell budget comparison, mean subtraction for
orthogonality, the slow-variable-plus-inhibitory-pool loop, the shared Hebbian
map with template readout, and the local concept-learning recipe. Those were
enough to get every headline behaviour working. The documents are good at
"what to build" and bad at "exactly how, with which numbers, measured how".

---

## 7. The three most important missing pieces

1. **One canonical configuration with exact update equations and units.**
   Specifically: the slow-variable equation and its τ in steps; the activity
   threshold in absolute units; the L5 threshold's reference budget; the
   readout delay; the reset policy between items; and which of the 280 / 1024
   / 2048 / 4096 output versions is the model. Without this, E1 and the chain
   are parameter-dependent, and the reported numbers are not reproducible.

2. **The learning rules in formula form.** `W_m` (η, target, row
   normalization, whether there is an anti-Hebbian term, number of epochs),
   the concept layer (lr, decay, normalization, response threshold), and the
   A-level threshold/gain calibration (search range and exact stopping rule).
   The documents name all of these mechanisms but never write the equations.

3. **Operational definitions of every metric and the evaluation protocol.**
   "Overlap", "match > 0.5", "responds", "held state most like itself",
   "permutation test p < 0.01", "code length", "active fraction"; plus the
   presentation durations, reset policy, seed list, train/test split, and a
   reference output file to diff against. Without these, even a correct
   rebuild cannot be scored, and the 0.58 / 1.00 / 0.000 / 37 numbers are
   unfalsifiable.

A fourth, if allowed: **a single self-contained reference script and its
parameter file for the final model** (the documents point at
`motor_concept5.py` / `concept_final3.py` / `local_adapt2.py` as different
"final" models). The README already admits that no script implements the
diagram; that is the root cause of most of the ambiguity above.

---

## Appendix: exact reproduction commands

```
cd D:\deepseekherness\replication
C:\Users\Administrator\AppData\Local\Programs\Python\Python313\python.exe 01_codes.py
... 02_hold.py ... 10_concept_sensitivity.py
```

Raw outputs are in `out/*.log`. The chain determinism check (two identical
SHA256 hashes) was run separately and is quoted in 3.10.
