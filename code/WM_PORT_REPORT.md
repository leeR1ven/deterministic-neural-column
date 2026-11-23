# WM_PORT_REPORT

**Task.** Take the working-memory mechanism out of `walk_rel2.py` and put it into the
sensory-stack demo `local_adapt2.py`, then measure whether the transplant works.
Deliverable: `gpu/wm_port.py` (new file; no existing file was modified).

**Verdict: the transplant works.** With `walk_rel2`'s loop unchanged (per-cell trace
`m`, per-cell threshold `TH23v`, inhibitory pool) and `local_adapt2`'s sensory path
unchanged (`Local3DStack` → `S` → `W_in`), on seeds 0/1/2 and with the only addition a
per-item threshold that turns the dense sensory drive into the sparse 0/1 code the loop
expects:

| measurement | local_adapt2 (baseline, `wm_tests.py` seed 0) | wm_port (code 150, seeds 0/1/2) |
|---|---|---|
| SANITY  present k, hold 10, read out | 6/6 | **6/6, 6/6, 6/6** (matching overlap 0.72–1.00) |
| PERTURB flip 5 / 20 / 50 %, hold 10 | 0.681 / 0.248 / 0.229 | **0.836 / 0.821 / 0.806** (seed 0), 0.782/0.768/0.712 (1), 0.807/0.773/0.762 (2) |
| DISTRACTOR  A then B, which is left | A survives 3/3 (new input cannot overwrite) | **B 30/30** ordered pairs on every seed (updateable) |
| trace deleted (KAPPA=0) | — | **held 0 of 1024, sanity 1/6** (the trace is what holds it) |

Three honest qualifications, all measured below:

1. **The hold leaks at this code size.** It is a constant state for ~7–10 steps and decays
   afterwards (156 → 146 → 37 cells at 0 / 10 / 30 free steps). The reason is identified:
   the denser sensory codes drive the inhibitory pool 2–3× harder, so the hold-phase drive
   is `rel_hold ≈ −0.14` where `walk_rel2` has `−0.03`. That value sets the memory lifetime
   and cannot be fixed by changing `TAU` (measured, §8.1).
2. **The L5 perturbation asked for in the task is inert in `walk_rel2`'s wiring**, because
   L5 has no path back into L2/3 there. Numbers and the alternative (restore
   `local_adapt2`'s `W_p` feedback) are in §5.2 and §8.3.
3. **The memory is updateable but not distractor-resistant**: the newest input wins in
   all 30 pairs; the earlier item never survives. §8.4.

---

## 1. What the two mechanisms actually are (measured, not assumed)

### 1.1 `walk_rel2.py`'s loop

`held_state()` holds a 1024-cell L2/3 pattern with three coupled quantities:

```python
d23  = bias + input - W_ie @ xi          # input is a 0/1 code, amplitude 1
rel  = d23 / BUD                         # BUD = 1 + bias = 1.03
mn   = m / m_ref                         # per-cell; m_ref frozen at the last input step
fire = (rel + KAPPA * mn * TH23v > TH23v)  # KAPPA = 1.5
m    = m + (rel - m) / TAU               # TAU = 400, the slow variable
xi   = 1[bias_i + W_ei @ fire > TH_I * bud_i]   # 256-cell inhibitory pool
```

`TH23v` is the per-cell 90th percentile of `(code + bias)/BUD` over the 12 items.

Measured with `wm_diag_ref.py` (seed 0, the 12-word configuration):

| quantity | value |
|---|---|
| code size per item | 70–71 of 1024 |
| `rel` at presentation: on-item / off-item | 1.000 / 0.029 |
| `TH23v` | never-active cells 0.029, one-item cells 0.903, multi-item 1.0 (39 cells); mean 0.186 |
| presentation count | 70–75 |
| held count after 12 free steps | 40–55, a subset of the code (37–54 of the held cells are code cells) |
| trace contrast `m_ref(on)/m_ref(off)` | 0.0207 vs <5e-5 (ratio > 400) |
| `mn` after 10 free steps, on-item cells | 0.90–0.93 |
| held-vs-presentation overlap | 0.54–0.76 |
| held pairwise overlap between items | **0.064** |
| free-run trajectory, item 0, counts | 66,71,…,74 (cue) then 31,54,35,40,40,40,40,40,40 (hold) |

Two properties matter for the port:

* the presentation drive is **1.0 on the item's cells and 0.03 off them** — a factor of 34;
* `mn` is **normalised per cell**, so it is ≈1 for every cell at the moment the input is
  removed. Discrimination comes only from the *absolute* drift `m += (rel_hold − m)/TAU`:
  a cell with `m_ref = 0.021` loses ~0.6 % of its trace per step, a cell with
  `m_ref = 1e-6` loses its (negative) trace almost immediately. **The trace can separate
  cells only if `m_ref` differs by a large factor between the item's cells and the rest.**

### 1.2 `local_adapt2.py`'s hold

`local_adapt2` has no slow variable: `settle()` recomputes `n23 = (drive > th23*bud)` from
`W_p @ x5 + bias + W_in @ s` every step. Its reported "hold" is a fixed point of that
map. The repo's own probe (`wm_tests.py`, run for this report, seed 0) gives:

* per-item theta 0.25/0.28/0.22/0.22/0.22/0.28, presentation sizes 128/127/192/384/256/311
* SANITY 6/6 (the presentation state equals itself after 10 free steps)
* DISTRACTOR: **A survives 3/3** (1.00, 0.99, 0.75) — a new input cannot overwrite it
* PERTURB on the L5 state: **0.681 / 0.248 / 0.229** at 5 / 20 / 50 %

That is the baseline the task quotes (0.23–0.68).

---

## 2. What was changed conceptually

Kept verbatim from `local_adapt2.py`: the sensory path — `Local3DStack(DIMS, seed)`,
`st.calibrate(codes, 0.12)`, `S = forward(codes)[-1].round()` normalised to the mean
count, and the `W_in` projection (each L2/3 cell averages its 80 nearest sensory cells,
row sum 1.00). Also kept: the Jaccard readout convention
`overlap = |a AND b| / |a OR b|` against stored templates.

Kept verbatim from `walk_rel2.py`: the loop above — same step order, same `BUD = 1+bias`,
same `KAPPA = 1.5`, `TAU = 400`, same `TH23v` definition (90th percentile over the items),
same inhibitory pool (`W_ei` budget 0.34, `W_ie` budget 0.65, `TH_I = linspace(0.05,0.60)`,
`bud_i = 0.34`), same `W_q` L5 readout. Nothing bisects a threshold at run time and there
is no threshold recursion on the L2/3 side.

Added, and this is the **only** conceptual change: the dense drive `W_in @ s_k` is cut at
one per-item calibrated threshold to form the 0/1 code `c_k` that the loop receives, in
the same 1024-d space as the L2/3 state (exactly like `walk_rel2`'s `SIN[k]`). The
threshold is calibrated on the **drive**, per item, by bisection to a target cell count —
which is the same kind of calibration `walk_rel2` already does with its per-item `GAIN`
(it bisects a gain so every item's code lands at ~70 cells). The L5 threshold is likewise
recalibrated (density target 70 of 512) because `walk_rel2`'s 0.17 was calibrated for
70-cell L2/3 codes; the cell counts are printed so the recalibration is visible.

Why the sensory drive must be binarised is measured in §3.1. It is not a cosmetic choice:
with the dense drive the trace cannot discriminate, and the loop goes into a limit cycle.

---

## 3. The scale problem, and the calibration

### 3.1 Dense drive vs 0/1 code, same loop, same threshold, same cell count

`wm_diag_dense.py` (seed 0) runs the *identical* loop on the *identical* sensory drive,
once as a dense drive (regime A: `d23 = bias + W_in@s − W_ie@xi`) and once as a 0/1 code
(regime B: `d23 = bias + c_k − W_ie@xi`). Both use a threshold of 0.222 of the budget,
which selects the same 156 presentation cells, and the same `TH23v` rule (each computed
from its own regime's rel, as the rule specifies).

| | A dense drive | B 0/1 code |
|---|---|---|
| `rel` on item / off item | 0.25 / 0.13 (**×1.88**) | 1.00 / 0.03 (**×34**) |
| `m_ref` on item / off item | 0.0021 / 0.0003 (**×6.6**) | 0.0188 / ~1e-6 (**×18 844**) |
| `mn` at cue end (on / off) | 1.00 / −831 | 1.00 / −2740 |
| `mn` after 12 free steps, on item | **0.361** (trace already collapsed) | **0.844** |
| fire counts, cue 10 + hold 12 | 218,127,215,66,256,27,288,15,293,134,26,103,12,64,14,44,22,33,28,33,30,29 | 29,156,156,156,156,156,156,156,156,156,156,156,156,156,156,156,155,155,153,149,149,142 |
| held count | 29 of 1024, an arbitrary phase of the oscillation | **142**, the item's state |

The dense drive produces a **limit cycle**: the inhibition and the threshold fight each
other every step (15 ↔ 293 cells), `m` is dragged negative on the troughs, and the "held"
state is whatever parity the last step lands on. The 0/1 code produces a state that is
constant for 15+ steps and then decays slowly.

The numbers explain the mechanism: `mn` is per-cell normalised, so at the end of the cue
`mn ≈ 1` in **both** groups in **both** regimes. What separates them afterwards is the
absolute drift, whose size relative to `m_ref` is what matters. With a dense drive the
on-item cells' `mn` has already fallen to 0.361 after 12 hold steps (versus 0.844 with the
0/1 code), so they lose the threshold reduction and the state dies — which cells happen to
survive is then decided by the inhibition oscillation, not by the item. With a 0/1 code
`m_ref` differs by 18 844×, the off-item cells are driven to `mn ≈ −3000` and can never
fire, and only the item's cells hold.

This is the "wrong reference" of the previous attempt, stated precisely: the threshold had
been calibrated so that the **density** at presentation was right (235 cells), but the
quantity the trace needs is the **`m_ref` contrast between the item's cells and the
background**, and a threshold in the middle of a narrow dense-drive distribution leaves
that contrast at ~5.

### 3.2 The previous attempt, reproduced and explained

`wm_demo.py` (the earlier port, run for this report, seed 0) prints:

```
threshold 0.194 -> 235 cells per item at presentation
seed 0 | tau 400 kappa 1.5 | held sizes [0, 1, 0, 5, 0, 1] of 1024
SANITY 3/6
PERTURB flip 5% -> 0.01 | 20% -> 0.01 | 50% -> 0.01
```

Two independent causes, both measured:

1. The dense drive makes the on-item drive only 1.9× the off-item drive (§3.1), so
   `m_ref(on)` is only ~3–7× `m_ref(off)`; `mn` is ≈1 for both groups and
   `rel + 1.5·mn·TH23 > TH23` reduces to `rel > −0.5·TH23`, which the tonic bias and the
   whole layer can satisfy. The layer then fires 1024/1024, the pool fires 256/256, next
   step everything is inhibited, and the loop oscillates between all-on and all-off. The
   held size (0–5) is the parity of that oscillation.
2. The perturb number 0.01 is not a measurement of anything: in `wm_demo.run()` both
   `x23` and `x5` are re-initialised to zeros at the top of the function, and the flipped
   `x5` is never passed back in. The 0.01 is the drift between two independent restarts.

The task's phrasing ("the threshold was calibrated with the wrong reference") is right;
the reference that was wrong is the `m_ref` contrast, not the density.

### 3.3 The calibration actually used

For each item, `th_k` is the largest threshold whose count is ≥ `CODE_SIZE` (bisection on
`W_in @ s_k`, 60 iterations). The resulting counts are printed per item and per seed.
`TH23v` then follows `walk_rel2`'s rule with no extra fitting:

```
TH23v_j = quantile_0.9 over the 6 items of (c_k[j] + bias_j) / BUD_j
```

With 6 items the quantile takes three coarse values — 0.03 for cells in no code, ≈0.5 for
cells in one code, 1.0 for cells in several — where `walk_rel2`'s 12 items give a smoother
spread (mean 0.186). This is a real structural difference between the two scripts, and it
is why the port's `TH23v` mean is higher (0.40–0.43) than the reference's.

---

## 4. Stage-by-stage cell counts (default configuration: code 150)

Seed 0, `python wm_port.py 0 150`:

```
[1] sensory stack
    raw last-layer counts  [317, 288, 296, 340, 479, 293] of 2800  (before the rescaling below)
    S counts per item      [335, 335, 335, 335, 335, 335] of 2800  (layer theta 0.98/0.97/0.97)
    W_in row sum 1.00 | drive per item: max 0.401 mean 0.121 std-over-cells 0.063
    drive correlation between items 0.18
[2] sensory -> L2/3 input code
    threshold per item     ['0.1984', '0.1747', '0.1842', '0.1850', '0.1839', '0.1718']
    code counts per item   [156, 179, 164, 144, 126, 151] of 1024   (target 150)
    code pairwise Jaccard  0.15
    rel on code 1.00 vs off code 0.03
    TH23v: never-active 474 | one-item 326 | multi-item 224 | mean 0.40
[3] the loop
    presentation counts    [156, 179, 164, 144, 126, 151] of 1024
    held counts (10 free)  [146, 144, 142, 139, 124, 144] of 1024
    L5 threshold 0.317 -> 98 of 512 at presentation (walk_rel2 0.17 would give 311)
    L5 at present [113, 138, 105, 79, 59, 95] of 512 | at hold [88, 78, 70, 71, 56, 79]
    TRACE CONTRAST m_ref(code)/m_ref(off-code) = [18832 18482 18662 19149 19463 18909]
    mn at hold: on-code 0.90 | off-code -6336
    inhibition pool: present [77,84,79,67,62,75] of 256 | hold [68,71,70,66,61,69]
                     rel at hold -0.12..-0.16 (mean -0.143)
    held self-vs-template 0.92 | held pairwise Jaccard 0.17
    hold trajectory item 0: 156 ×7, 155,155,151,146,148,138,133,126,116,109,101,81,78,
                            69,60,62,54,47,46,45,43,39,38,37  (30 free steps)
    control KAPPA=0: held [0,0,0,0,0,0] of 1024, sanity 1/6
```

Every number the task asked to see is there: presentation 126–179 of 1024 (in the
requested 100–400 band), held 124–146 (tens to a few hundred, item-specific), a
trace contrast of ~1.9×10⁴ where the dense drive gives ~6, and the `KAPPA = 0` control
showing the hold is carried by the trace, not by the bias.

Two details of the sensory path matter for the calibration and are printed above:

* the raw last-layer densities differ per item — 288–479 of 2800 (11–17 %) — and
  `local_adapt2`'s `S = S/S.sum * mean` rescales every item to a common sum, so the drive
  amplitude differs by up to 1.5× between items. The per-item threshold absorbs this
  (0.172–0.198), which is why a single global threshold would put some items at 90 cells
  and others at 500 — the same failure `settle_item` was written to avoid.
* the drive is **quantised**: `W_in` is a 0/1 average over 80 sensory cells and one item's
  `S` is a fixed scalar times a 0/1 vector, so `drive_j = k_j · const` where `k_j` is the
  number of active sensory cells in cell `j`'s receptive field (0–27 here). A threshold
  can only be placed between two integer counts, so the code size jumps in plateau
  sizes: for target 150 the counts land on 126 or 179, never on 150. The threshold, not
  the count, is the calibrated quantity.

---

## 5. The three measurements

All three are in `wm_port.py` and in `wm_port_final_seed{0,1,2}.log`.

### 5.1 SANITY — present k, hold 10, read out (Jaccard vs the six presentation templates)

```
seed 0: alpha .94  beta .80  gamma .87  delta .97  epsilon .98  zeta .95   -> 6/6
seed 1: alpha .97  beta .95  gamma .90  delta .72  epsilon .90  zeta .75   -> 6/6
seed 2: alpha .81  beta .93  gamma .73  delta .91  epsilon .97  zeta 1.00  -> 6/6
matching overlap 0.72..1.00; no readout with best overlap < 0.40
```

The templates are the end-of-cue states recorded from independent presentations, so this
is not a self-comparison. It is a weak criterion on its own (deleting the hold steps would
also give 6/6), and it is reported that way; the discriminating measurements are 5.2/5.3
and the `KAPPA=0` control.

### 5.2 PERTURB — present, hold 2, flip 5/20/50 %, run 10 more

The task's number is `ov_pre` = overlap with the pre-flip state. Because the held state is
not perfectly stationary at this code size, two controls are reported with it:
`ov_base` = overlap of the **no-flip** run with the pre-flip state (pure drift), and
`ov_ctrl` = overlap of the flipped run with the **no-flip** run after the same 10 steps
(1.00 = the perturbation was completely undone). `ov_imm` is the overlap one step after
the flip, which proves the flip entered the loop at all.

Population `x23` = the L2/3 state that carries the trace; three flip rules (uniform = the
`wm_tests.py` rule, frac of all cells; active = frac of the active cells turned off; swap =
count-preserving).

Seed 0:

```
--- x23 (memory state) ---
uniform  5%  ov_pre 0.836  ov_base 0.847  ov_imm 0.999  ov_ctrl 0.985
uniform 20%  ov_pre 0.821  ov_base 0.847  ov_imm 0.763  ov_ctrl 0.962
uniform 50%  ov_pre 0.806  ov_base 0.847  ov_imm 0.634  ov_ctrl 0.910
active   5%  ov_pre 0.852  ov_base 0.847  ov_imm 1.000  ov_ctrl 0.993
active  20%  ov_pre 0.857  ov_base 0.847  ov_imm 1.000  ov_ctrl 0.988
active  50%  ov_pre 0.877  ov_base 0.847  ov_imm 1.000  ov_ctrl 0.963
swap     5%  ov_pre 0.847  ov_base 0.847  ov_imm 1.000  ov_ctrl 0.997
swap    20%  ov_pre 0.848  ov_base 0.847  ov_imm 1.000  ov_ctrl 0.995
swap    50%  ov_pre 0.846  ov_base 0.847  ov_imm 1.000  ov_ctrl 0.992
```

Seeds 1 and 2 (uniform flip): 0.782/0.768/0.712 and 0.807/0.773/0.762, with
`ov_base` 0.783 / 0.817 and `ov_ctrl` 0.974/0.940/0.902 and 0.971/0.943/0.882.

Reading:

* the task's metric (uniform flip) is **0.71–0.84** on every seed — above the baseline's
  0.68 at 5 % and far above its 0.23 at 50 %;
* the **recovery** evidence is `ov_ctrl` = 0.88–0.99: even a 50 % uniform flip (512 cells
  inverted, `ov_imm` 0.63, i.e. the state is badly damaged one step later) converges back
  to the no-flip trajectory to within the residual of the flip's effect on the trace;
* at 5 % and for the `active`/`swap` rules the perturbation is undone essentially
  exactly (`ov_imm` 1.000): removing cells lowers the inhibition, which lets them fire
  again, so the trace restores them on the next step;
* seed 0's 50 % case leaves `ov_ctrl` 0.910 rather than 1.00: the 10 perturbed steps also
  write into `m` (`m` integrates `rel` every step), leaving a small permanent offset — the
  trace is the memory, and a large enough perturbation does write into it.

**The L5 flip as asked, and why it is not evidence.** In `walk_rel2`'s wiring L5 is a pure
readout (`x5 = 1[0.02 + W_q @ x23 > 0.17·bud5]`), with no path back into L2/3. Flipping
L5 cells is therefore erased by the next recomputation:

```
--- x5 (L5 readout layer), seed 0 ---
uniform 5/20/50%  ov_pre 0.631 = ov_base 0.631   ov_imm 1.000  ov_ctrl 1.000
```

`ov_pre` is *identical* to the no-flip drift for all three fractions and both other rules:
the flip has no effect on anything. The baseline's 0.68/0.25/0.23 comes from a model where
`x5` is a loop variable (`W_p @ x5` feeds L2/3); in `walk_rel2`'s wiring that loop does not
exist, so an L5 flip cannot be corrected *or* persist — it is simply not seen. This is
also why the earlier `wm_demo.py` perturbation read 0.01: it flipped `x5` in a loop that
did not read it.

Same diagnostic for the x23 flip when the pool is *not* recomputed (`x23_nopool`): 100 %
inert, `ov_ctrl` 1.000 — because `x23` is not a recurrent variable either. The state that
carries the perturbation is `(m, xi)`, so a physically meaningful flip must also update the
inhibitory pool the flipped pattern drives; `wm_port.apply_flip()` does exactly what the
loop itself does at the end of a step. Without that, `walk_rel2`'s loop cannot be perturbed
at all, and a perturbation measurement on it is vacuous.

### 5.3 DISTRACTOR — present A (10), hold 2, present B (10), hold 10

All 30 ordered pairs, both `m_ref` policies (re-referenced at the end of B, or kept from
A's cue):

```
seed 0: A survived 0/30 | B took over 30/30 | neither 0/30 | unreadable 0/30
        held 86-133 | mean ovA 0.20 ovB 0.69
seed 1: A survived 0/30 | B took over 30/30 | unreadable 0/30 | mean ovA 0.21 ovB 0.65
seed 2: A survived 0/30 | B took over 30/30 | unreadable 0/30 | mean ovA 0.20 ovB 0.68
without re-referencing m_ref at B: B 30/30 on every seed, mean ovB 1.00
```

The new input **can** overwrite the old state, which is the property `local_adapt2` lacks
(A survived 3/3 there). The readout is unambiguous: the final state matches B's template
(0.65–0.69) much better than A's (0.20–0.21), and no pair is unreadable. The result does
not depend on the `m_ref` policy. Note what this is and is not: the mechanism implements
"the most recent input wins". It does **not** hold A through B — that would need a gating
or consolidation signal, which neither script has.

---

## 6. Controls

| control | result |
|---|---|
| `KAPPA = 0` (trace term deleted, everything else identical) | held 0,0,0,0,0,0 of 1024; sanity 1/6. The hold is produced by the trace. |
| x23 flip without recomputing the pool | 100 % inert (`ov_ctrl` 1.000): `x23` is not recurrent, `xi` is. |
| x5 flip, `fb=0` | 100 % inert (`ov_pre` = `ov_base`); L5 is a readout. |
| determinism | two runs of `wm_port.py 0 150` produced byte-identical measurement lines. |
| presentation = code | the loop reproduces the input code exactly at presentation, so the code is a fixed point of the trace-augmented map (and dies at `KAPPA=0`), not an independent attractor. |

---

## 7. What worked

* The trace mechanism transplants to the sensory path unchanged: same rule, same
  constants, same step order, no threshold recursion. `KAPPA=0` shows it is doing the work.
* Cells are item-specific and sparse: presentation 126–208, held 121–156 of 1024,
  pairwise held overlap 0.16–0.18 (input codes 0.15), and 6/6 readout on three seeds.
* Perturbation recovery: `ov_ctrl` 0.88–0.99 after flipping up to 50 % of the state in one
  step (`ov_imm` 0.61–0.64), against a baseline whose 50 % flip leaves 0.229.
* Overwriting: 30/30 ordered pairs end on the new item, on all seeds, under both `m_ref`
  policies.
* Everything is calibrated before the loop and printed: the only free choices are the code
  size and the L5 density, both reported.

## 8. What did not work

### 8.1 The hold leaks, and the leak is not a `TAU` problem

At code 150 the state is constant for ~7 steps and then decays (156 → 146 → 37 cells at
0/10/30 free steps; `ov(step10, step20)` 0.46–0.57, `ov(step10, step30)` 0.22–0.35).
The cause is identified and is not the trace time constant:

```
m_ref      ∝ rel_present * cue / TAU      (what the cue deposits)
drift/step = (|rel_hold| + m) / TAU       (what the hold removes)
=> relative decay ≈ |rel_hold| / (rel_present * cue) + 1/TAU
```

`m_ref` and the drift both scale as `1/TAU`, so `TAU` cancels to first order and the
first term dominates: with `rel_present = 1`, `cue = 10`, `|rel_hold| = 0.143` the decay is
~0.014 per step, versus ~0.003 per step in `walk_rel2` (`|rel_hold| = 0.03`). Measured
confirmation: `wm_port.py 0 150 1.5 1200` (TAU × 3) gives held 150 cells at 10 steps and
44 at 30 steps — the same decay (`ov(10,20)` 0.57, `ov(10,30)` 0.29).

The quantity that could not be matched is `rel_hold`. It is set by the inhibitory pool,
which is 2–3× more active than in the reference because the sensory codes are 2–3× denser
than `walk_rel2`'s 70-cell codes: pool 62–97 of 256 firing at presentation (reference: 33),
mean `rel` during hold −0.14 (reference: −0.03). `walk_rel2` itself notes the same wall —
"codes of 138 get crushed by their own inhibition" (`walk_rel2.py`, comment above `GAIN`).
Code 150 sits above that wall; the state still reads out correctly at the 10-step horizon
the task specifies, but it is not a permanent memory.

### 8.2 The held state is a fixed point at small code sizes

At code 100–120 the held state equals the presentation state (`held self-vs-template`
0.98–1.00) and is stationary for 15+ steps (`ov(10,20)` 0.83–1.00, `ov(10,30)` 0.35–0.93).
That is a genuine memory — it dies at `KAPPA=0` and the perturbation results are 0.96–1.00
— but it is a fixed point of the trace-augmented map, not a pattern attractor: there is no
recurrent excitation in `walk_rel2`'s loop, so the state is *reconstructed from the trace*
each step rather than relaxed into from a pattern. Calling it a "basin" is only correct in
the sense that a corrupted fast state is projected back onto the trace-determined state.

### 8.3 Making the L5 flip meaningful does not work with `local_adapt2`'s `W_p`

`walk_rel2`'s loop has no L5→L2/3 path, so an L5 flip is invisible (§5.2). Restoring
`local_adapt2`'s block-random `W_p` (`fb=1.0`, code 150, L5 target 180 → L5 62–139 of 512)
puts L5 back in the loop but does not make the flip a usable probe:

```
x23 flip 50% uniform: ov_pre 0.946  ov_base 0.981  ov_imm 0.639  ov_ctrl 0.964
x5  flip 50% uniform: ov_pre 0.962  ov_base 0.956  ov_imm 0.997  ov_ctrl 0.973
```

An L5 flip of half the layer moves the next state by 0.3 % (`ov_imm` 0.997): the trace has
already locked the L2/3 cells (`mn ≈ −3000` for cells outside the item's code, so extra L5
drive cannot recruit them, and the item's cells are above threshold anyway). The only
visible effect is a slow leak into `m` (`ov_ctrl` 0.973). Turning the gain up does not
help: `fb=2.0` saturates the layer (presentation 476–839 of 1024 cells, L5 476–512 of 512)
and SANITY falls to 0.65–0.85. And the random `W_p` costs distractor selectivity: A
survives 4/30, the held state grows to 202–303 cells, mean `ovA` 0.50 vs `ovB` 0.61.

So the L5 flip cannot be the memory probe in this architecture, at any gain tested. The
measurable state is the one that carries the memory — L2/3 + the inhibitory pool — and
that is the x23 measurement in §5.2. To make L5 usable one would need a *learned* `W_p`
(or a trace on L5 itself) and an L5 population dense enough to matter, not a larger
random projection.

### 8.4 No resistance to distraction

At the default configuration A never survives B (0/30 on all three seeds). The trace
boosts whatever was active at the end of the last input, and the inhibition silences the
rest, so the newest item always wins. This is the opposite failure of `local_adapt2`
(which cannot be overwritten), not the classical working-memory property (maintain A
through B). With weaker inhibition or L5 feedback A survives 4–6/30, but then the state is
a mixture (`ovA` ≈ 0.5), not a maintained A.

### 8.5 A collapse case in the distractor at some settings

At code 120 (seed 0) one of 30 pairs (A=zeta, B=epsilon) ends with 14 cells and best
overlap 0.10 — the union of the old and new codes drives enough inhibition to crush the
state. At code 150 (all three seeds) and code 100 there are 0/30 unreadable pairs; at code
200 there are 6/30. The summary line counts argmax winners, so the "unreadable" column was
added to keep this visible.

---

## 9. What would be needed to fix the remaining issues

1. **A stationary hold at presentation 100–400.** The single scalar that matters is the
   inhibitory gain relative to the code density. Lowering `W_ie` (`inh=0.5`) at code 150
   gives `rel_hold = −0.06`, a perfectly stationary hold (156 cells for 30 steps,
   `ov(10,30)` 0.97), SANITY 6/6 with overlap 1.00 and perturbation 1.00 at every fraction —
   **but** the distractor becomes a mixture (A survives 4/30, held 222–251, `ovA` 0.46).
   One gain cannot both keep `rel_hold` small and give the pool enough authority to
   silence the previous item. The two roles have to be separated, e.g. a pool with a
   slower/adapting threshold, or a separate gating signal.
2. **A pattern attractor rather than trace reconstruction.** Add recurrent excitation
   aligned with each item's assembly (as `basin.py` proposes with `W = sum_k c_k c_k^T`), so
   that the fast state itself has a basin. That would also let the state be perturbed
   without touching `m`.
3. **The L5 population in the loop without the random `W_p`.** Learn `W_p` (or put a trace
   on L5 as well) instead of using `local_adapt2`'s untrained block-random matrix; §8.3
   shows the random version is what corrupts the distractor.
4. **Resistance to distraction**, if it is wanted, needs a write gate (e.g. only update
   `m_ref` when the input is novel relative to the held state) — the current mechanism has
   no way to distinguish "new item" from "more of the same".

---

## 10. Reproduction

```
python wm_port.py 0 150          # headline: seed 0, default configuration
python wm_port.py 1 150          # seed 1
python wm_port.py 2 150          # seed 2
python wm_port.py 0 120          # more stationary hold, still in the 100-400 band
python wm_port.py 0 200          # too dense: the hold decays inside the measurement window
python wm_port.py 0 150 1.5 400 0 0.5   # weaker inhibition: stationary hold, leakier distractor
python wm_port.py 0 150 1.5 400 1.0 1.0 180   # L5 feedback on, denser L5: L5 flip still inert
python wm_port.py 0 150 1.5 400 2.0 1.0 180   # feedback too strong: layer saturates
python wm_diag_ref.py 0          # instrumentation of walk_rel2's loop
python wm_diag_dense.py 0        # dense drive vs 0/1 code, same loop (the central diagnostic)
python wm_demo.py 0              # the failed previous port, for comparison
python wm_tests.py 0             # the local_adapt2 baseline (0.681/0.248/0.229)
```

Files created for this task (no existing file was modified):
`wm_port.py`, `wm_diag_ref.py`, `wm_diag_dense.py`, `wm_proto.py` (first exploration),
and the logs `wm_port_final_seed{0,1,2}.log`, `wm_port_variant_*.log`,
`wm_port_seed*.log`, `wm_tests_seed0.log`, `wm_diag_dense.log`.

## Appendix A — configurations swept (seed 0 unless stated)

| config | presentation | held (10) | stability ov(10,20)/(10,30) | rel_hold | SANITY | PERTURB x23 5/20/50 (ov_ctrl) | DISTRACTOR |
|---|---|---|---|---|---|---|---|
| code 100 | 93–122 | 93–122 | 0.83 / 0.37 (seed 0); 1.00 / 0.93 (seed 1) | −0.080 | 6/6 | 0.99/0.99/0.98 (1.00/1.00/0.98) | B 30/30 (seed 0); B 28/30, A 2/30 (seed 1) |
| code 120 | 95–126 | 95–126 | 0.84 / 0.35 | −0.107 | 6/6 | 0.98/0.98/0.96 (1.00/0.99/0.97) | B 30/30, 1/30 unreadable |
| **code 150** | **126–179** | **124–146** | **0.47 / 0.25** | **−0.143** | **6/6** | **0.84/0.82/0.81 (0.99/0.96/0.91)** | **B 30/30, 0 unreadable** |
| code 200 | 181–261 | 109–155 | 0.43 / 0.20 | −0.160 | 6/6 | 0.58/0.61/0.61 (0.86/0.80/0.77) — equals the drift | B 30/30, 6/30 unreadable |
| code 150, inh 0.5 | 126–179 | = presentation | 1.00 / 0.97 | −0.062 | 6/6 | 1.00/1.00/1.00 | B 26/30, A 4/30, mixture |
| code 150, fb 1.0, L5 180 | 160–180 | 157–178 | 0.83 / 0.51 | −0.075 | 6/6 | 0.98/0.97/0.95 (1.00/0.99/0.96); L5 flip ov_imm 0.997 | B 26/30, A 4/30, mixture |
| code 150, fb 2.0 | 376–839 | 537–696 | — | — | 6/6 (0.65–0.85) | — | — (layer saturated) |

## Appendix B — exact baseline probes (`wm_tests.py 0`, for comparison)

```
per-item theta: 0.25 0.28 0.22 0.22 0.22 0.28
presentation sizes: 128 127 192 384 256 311
SANITY stayed: 6/6
DISTRACTOR A survived in 3/3 cases (overlap 1.00, 0.99, 0.75)
PERTURB: 5% -> 0.681 | 20% -> 0.248 | 50% -> 0.229
```

Environment: Python 3.13, torch 2.7.1+cu118, CUDA available; everything runs in
seconds on one GPU. Every run is seeded (`numpy.default_rng(seed)` for the sensory stack,
`seed + 7` for the loop matrices as in `walk_rel2`, and a per-(item, fraction, rule) RNG for
the flips), and two identical runs produced identical measurement lines.
