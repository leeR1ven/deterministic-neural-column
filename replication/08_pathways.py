"""Step 8: the three output pathways with the documents' own constants."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import SparseEncoder, chain_words, word_pattern
from dnc.loop import WorkingMemory
from dnc.pathways import ThreePathways, MOTOR, N_BG

words = chain_words(12)   # L5 threshold calibration needs a vocabulary
pats = [word_pattern(w) for w in words]
enc = SparseEncoder(280, 1024, 80, 70, 0.90, seed=0)
enc.fit(pats)
drives = [enc.encode(p, i)[1] for i, p in enumerate(pats)]
wm = WorkingMemory(enc, tau_memory=4000.0, theta5_q=0.90, seed=0)
scores, states = [], []
for k in range(12):
    wm.reset(); wm.present(drives[k], 20); wm.hold(300)
    scores.append(wm.l5_scores()); states.append(wm.s.copy())
wm.calibrate_l5(scores, q=0.90)
l5 = [np.asarray(s > wm.theta5, dtype=float) for s in scores]
targets = [pats[k] for k in range(3)]     # each speaking item reproduces its word code
circ = ThreePathways(targets, l5)

scen = [
    ("THINK (no go, no speaking item)", dict(item=None, go=False, itch=False, hold=False)),
    ("THINK-WITH-ITEM (no go)",          dict(item=0, go=False, itch=False, hold=False)),
    ("ACT (go + speaking item)",         dict(item=0, go=True, itch=False, hold=False)),
    ("ACT (go + speaking item 2)",       dict(item=2, go=True, itch=False, hold=False)),
    ("INSTINCT alone (itch, no BG)",     dict(item=None, go=False, itch=True, hold=False)),
    ("PUBLIC SUPPRESS (itch + hold)",    dict(item=None, go=False, itch=True, hold=True)),
    ("HOLD (go + item + hold)",          dict(item=0, go=True, itch=False, hold=True)),
]
print(f"motor cells {MOTOR} (3 x 280), basal ganglia {N_BG}")
for name, kw in scen:
    s5 = l5[kw["item"]] if kw["item"] is not None else np.zeros_like(l5[0])
    r = circ.run(s5=s5, **kw)
    extra = ""
    if "match_jaccard" in r:
        extra = (f" | match Jaccard {r['match_jaccard']:.3f}, "
                 f"precision {r['precision']:.3f}, recall {r['recall']:.3f}")
    print(f"{name:34s} motor {r['n_motor']:4d}  bg {r['bg_active']:3d}{extra}")
