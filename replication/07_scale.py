"""Step 7: chain at scale (12 / 36 / 80 words) and across seeds."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import SparseEncoder, chain_words, word_pattern, jaccard, overlap_matrix
from dnc.loop import WorkingMemory
from dnc.readout import train_shared, drive, template_match

READ_HOLD = 300


def build(words, seed=0):
    pats = [word_pattern(w) for w in words]
    enc = SparseEncoder(280, 1024, 80, 70, 0.90, seed=seed)
    enc.fit(pats)
    drives = [enc.encode(p, i)[1] for i, p in enumerate(pats)]
    wm = WorkingMemory(enc, tau_memory=4000.0, theta5_q=0.90, seed=seed)
    scores, l23 = [], []
    for k in range(len(words)):
        wm.reset(); wm.present(drives[k], 20); wm.hold(READ_HOLD)
        scores.append(wm.l5_scores()); l23.append(wm.s.copy())
    wm.calibrate_l5(scores, q=0.90)
    l5 = [np.asarray(s > wm.theta5, dtype=float) for s in scores]
    return enc, pats, drives, wm, np.stack(l23), l5


def run(n_words, seed=0, steps=12, target="raw"):
    words = chain_words(n_words)
    enc, pats, drives, wm, l23, l5 = build(words, seed)
    if target == "raw":
        tg = [pats[(k + 1) % n_words] for k in range(n_words)]
        targets_full = pats
    else:  # concept-level: reproduce the next word's 1024-cell L2/3 code
        tg = [l23[(k + 1) % n_words] for k in range(n_words)]
        targets_full = [l23[k] for k in range(n_words)]
    W = train_shared(l5, tg, budget=0.05)
    # chain: recognise next code by cosine, re-inject that code
    cur, ok, seq = 0, 0, []
    for _ in range(steps):
        wm.reset(); wm.present(drives[cur], 20); wm.hold(READ_HOLD)
        s5 = wm.read_l5()
        idx, _ = template_match(drive(W, s5), targets_full)
        ok += int(idx == (cur + 1) % n_words)
        seq.append(words[idx])
        cur = idx
    ident = sum(int(np.argmax([jaccard(l23[k], l23[j]) for j in range(n_words)]) == k)
                for k in range(n_words))
    ov = overlap_matrix(l23); iu = np.triu_indices(n_words, 1)
    return dict(n=n_words, seed=seed, ok=ok, steps=steps, ident=ident,
                overlap=float(ov[iu].mean()), seq=seq,
                l5_identity=sum(int(np.argmax([jaccard(l5[k], l5[j]) for j in range(n_words)]) == k)
                                for k in range(n_words)))


if __name__ == "__main__":
    for n in (12, 36, 80):
        for tgt in ("raw", "code"):
            r = run(n, seed=0, steps=12, target=tgt)
            print(f"{n:3d} words, target={tgt:4s}: chain {r['ok']}/12 | "
                  f"held identity {r['ident']}/{n} | L5 identity {r['l5_identity']}/{n} | "
                  f"state overlap {r['overlap']:.4f}")
    print("\nseed sweep, 12 words, raw target, 12 steps:")
    for seed in range(6):
        r = run(12, seed=seed, steps=12, target="raw")
        print(f"  seed {seed}: chain {r['ok']}/12, held identity {r['ident']}/12, "
              f"overlap {r['overlap']:.4f}")
