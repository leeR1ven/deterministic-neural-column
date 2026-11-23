"""Step 6: capacity law -- how many transitions fit in an output layer?

Documents (SPEC 24/45-48):
    code length ~40-70 cells
    transitions <~ 2 * output channels / code length
    36 words fail on a 280-channel output but work on a 1024-channel output

The documents' 36-word diagnosis (SPEC 45) was made on the *discrete* drive
("the 10 strongest driven cells belong to the target").  The final readout is
unthresholded cosine template matching (SPEC 41), which is far more tolerant.
Both are measured here.
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import SparseEncoder, chain_words, word_pattern, overlap_matrix, jaccard
from dnc.loop import WorkingMemory
from dnc.readout import train_shared, drive, template_match, fidelity

READ_HOLD = 300


def random_codes(n, width, code_len, seed=0):
    rng = np.random.default_rng(seed + 2024)
    codes = np.zeros((n, width))
    for i in range(n):
        idx = rng.choice(width, size=code_len, replace=False)
        codes[i, idx] = 1.0
    return [codes[i] for i in range(n)]


def run(n_words, width, code_len, seed=0, verbose=False):
    words = chain_words(n_words)
    pats = [word_pattern(w) for w in words]
    enc = SparseEncoder(280, 1024, 80, 70, 0.90, seed=seed)
    enc.fit(pats)
    drives = [enc.encode(p, i)[1] for i, p in enumerate(pats)]
    wm = WorkingMemory(enc, tau_memory=4000.0, theta5_q=0.90, seed=seed)
    scores = []
    for k in range(n_words):
        wm.reset(); wm.present(drives[k], 20); wm.hold(READ_HOLD)
        scores.append(wm.l5_scores())
    wm.calibrate_l5(scores, q=0.90)
    l5 = [np.asarray(s > wm.theta5, dtype=float) for s in scores]
    targets = random_codes(n_words, width, code_len, seed=seed)
    tg = [targets[(k + 1) % n_words] for k in range(n_words)]
    W = train_shared(l5, tg, budget=0.05)

    ok, cos, top10, topn = 0, [], [], []
    for k in range(n_words):
        d = drive(W, l5[k])
        idx, _ = template_match(d, tg)
        ok += int(idx == k)
        f = fidelity(d, tg[k])
        cos.append(f["cosine"])
        t = tg[k] > 0
        top10_idx = np.argsort(d)[-10:]
        top10.append(float(t[top10_idx].mean()))
        topn.append(f["top_precision"])
    # L5 state separability
    ov = overlap_matrix(np.stack(l5))
    iu = np.triu_indices(n_words, 1)
    nn = sum(int(np.argmax([jaccard(l5[k], l5[j]) for j in range(n_words)]) == k)
             for k in range(n_words))
    if verbose:
        print(f"L5 density {np.mean([x.sum() for x in l5]):.0f}/512, "
              f"pairwise {ov[iu].mean():.3f}, nearest-identity {nn}/{n_words}")
    return dict(ok=ok, cos=float(np.mean(cos)), top10=float(np.mean(top10)),
                topn=float(np.mean(topn)), mf=float(n_words * code_len / width),
                l5_overlap=float(ov[iu].mean()), l5_identity=nn)


if __name__ == "__main__":
    print(f"{'words':>5} {'width':>6} {'len':>4} {'m*f':>6} {'readout':>9} "
          f"{'cosine':>7} {'top10':>6} {'topn':>6}")
    for n in (12, 36, 80):
        for width in (280, 1024, 2048):
            r = run(n, width, 70)
            print(f"{n:5d} {width:6d} {70:4d} {r['mf']:6.2f} {r['ok']:5d}/{n:<3d} "
                  f"{r['cos']:7.3f} {r['top10']:6.2f} {r['topn']:6.2f}")
    print()
    for n in (36, 80):
        r = run(n, 2048, 70, verbose=True)
        print(f"{n} words / 2048: L5 pairwise {r['l5_overlap']:.3f}, identity {r['l5_identity']}/{n}")
    print()
    for cl in (20, 40, 70):
        r = run(80, 2048, cl)
        print(f"80 words, width 2048, code_len {cl}: {r['ok']}/80  cosine {r['cos']:.3f} "
              f"top10 {r['top10']:.2f}  m*f {r['mf']:.2f}")
