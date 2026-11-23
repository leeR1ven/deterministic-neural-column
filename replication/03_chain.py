"""Step 3: motor map fidelity and the chain (alpha -> beta -> gamma ...)."""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dnc.core import SparseEncoder, chain_words, word_pattern, jaccard, overlap_matrix, cosine
from dnc.loop import WorkingMemory
from dnc.readout import train_shared, train_blocks, drive, template_match, fidelity

READ_HOLD = 300      # steps between input offset and readout (docs never give this)


def build(words, seed=0, q5=0.95):
    pats = [word_pattern(w) for w in words]
    enc = SparseEncoder(280, 1024, 80, 70, 0.90, seed=seed)
    enc.fit(pats)
    drives = [enc.encode(p, i)[1] for i, p in enumerate(pats)]
    wm = WorkingMemory(enc, tau_memory=4000.0, theta5_q=q5, seed=seed)
    # readout-time L5 scores, then calibrate the threshold on those states
    scores, states = [], []
    for k in range(len(words)):
        wm.reset()
        wm.present(drives[k], steps=20)
        wm.hold(READ_HOLD)
        scores.append(wm.l5_scores())
        states.append(wm.s.copy())
    wm.calibrate_l5(scores, q=q5)
    l5 = [np.asarray(sc > wm.theta5, dtype=float) for sc in scores]
    return enc, drives, wm, np.stack(states), l5


def main():
    words = chain_words(12)
    enc, drives, wm, l23, l5 = build(words)
    pats = [word_pattern(w) for w in words]
    print(f"words: {words}")
    print(f"L2/3 readout density {l23.mean():.3f} ({l23.sum(1).mean():.0f} cells), "
          f"L5 density {np.mean([x.sum() for x in l5]):.0f}/512")
    ov = overlap_matrix(l23); iu = np.triu_indices(12, 1)
    print(f"held-state pairwise Jaccard: {ov[iu].mean():.4f} (max {ov[iu].max():.3f})")
    ident = sum(int(np.argmax([jaccard(l23[k], l23[j]) for j in range(12)]) == k) for k in range(12))
    print(f"held state most similar to itself: {ident}/12")

    # targets: next word's raw 280-channel code, cyclic
    targets = [pats[(k + 1) % 12] for k in range(12)]
    W = train_shared(l5, targets, budget=0.05)
    blocks = train_blocks(l5, targets, budget=0.05)

    print("\n-- map fidelity (drive vs next-word 280-bit code) --")
    fs, fb, fr = [], [], []
    for k in range(12):
        t = targets[k]
        d_shared = drive(W, l5[k])
        d_block = drive(blocks[k], l5[k])
        d_wrong = drive(W, l5[(k + 5) % 12])
        fs.append(fidelity(d_shared, t))
        fb.append(fidelity(d_block, t))
        fr.append(fidelity(d_wrong, t))
    for name, rows in (("shared", fs), ("per-transition block", fb), ("mismatched state", fr)):
        print(f"{name:22s} mass_precision {np.mean([r['mass_precision'] for r in rows]):.3f} "
              f"cosine {np.mean([r['cosine'] for r in rows]):.3f} "
              f"top_jaccard {np.mean([r['top_jaccard'] for r in rows]):.3f}")

    # readout accuracy of the shared map on every word
    acc = 0
    for k in range(12):
        d = drive(W, l5[k])
        idx, sims = template_match(d, pats)
        acc += int(idx == (k + 1) % 12)
    print(f"\nshared-map next-word readout: {acc}/12")
    acc_b = 0
    for k in range(12):
        idx, _ = template_match(drive(blocks[k], l5[k]), pats)
        acc_b += int(idx == (k + 1) % 12)
    print(f"block-map next-word readout (oracle block): {acc_b}/12")

    # ---- chain ------------------------------------------------------------
    def walk(steps=12, use_blocks=False):
        out, ok = [], 0
        cur = 0
        for step in range(steps):
            wm.reset()
            wm.present(drives[cur], steps=20)
            wm.hold(READ_HOLD)
            s5 = wm.read_l5()
            if use_blocks:
                # block selection by maximum total drive (the documents report
                # that this criterion fails; reproduced here)
                d = np.stack([drive(b, s5) for b in blocks])
                bk = int(np.argmax(d.sum(axis=1)))
                d = d[bk]
            else:
                d = drive(W, s5)
            idx, _ = template_match(d, pats)
            out.append(words[idx])
            ok += int(idx == (cur + 1) % 12)
            cur = idx
        return out, ok

    seq, ok = walk(12)
    print(f"\nchain (shared map): {' '.join(seq)}")
    print(f"chain correct transitions: {ok}/12")
    seq_b, ok_b = walk(12, use_blocks=True)
    print(f"chain (blocks, max-drive selection): {' '.join(seq_b)}  correct {ok_b}/12")


if __name__ == "__main__":
    main()
