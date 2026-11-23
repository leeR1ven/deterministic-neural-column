"""Automated scaling study (run in the background).

For each (number of words, chain length) and each seed, run the current best model and
record:
    correct   how many transition steps were read correctly
    first_bad the step where it first went wrong (-1 if none)
    sim       mean similarity reported by the readout

Writes one CSV row per run so the study can be resumed/inspected, and prints a summary.

Usage: python sweep_scale.py
"""
import os
import sys
import subprocess
import tempfile

BASE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
SCRIPT = os.path.join(BASE, 'scale_more.py')
OUT = os.path.join(BASE, 'sweep_scale.csv')

CONFIGS = [(12, 8), (24, 12), (36, 16), (40, 20)]
SEEDS = [0, 1, 2, 3]


def run_one(n_items, clen, seed):
    with tempfile.NamedTemporaryFile('w+', suffix='.log', delete=False, encoding='utf-8') as fh:
        path = fh.name
    with open(path, 'w', encoding='utf-8') as fh:
        subprocess.run([PY, '-u', SCRIPT, str(seed), '0.17', '0.05', '0.02',
                        str(n_items), str(clen)], stdout=fh, stderr=subprocess.STDOUT)
    txt = open(path, encoding='utf-8', errors='ignore').read()
    os.unlink(path)
    steps, correct, first_bad, sims = [], 0, -1, []
    for line in txt.splitlines():
        line = line.strip()
        if not line.startswith('step '):
            continue
        steps.append(line)
        try:
            read = line.split('read ')[1].split()[0]
        except IndexError:
            continue
        want = line.split('want ')[-1].strip()
        if 'sim ' in line:
            try:
                sims.append(float(line.split('sim ')[1].split(')')[0]))
            except Exception:
                pass
        if want not in ('-', ''):
            if read == want:
                correct += 1
            elif first_bad < 0:
                first_bad = len(steps) - 1
    mean_sim = sum(sims) / len(sims) if sims else float('nan')
    return correct, first_bad, mean_sim, len(steps)


if not os.path.exists(OUT):
    open(OUT, 'w', encoding='utf-8').write('n_items,chain_len,seed,correct,first_bad,mean_sim,steps\n')

print('words | chain | seed | correct | first_bad | mean_sim', flush=True)
agg = {}
for n_items, clen in CONFIGS:
    for sd in SEEDS:
        try:
            correct, first_bad, sim, nsteps = run_one(n_items, clen, sd)
        except Exception as e:
            print('  %d/%d seed %d failed: %s' % (n_items, clen, sd, e), flush=True)
            continue
        open(OUT, 'a', encoding='utf-8').write('%d,%d,%d,%d,%d,%.3f,%d\n' % (
            n_items, clen, sd, correct, first_bad, sim, nsteps))
        print('  %2d | %2d | %d | %2d/%2d | %3d | %.2f' % (
            n_items, clen, sd, correct, clen - 1, first_bad, sim), flush=True)
        agg.setdefault((n_items, clen), []).append(correct)

print('\nsummary (correct steps per seed, out of chain_len-1):', flush=True)
for k, v in agg.items():
    print('  %2d words / %2d steps : %s  (mean %.1f)' % (
        k[0], k[1] - 1, v, sum(v) / len(v)), flush=True)
