#!/usr/bin/env python3
"""tools/report.py: per search run (runs/NAME.csv + runs/NAME.log, as written by origin), the patches per hour of at
least 2.5, 3 and 3.5M blocks2, the e-fold of the tail above Y0 (mean excess, error L/sqrt(n)) and from it the expected
time to a patch of at least each X (in brackets: e-fold +- one error).
usage: tools/report.py NAME[:SECONDS] ... [--y0 2.5] [--x 4.9,5.0]   (SECONDS: the run's length, else read from its log)"""
import sys, re, math, numpy as np
args = sys.argv[1:]; Y0 = 2.5; XS = [4.9, 5.0]
if '--y0' in args: i = args.index('--y0'); Y0 = float(args[i + 1]); del args[i:i + 2]
if '--x' in args: i = args.index('--x'); XS = [float(v) for v in args[i + 1].split(',')]; del args[i:i + 2]
def hms(s): return f'{s / 3600:.1f} h' if s >= 3600 else f'{s / 60:.0f} min'
for a in args:
    name, _, secs = a.partition(':'); v = []
    for l in open(f'runs/{name}.csv'):
        f = l.strip().split(',')
        if len(f) >= 4:
            try: a8, a4, ex = float(f[1]), float(f[2]), float(f[3]); v.append(ex if ex > 0 else a4 if a4 > 0 else a8)
            except ValueError: pass
    v = np.array(v); t = float(secs) if secs else 0
    if not secs:
        for l in open(f'runs/{name}.log'):
            m = re.match(r'done: \S+ seeds in ([0-9.]+) s', l) or re.match(r'([0-9.]+) s: ', l)
            if m: t = float(m.group(1))
    rate = 1.0
    for l in open(f'runs/{name}.log'):
        m = re.search(r'\(([0-9.e+]+)/s\)', l)
        if m: rate = float(m.group(1))
    ex = v[v >= Y0] - Y0; n = len(ex); L = ex.mean(); r0 = n / t
    fc = []
    for X in XS:
        ts = [1 / (r0 * math.exp(-(X - Y0) / l)) for l in (L, L + L / math.sqrt(n), L - L / math.sqrt(n))]
        fc.append(f'{X}M {hms(ts[0])} ({hms(ts[1])}-{hms(ts[2])})')
    print(f'{name}: {rate:.3g} seeds/s; per hour: 2.5M+ {3600 * np.sum(v >= 2.5) / t:.0f}, 3M+ {3600 * np.sum(v >= 3) / t:.0f}, 3.5M+ {3600 * np.sum(v >= 3.5) / t:.1f}; e-fold above {Y0} {L:.3f} +- {L / math.sqrt(n):.3f} (n {n}); best {v.max():.3f}; time to ' + ', '.join(fc))
