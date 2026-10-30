"""Spearman rho per held-out domain and the mean over domains, from predictions.csv.

Usage: python evaluate.py [predictions.csv]     (needs only pandas + scipy)
"""
import sys

import pandas as pd
from scipy.stats import spearmanr

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else 'predictions.csv')
for system, s in df.groupby('system', sort=False):
    rhos = {d: spearmanr(g.y, g.pred).correlation for d, g in s.groupby('domain', sort=False)}
    print(f'{system:14s} ' + '  '.join(f'{d} {r:.3f}' for d, r in rhos.items())
          + f'  | mean {sum(rhos.values()) / len(rhos):.3f}  (n={len(s)})')
