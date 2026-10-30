"""Out-of-fold leave-one-domain-out predictions of the two reported systems.

Fixed configuration, nothing tuned: median imputation, variance filter, standardisation,
PCA-10, elastic net (alpha 0.5, l1_ratio 0.5). Each training domain carries its own
per-domain z columns (label-free); the held-out domain's z columns are rebuilt from the
pooled raw mean/sd of the training domains, so nothing of the test domain but its raw
features enters (inductive protocol).

Input: the feature matrices that ml/experiment.py load_matrices() assembles (per-domain
feature parquet under ml/$KT_FEAT_DIR plus the judge caches; see ml/CONFIG.md).
Output: domain,session,system,y,pred with your own session ids. The released
predictions.csv is this output with session ids replaced by opaque per-domain indices.

Usage: python predict.py [--out my_predictions.csv]
"""
import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault('KT_FEAT_DIR', 'features_v2')
sys.path.insert(0, str(Path(__file__).resolve().parent / 'ml'))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import experiment as E

SYSTEMS = {'stack': 'lowdim_all2349c_z',      # four judges + compact nonverbal block (summaries + deltas)
           'language-only': 'llm2349c_z'}    # the four judge ensembles alone
SELECTOR, K, MODEL = 'pca', 10, 'elasticnet'


def inductive(mats, train_domains, test_ds):
    """mats[test_ds] with every z_ column rebuilt from the pooled raw mean/sd of train_domains."""
    tr = pd.concat([mats[d] for d in train_domains])
    out = mats[test_ds].copy()
    for zc in [c for c in out.columns if c.startswith('z_')]:
        c = zc[2:]
        mu, sd = tr[c].mean(), tr[c].std()
        out[zc] = (out[c] - mu) / sd if (sd and np.isfinite(sd)) else 0.0
    return out


def fit_predict(train, test, featureset):
    cols = E.feature_cols(train, featureset)
    pipe = E.make_pipeline(SELECTOR, E.MODELS[MODEL](), K, E.n_effective(train[cols].values))
    pipe.fit(train[cols].values, train['target'].values)
    return pipe.predict(test[cols].values)


def lodo(mats, featureset):
    rows = []
    for t in mats:
        trd = [d for d in mats if d != t]
        test = inductive(mats, trd, t)
        p = fit_predict(pd.concat([mats[d] for d in trd]), test, featureset)
        rows += [dict(domain=t, session=s, y=y, pred=q) for s, y, q in zip(test.index, test['target'], p)]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', type=Path, default=Path('my_predictions.csv'))
    a = ap.parse_args()
    mats = E.load_matrices()
    df = pd.concat([lodo(mats, fs).assign(system=name) for name, fs in SYSTEMS.items()])
    df[['domain', 'session', 'system', 'y', 'pred']].to_csv(a.out, index=False, float_format='%.10g')
    for (name, d), g in df.groupby(['system', 'domain'], sort=False):
        print(f'{name:14s} {d:9s} rho={spearmanr(g.y, g.pred).correlation:.3f}')
    print(f'-> {a.out}')


if __name__ == '__main__':
    main()
