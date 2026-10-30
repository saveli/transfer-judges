"""LODO (leave-one-dataset-out) prediction of expert overall impression.

Reads ml/features/<dataset>.parquet, predicts the 1-5 target.

Two evaluation settings:
  LODO  - train on all-but-one dataset (pooled), test on the held-out one. The transfer
          / generalisation question.
  CV    - 5-fold within each dataset. In-domain reference ceiling.

Every model is a leakage-safe sklearn Pipeline:
  VarianceThreshold (drop the ~80 constant opensmile floors) -> StandardScaler
  -> feature selection / DR -> regressor.
Selection / DR is its own ablation axis (required: ~590 features vs ~50-55 sessions).

Ablation axes: feature set (functionals vs mean-only) x selector x model.
Baseline = predict the train-set mean (MAE/RMSE a model must beat).

Output: ml/results/lodo_phase3.csv + console summary.
"""
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import (VarianceThreshold, SelectKBest, f_regression,
                                       mutual_info_regression, RFE)
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.model_selection import KFold
import mord

import dsconfig as ds_cfg


class OrdinalAT:
    """mord all-threshold ordinal logistic — target is an ordered Likert, so model it as such.
    Fits on the observed target levels and predicts one of them back. The levels are not
    assumed integer: the de-rounded teacher target (see dsconfig) carries 2.33 and 3.67, and
    rounding those to int would re-collapse them onto the era-A 2/4 the repair separated."""
    def __init__(self, alpha=1.0):
        self.m = mord.LogisticAT(alpha=alpha)

    def fit(self, X, y):
        y = np.asarray(y, float)
        self.classes_ = np.unique(y)                     # mord needs consecutive 0..K-1; levels have gaps
        self.m.fit(X, np.searchsorted(self.classes_, y))
        return self

    def predict(self, X):
        idx = np.clip(self.m.predict(X).astype(int), 0, len(self.classes_) - 1)
        return self.classes_[idx].astype(float)

warnings.filterwarnings('ignore')
import os
FEAT = Path(__file__).resolve().parent / os.environ.get('KT_FEAT_DIR', 'features')
OUT = Path(__file__).resolve().parent / 'results'
SUFFIX = ('_std', '_min', '_max', '_median')  # functional suffixes beyond _mean
# LING_CORE analog (consensus 4-subset of the shallow linguistic block)
SL4 = ('sl_words_per_min', 'sl_question_ratio', 'sl_speech_ratio', 'sl_segdur_mean')


LLM = Path(__file__).resolve().parent / 'llm'
LLM2 = Path(__file__).resolve().parent / 'llm_dialog'
LLM3 = Path(__file__).resolve().parent / 'llm_dialog_think'
LLM4 = Path(__file__).resolve().parent / 'llm_theory'
LLM5 = Path(__file__).resolve().parent / 'llm_phases'
LLM6 = Path(__file__).resolve().parent / 'llm_theory_v2'
LLM7 = Path(__file__).resolve().parent / 'llm_theory_qwen36'    # phase 10c bake-off passers
LLM8 = Path(__file__).resolve().parent / 'llm_theory_phi4'
LLM9 = Path(__file__).resolve().parent / 'llm_theory_granite30'
LLM9F = Path(__file__).resolve().parent / 'llm_theory_granite30_fix'
LLMB = Path(__file__).resolve().parent / 'llm_theory_bare'   # anchoring control (review C3)  # vacc AU50-repaired
LLM7M = Path(__file__).resolve().parent / 'llm_theory_qwen36_mean'    # 3-seed mean (de-noised)
# AU50-repaired vaccines (16.1% of diarised segments were role-misattributed; repaired
# from the human per-role speech GT). ticks/teacher symlink to the original caches.
LLM2F = Path(__file__).resolve().parent / 'llm_dialog_fix'
LLM3F = Path(__file__).resolve().parent / 'llm_dialog_think_fix'
LLM4F = Path(__file__).resolve().parent / 'llm_theory_fix'
LLM6F = Path(__file__).resolve().parent / 'llm_theory_v2_fix'
EMBED = Path(__file__).resolve().parent / 'embed'
EMBED_CLS = Path(__file__).resolve().parent / 'embed_cls'      # gbert CLS (weak end)
EMBED_CLS_XLMR = Path(__file__).resolve().parent / 'embed_cls_xlmr'  # xlm-r CLS (untrained floor)
EMBED_NOMIC = Path(__file__).resolve().parent / 'embed_nomic'  # nomic v2-moe (strong end)
ZPREFIXES = ('cmp_', 'dlt_', 'llmb_', 'llm_', 'llm2_', 'llm3_', 'llm4_', 'llm5_', 'llm6_', 'llm7_', 'llm8_', 'llm9_', 'llm9f_', 'llm7m_', 'llm2f_', 'llm3f_', 'llm4f_', 'llm6f_', 'pro_', 'dpr_', 'syn_', 'mut_', 'tt_', 'sl_')  # low-dim blocks that get a per-domain z copy


def _llm_frame(dataset, root=None, prefix='llm'):
    import json
    d = (root or LLM) / dataset
    if not d.exists():
        return None
    rows = {p.stem: {f'{prefix}_{k}': v for k, v in json.loads(p.read_text()).items()} for p in d.glob('*.json')}
    return pd.DataFrame.from_dict(rows, orient='index') if rows else None


def load_matrices():
    mats = {}
    for d in ds_cfg.READY:
        m = pd.read_parquet(FEAT / f'{d}.parquet')
        # the parquet caches carry the target as it stood when features were built; re-read
        # it so target repairs (teacher 4pt/5pt de-rounding) do not need a DB feature rebuild
        tgt = ds_cfg.load_target(d)
        m['target'] = [tgt.get(s, np.nan) for s in m.index]
        for root, prefix in [(LLM, 'llm'), (LLM2, 'llm2'), (LLM3, 'llm3'), (LLM4, 'llm4'), (LLM5, 'llm5'),
                             (LLM6, 'llm6'), (LLM7, 'llm7'), (LLM8, 'llm8'), (LLM9, 'llm9'),
                             (LLM9F, 'llm9f'), (LLM7M, 'llm7m'), (LLMB, 'llmb'),
                             (LLM2F, 'llm2f'), (LLM3F, 'llm3f'), (LLM4F, 'llm4f'), (LLM6F, 'llm6f')]:
            llm = _llm_frame(d, root, prefix)
            if llm is not None:
                m = m.join(llm)                           # llm_* / llm2_* construct scores
        for embdir in (EMBED, EMBED_CLS, EMBED_CLS_XLMR, EMBED_NOMIC):
            p = embdir / f'{d}.parquet'
            if p.exists():
                m = m.join(pd.read_parquet(p))            # emb_*/embc_*/embn_* embeddings
        # per-domain z copies of the curated low-dim blocks (z_<col>): removes cross-domain
        # scale/offset drift (LLM anchor drift, actor/sex prosody register) using only the
        # domain's own feature marginals — no target involved, LODO-safe.
        for c in [c for c in m.columns if c.startswith(ZPREFIXES)]:
            sd = m[c].std()
            m[f'z_{c}'] = (m[c] - m[c].mean()) / sd if sd and np.isfinite(sd) else 0.0
        mats[d] = m
    return mats


def feature_cols(df, featureset):
    cols = [c for c in df.columns if c != 'target']
    emo = [c for c in cols if c.startswith('emow2v_')]
    base = [c for c in cols if not c.startswith('emow2v_')]
    cmp_ = [c for c in cols if c.startswith('cmp_')]
    dlt = [c for c in cols if c.startswith('dlt_')]
    llm = [c for c in cols if c.startswith('llm_')]
    llm2 = [c for c in cols if c.startswith('llm2_')]
    llm3 = [c for c in cols if c.startswith('llm3_')]
    llm4 = [c for c in cols if c.startswith('llm4_')]
    llm5 = [c for c in cols if c.startswith('llm5_')]
    llm6 = [c for c in cols if c.startswith('llm6_')]
    llm7 = [c for c in cols if c.startswith('llm7_')]
    llm8 = [c for c in cols if c.startswith('llm8_')]
    llm9 = [c for c in cols if c.startswith('llm9_')]
    llm9f = [c for c in cols if c.startswith('llm9f_')]
    llm7m = [c for c in cols if c.startswith('llm7m_')]
    llm2f = [c for c in cols if c.startswith('llm2f_')]
    llm3f = [c for c in cols if c.startswith('llm3f_')]
    llm4f = [c for c in cols if c.startswith('llm4f_')]
    llm6f = [c for c in cols if c.startswith('llm6f_')]
    sl = [c for c in cols if c.startswith('sl_')]
    pro = [c for c in cols if c.startswith('pro_')]
    dpr = [c for c in cols if c.startswith('dpr_')]
    syn = [c for c in cols if c.startswith('syn_')]
    mut = [c for c in cols if c.startswith('mut_')]
    tt = [c for c in cols if c.startswith('tt_')]
    emb = [c for c in cols if c.startswith('emb_')]
    embc = [c for c in cols if c.startswith('embc_')]
    embx = [c for c in cols if c.startswith('embx_')]
    embn = [c for c in cols if c.startswith('embn_')]
    z = lambda block: [f'z_{c}' for c in block if f'z_{c}' in cols]
    nv = z(dlt) + z(syn) + z(mut) + z(tt)                # strictly nonverbal, dyadic
    # every embedding ladder is excluded from the wide block: `functionals` is the
    # openSMILE/emoW2V dimensionality baseline, and the phase-10 embed caches
    # (embc_/embx_/embn_) would otherwise silently add 2304 columns to it.
    wide = [c for c in base if not c.startswith(ZPREFIXES + ('emb_', 'embc_', 'embx_',
                                                             'embn_', 'z_'))]
    sets = {
        'mean_only': [c for c in wide if not c.endswith(SUFFIX)],
        'functionals': wide,                             # full wide set + emoW2V
        'compact': cmp_,                                 # 12 interpretable scalars
        'compact+deltas': cmp_ + dlt,                    # + dyadic interaction
        'emow2v_only': emo,
        'llm_only': llm,                                 # LLM construct scores (5)
        'compact+llm': cmp_ + llm,                       # best non-text + text
        'lowdim_all': cmp_ + dlt + llm,                  # all curated low-dim (phase-3 winner)
        'embed_only': emb,                               # sentence embeddings (PCA'd)
        # ---- phase 4 additions -------------------------------------------------
        'llm2_only': llm2,                               # dialogue-context prompt, gemma4
        'lowdim_all2': cmp_ + dlt + llm2,                # phase-3 winner with dialog LLM
        'syn_only': syn,                                 # dyadic synchrony (f0/loudness)
        'lowdim_syn': cmp_ + dlt + llm + syn,            # winner + synchrony
        'lowdim_all_z': z(cmp_) + z(dlt) + z(llm),       # per-domain calibrated winner
        'lowdim_all2_z': z(cmp_) + z(dlt) + z(llm2),     # calibrated + dialog LLM
        'lowdim_syn2_z': z(cmp_) + z(dlt) + z(llm2) + z(syn),  # everything new
        # ---- phase 5: think=True dialogue constructs ---------------------------
        'llm3_only': llm3,                               # dialog prompt + thinking mode
        'lowdim_all3': cmp_ + dlt + llm3,
        'lowdim_all3_z': z(cmp_) + z(dlt) + z(llm3),
        'llm2+llm3': llm2 + llm3,                        # nothink + think ensemble
        'lowdim_all23_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3),
        # ---- phase 6: dyadic nonverbal + turn-taking (venue-gate ablation) ------
        'tt_only': tt,                                   # turn-taking / response latency
        'mut_only': mut,                                 # mutual gaze / mutual smile
        'llm23_z': z(llm2) + z(llm3),                    # calibrated LLM-only baseline for the gate
        'lowdim_all23tt_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(tt),
        'lowdim_all23mut_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(mut),
        'lowdim_all23ttmut_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(tt) + z(mut),
        'nonverbal_dyadic_z': nv,                        # the gate's nonverbal block
        # ---- phase 7: theory-driven LLM constructs (BGR + OSCE anchors) ---------
        'llm4_only': llm4,
        'llmb_only': [c for c in cols if c.startswith('llmb_')],
        'llm24_z': z(llm2) + z(llm4),
        'llm234_z': z(llm2) + z(llm3) + z(llm4),
        'lowdim_all234_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4),
        # ---- phase 8: phase/technique judge + emotiefflib/emow2v features ---
        'llm5_only': llm5,
        'llm45_z': z(llm4) + z(llm5),                    # both theory judges
        'llm2345_z': z(llm2) + z(llm3) + z(llm4) + z(llm5),
        'lowdim_all2345_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4) + z(llm5),
        # ---- phase 9: anti-band theory judge (v2) --------------------------------
        'llm6_only': llm6,
        'llm236_z': z(llm2) + z(llm3) + z(llm6),
        'llm2346_z': z(llm2) + z(llm3) + z(llm4) + z(llm6),
        'lowdim_all236_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm6),
        'lowdim_all2346_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4) + z(llm6),
        # ---- phase 10c: bake-off round-3 judges (theory-v2 prompt on new models;
        # screened label-free, so ensemble additions carry no selection caveat) ------
        'llm7_only': llm7,                               # qwen3.6-27B (clean screen pass)
        'llm8_only': llm8,                               # phi4-14B (marginal pass, band-compressed)
        'llm9_only': llm9,                               # granite4.1-30B (pass, generous means)
        'llm2349_z': z(llm2) + z(llm3) + z(llm4) + z(llm9),
        'lowdim_all2349_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4) + z(llm9),
        'llm7m_only': llm7m,
        'llm23479m_z': z(llm2) + z(llm3) + z(llm4) + z(llm7m) + z(llm9),
        'llm2349f_z': z(llm2) + z(llm3) + z(llm4) + z(llm9f),
        'lowdim_all2349f_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4) + z(llm9f),
        # ---- phase 10d: ALL-CORRECTED candidates (c = vacc AU50 repair on every
        # gemma3/qwen3 judge + seed-mean qwen3.6 + repaired granite). Corrections are
        # uniform data-quality fixes; nested selection chooses membership only.
        'llm234c_z': z(llm2f) + z(llm3f) + z(llm4f),
        'llm2346c_z': z(llm2f) + z(llm3f) + z(llm4f) + z(llm6f),
        'llm2349c_z': z(llm2f) + z(llm3f) + z(llm4f) + z(llm9f),
        'llm23479c_z': z(llm2f) + z(llm3f) + z(llm4f) + z(llm7m) + z(llm9f),
        'llm234789c_z': z(llm2f) + z(llm3f) + z(llm4f) + z(llm7m) + z(llm8) + z(llm9f),
        'lowdim_all234c_z': z(cmp_) + z(dlt) + z(llm2f) + z(llm3f) + z(llm4f),
        'lowdim_all2346c_z': z(cmp_) + z(dlt) + z(llm2f) + z(llm3f) + z(llm4f) + z(llm6f),
        'lowdim_all2349c_z': z(cmp_) + z(dlt) + z(llm2f) + z(llm3f) + z(llm4f) + z(llm9f),
        'lowdim_all23479c_z': z(cmp_) + z(dlt) + z(llm2f) + z(llm3f) + z(llm4f) + z(llm7m) + z(llm9f),
        'lowdim_all234789c_z': z(cmp_) + z(dlt) + z(llm2f) + z(llm3f) + z(llm4f) + z(llm7m) + z(llm8) + z(llm9f),
        'llm2347_z': z(llm2) + z(llm3) + z(llm4) + z(llm7),
        'llm23479_z': z(llm2) + z(llm3) + z(llm4) + z(llm7) + z(llm9),
        'llm234789_z': z(llm2) + z(llm3) + z(llm4) + z(llm7) + z(llm8) + z(llm9),
        'lowdim_all2347_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4) + z(llm7),
        'lowdim_all23479_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4) + z(llm7) + z(llm9),
        'lowdim_all234789_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4) + z(llm7) + z(llm8) + z(llm9),
        # ---- phase 10: shallow linguistic block (LING port, 7 + core 4) ---
        'shallow_ling7': sl,
        'shallow_ling4': [c for c in sl if c in SL4],
        'shallow_ling7_z': z(sl),
        'shallow_ling4_z': z([c for c in sl if c in SL4]),
        'lowdim_all234sl_z': z(cmp_) + z(dlt) + z(llm2) + z(llm3) + z(llm4) + z(sl),
        # ---- phase 11: prosody-v2 (semitone F0, spread/slope, jitter/shimmer/HNR) --
        'prosody2_only': pro,
        'prosody2_dyadic_z': z(pro) + z(dpr),
        'nonverbal_dyadic2_z': z(dlt) + z(syn) + z(mut) + z(tt) + z(pro) + z(dpr),
        'lowdim_all23479c_pro_z': z(cmp_) + z(dlt) + z(pro) + z(dpr) +
                                  z(llm2f) + z(llm3f) + z(llm4f) + z(llm7m) + z(llm9f),
        # ---- 07-28: strictly-nonverbal marginal. The lowdim_* stacks add compact
        # features too, and `compact` carries transcript sentiment and words/sec, so
        # their marginal over the judges is a mixed multimodal one. These pair each
        # judge ensemble with the nonverbal-dyadic block ALONE (affect deltas, mutual
        # gaze/smile, turn-taking, synchrony), which is what the modality-attribution
        # claim is actually about.
        'llm234c_nv_z': z(llm2f) + z(llm3f) + z(llm4f) + nv,
        'llm2346c_nv_z': z(llm2f) + z(llm3f) + z(llm4f) + z(llm6f) + nv,
        'llm2349c_nv_z': z(llm2f) + z(llm3f) + z(llm4f) + z(llm9f) + nv,
        'llm23479c_nv_z': z(llm2f) + z(llm3f) + z(llm4f) + z(llm7m) + z(llm9f) + nv,
        'llm234789c_nv_z': (z(llm2f) + z(llm3f) + z(llm4f) + z(llm7m) + z(llm8)
                            + z(llm9f) + nv),
        'llm23479c_nv2_z': (z(llm2f) + z(llm3f) + z(llm4f) + z(llm7m) + z(llm9f)
                            + nv + z(pro) + z(dpr)),
        # ---- phase 10: embedder ladder (CLS -> MiniLM -> nomic v2-moe) -------------
        'embed_cls_only': embc,
        'embed_cls_xlmr_only': embx,
        'embed_nomic_only': embn,
    }
    if featureset not in sets:
        raise ValueError(featureset)
    return sets[featureset]


def n_effective(X):
    """Columns that survive VarianceThreshold (non-constant, not all-NaN) — PCA k must
    respect the post-VT width, not the nominal one (degenerate blocks collapse)."""
    X = np.asarray(X, float)
    ok = 0
    for j in range(X.shape[1]):
        col = X[:, j][np.isfinite(X[:, j])]
        ok += int(col.size > 0 and col.std() > 0)
    return max(1, ok)


def _mi_score(X, y):
    return mutual_info_regression(X, y, random_state=0)   # deterministic MI (kNN estimator)


def make_pipeline(selector, model, k, n_features):
    k = min(k, n_features)                               # clamp for narrow feature sets (emow2v_only)
    steps = [('imp', SimpleImputer(strategy='median')), ('vt', VarianceThreshold()), ('sc', StandardScaler())]
    if selector == 'kbest':
        steps.append(('sel', SelectKBest(f_regression, k=k)))
    elif selector == 'mi':
        steps.append(('sel', SelectKBest(_mi_score, k=k)))     # information gain (nonlinear)
    elif selector == 'rfe':
        steps.append(('sel', RFE(Ridge(alpha=10.0), n_features_to_select=k, step=0.2)))  # statistical elimination
    elif selector == 'pca':
        # seeded: svd_solver='auto' switches to randomized SVD once the block is wide
        # (functionals 563, mean_only 131 after the embedding exclusion above), and that
        # path is unseeded by default
        steps.append(('sel', PCA(n_components=k, random_state=0)))
    steps.append(('model', model))
    return Pipeline(steps)


MODELS = {
    'ridge': lambda: Ridge(alpha=10.0),
    'elasticnet': lambda: ElasticNet(alpha=0.5, l1_ratio=0.5, max_iter=5000),
    'svr': lambda: SVR(kernel='rbf', C=1.0, gamma='scale'),
    'ordinal': lambda: OrdinalAT(alpha=1.0),
    'gbm': lambda: GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05, random_state=0),
}
# phase 10: extended small-k/low-dim grid (prior in-house experience: consensus picks were
# k=1-5 and PCA inner-CV chose the grid floor 5 in 20/20 folds — the old {20,50} floor
# likely overshoots at n≈195). Pre-committed axis, no post-hoc selection.
KGRID = [1, 2, 3, 5, 8, 12, 20, 50]
SELECTORS = ([('kbest', k) for k in KGRID] + [('mi', k) for k in KGRID] +
             [('pca', k) for k in (2, 3, 5, 7, 10)] + [('rfe', 5), ('rfe', 20)])
FEATURESETS = ['functionals', 'compact', 'compact+deltas', 'emow2v_only', 'mean_only',
               'llm_only', 'compact+llm', 'lowdim_all', 'embed_only',
               'llm2_only', 'lowdim_all2', 'syn_only', 'lowdim_syn',
               'lowdim_all_z', 'lowdim_all2_z', 'lowdim_syn2_z',
               'llm3_only', 'lowdim_all3', 'lowdim_all3_z', 'llm2+llm3', 'lowdim_all23_z',
               'tt_only', 'mut_only', 'llm23_z', 'lowdim_all23tt_z', 'lowdim_all23mut_z',
               'lowdim_all23ttmut_z', 'nonverbal_dyadic_z',
               'llm4_only', 'llm24_z', 'llm234_z', 'lowdim_all234_z',
               'llm5_only', 'llm45_z', 'llm2345_z', 'lowdim_all2345_z',
               'llm6_only', 'llm236_z', 'llm2346_z', 'lowdim_all236_z', 'lowdim_all2346_z']
RESULTS_CSV = 'lodo_phase10.csv'


def metrics(y_true, y_pred, y_train_mean):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    ok = len(set(y_pred)) > 1
    rho = spearmanr(y_true, y_pred).correlation if ok else 0.0
    r = pearsonr(y_true, y_pred)[0] if ok else 0.0
    acc1 = np.mean(np.abs(y_true - y_pred) <= 1)
    base_mae = np.mean(np.abs(y_true - y_train_mean))
    return dict(MAE=mae, RMSE=rmse, spearman=rho, pearson=r, acc_within1=acc1, base_MAE=base_mae)


def run():
    mats = load_matrices()
    rows = []

    for featureset in FEATURESETS:
        if not feature_cols(next(iter(mats.values())), featureset):
            print(f'  [skip {featureset}: no columns — cache not built]')
            continue
        for sel, k in SELECTORS:
            for mname, mfn in MODELS.items():
                # ---- LODO: train on the rest, test on held-out ----
                for test_ds in mats:
                    train = pd.concat([mats[d] for d in mats if d != test_ds])
                    test = mats[test_ds]
                    cols = feature_cols(train, featureset)
                    Xtr, ytr = train[cols].values, train['target'].values
                    Xte, yte = test[cols].values, test['target'].values
                    pipe = make_pipeline(sel, mfn(), k, n_effective(Xtr))
                    pipe.fit(Xtr, ytr)
                    m = metrics(yte, pipe.predict(Xte), ytr.mean())
                    rows.append(dict(setting='LODO', test=test_ds, featureset=featureset,
                                     selector=f'{sel}{k}', model=mname, n_test=len(yte), **m))

                # ---- in-domain 5-fold CV reference ----
                for ds_name, df in mats.items():
                    cols = feature_cols(df, featureset)
                    X, y = df[cols].values, df['target'].values
                    preds = np.zeros_like(y, float)
                    for tr_i, te_i in KFold(5, shuffle=True, random_state=0).split(X):
                        pipe = make_pipeline(sel, mfn(), k, n_effective(X[tr_i]))
                        pipe.fit(X[tr_i], y[tr_i])
                        preds[te_i] = pipe.predict(X[te_i])
                    m = metrics(y, preds, y.mean())
                    rows.append(dict(setting='CV', test=ds_name, featureset=featureset,
                                     selector=f'{sel}{k}', model=mname, n_test=len(y), **m))

    res = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    res.to_csv(OUT / RESULTS_CSV, index=False)

    # ---- console summary: best per setting/featureset by mean MAE across test sets ----
    pd.set_option('display.width', 160, 'display.max_columns', 20)
    print(f'\n{len(res)} runs -> ml/results/{RESULTS_CSV}\n')
    for setting in ['LODO', 'CV']:
        sub = res[res.setting == setting]
        agg = (sub.groupby(['featureset', 'selector', 'model'])
                  .agg(MAE=('MAE', 'mean'), spearman=('spearman', 'mean'), pearson=('pearson', 'mean'),
                       acc1=('acc_within1', 'mean'), base=('base_MAE', 'mean'))
                  .reset_index().sort_values('MAE'))
        print(f'=== {setting} (mean over test sets) — top 8 by MAE ===')
        print(agg.head(8).to_string(index=False))
        # best config per featureset by spearman (correlation is the discriminating metric — MAE is saturated)
        best = (agg.sort_values('spearman', ascending=False)
                   .groupby('featureset').first().reset_index()
                   .sort_values('spearman', ascending=False))
        print(f'  -- best per featureset by spearman --')
        print(best[['featureset', 'selector', 'model', 'spearman', 'pearson', 'MAE', 'acc1']].to_string(index=False))
        print(f'  [baseline predict-mean MAE ~ {sub.base_MAE.mean():.3f}]\n')
    return res


if __name__ == '__main__':
    run()
