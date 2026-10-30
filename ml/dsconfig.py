"""Per-dataset config + target loaders for the cross-dataset (LODO) experiment.

One place that knows, per dataset: the NOVA db name, the assessed role (the person
being rated — doctor in the clinical sets, teacher in the classroom set; the
patient/parent is an actor/foil), where the per-session export CSVs live, and how to
load the expert overall-impression target normalised to 1-5 (higher = better).

Targets:
  ticks    Ground_Truth.sav            BGR_Exp_Gesamteindruck, Rating = 6 - x   (1=bad..5=best)
  vaccines zusammengefuegte_*.sav      BGR_Exp_Gesamteindruck, Rating = 6 - x   (same polarity)
  teacher  fragebogen-*.xlsx 'e1'      Gesamturteil (overall) -- see TEACHER note below

TEACHER: e1 is the expert overall verdict per participant; zuordnung.xlsx is the full
roster (set, tn, name, short, session, fb) and the fb column maps each set to its
fragebogen-<fb>.xlsx file. 85 sessions over sets era_a1..era_b3.

TEACHER SCALE SWITCH (repaired 2026-07-24, see _teacher_deround): the questionnaire
moved from a 5-point to a 4-point overall item after the first 24 sessions, and the
4-point answers were mapped onto 1-5 by round(x*5/4) before being stored. That map is
injective ({1,2,3,4} -> {1,3,4,5}, hence the empty "2") so it inverts exactly; what it
does NOT do is make the two eras comparable, because it lands raw 2 (bottom third of
the 4-point scale) on 3 (middle of the 5-point scale). Since both eras live in one LODO
fold, Spearman is computed *across* them and the misalignment costs ~.08 on that fold.
We therefore invert the rounding and place the 4-point answers on the 5-point range
linearly. Set KT_TEACHER_LEGACY=1 to reproduce the pre-repair (frozen) numbers.
"""
import os
from pathlib import Path
import pandas as pd
import pyreadstat

# data root (env KT_DATA_ROOT): questionnaires under GT_ROOT, NOVA sessions under NOVA_DATA
DATA_ROOT = Path(os.environ.get('KT_DATA_ROOT', 'data'))
GT_ROOT = DATA_ROOT / 'groundtruth'
NOVA_DATA = DATA_ROOT / 'nova'
PROJECT = Path(__file__).resolve().parent.parent  # repo root

# sets collected on the native 5-point item; everything later is 4-point-then-rounded
TEACHER_ERA_A = frozenset({'era_a1', 'era_a2', 'era_a3'})
# inverse of round(x*5/4), then 4-point raw -> 5-point range: 1 + (raw-1)*4/3
_DEROUND = {1: 1.0, 3: 1 + 1 * 4 / 3, 4: 1 + 2 * 4 / 3, 5: 5.0}
TEACHER_LEGACY = os.environ.get('KT_TEACHER_LEGACY') == '1'


# ---- per-dataset static config -------------------------------------------------
DATASETS = {
    'ticks': {
        'db': 'ticks',
        'assessed_role': 'doctor',
        'foil_role': 'patient',
        'export_dir': PROJECT / 'ticks' / 'export',
        'set_file': PROJECT / 'ticks' / 'all.set',
    },
    'vaccines': {
        'db': 'vaccines',
        'assessed_role': 'doctor',
        'foil_role': 'patient',
        'export_dir': PROJECT / 'vaccines' / 'export',
        'set_file': PROJECT / 'vaccines' / 'all.set',
    },
    'teacher': {
        'db': 'teacher',
        'assessed_role': 'teacher',
        'foil_role': 'parent',
        'export_dir': PROJECT / 'teacher' / 'export',
        'set_file': None,
    },
}

# datasets that have a usable target right now (LODO folds).
# teacher = the 24 sessions decodable from new_rating.csv (rounds 1-3); full 81 awaits the
# (round, tn) -> DB-session roster for rounds 0,4,5,6.
READY = ['ticks', 'vaccines', 'teacher']


# ---- target loaders ------------------------------------------------------------
def _ticks_target():
    df, _ = pyreadstat.read_sav(
        str(GT_ROOT / 'ticks' / 'questionnaires' / 'Ground_Truth.sav'))
    df['session'] = df['VP_Nr'].apply(lambda x: f'T_{int(x):02d}')
    df['Rating'] = 6 - df['BGR_Exp_Gesamteindruck']
    return df.dropna(subset=['Rating']).set_index('session')['Rating'].to_dict()


def _vaccines_target():
    df, _ = pyreadstat.read_sav(
        str(GT_ROOT / 'vaccines' / 'questionnaire' / 'zusammengefuegte_Frageboegen.sav'))
    df['session'] = df.apply(
        lambda x: f"{int(x['VP_Nr']):02d}_{int(x['Schauspielpat_Nr'])}", axis=1)
    df['Rating'] = 6 - df['BGR_Exp_Gesamteindruck']
    return df.dropna(subset=['Rating']).set_index('session')['Rating'].to_dict()


def _teacher_deround(e1, set_name):
    """Stored 5-point e1 -> comparable 5-point value, for one questionnaire set.

    Era-A sets are already native 5-point. Era-B sets hold round(raw*5/4) of a 4-point
    answer; inverting that and re-spreading over 1-5 is exact up to the linear placement
    (any strictly monotone placement between the era-A integers gives the same Spearman).
    Idempotent: values not in the stored era-B image are passed through unchanged.
    """
    if TEACHER_LEGACY or set_name in TEACHER_ERA_A or set_name == '_done':
        return float(e1)
    return _DEROUND.get(int(e1), float(e1)) if float(e1).is_integer() else float(e1)


def _teacher_target():
    # Full target, replicating the internal analysis notebook:
    # zuordnung.xlsx (set, tn, short, session) is the roster; each fragebogen-{i}.xlsx has
    # an 'answers' sheet (tn -> a1..e1) and a 'tn2name' sheet (tn -> name). Merge the two on
    # tn, rename name->short, join the roster on [tn, short] to get the session, then e1 is
    # the expert overall verdict (-> Rating), de-rounded per the scale-switch note above.
    base = PROJECT / 'teacher'
    zuord = pd.read_excel(base / 'zuordnung.xlsx')
    # fragebogen file numbers don't track set positions (era_b2 lives in -6, era_b3 in -8),
    # so the roster's fb column carries the file number per set
    fbmap = zuord.drop_duplicates('set').set_index('set')['fb']
    quest = None
    for x in zuord['set'].unique():
        sh = pd.read_excel(base / f'fragebogen-{fbmap[x]}.xlsx', sheet_name=[0, 1])
        tmp = sh[0].merge(sh[1]).rename(columns={'name': 'short'})
        tmp = tmp.merge(zuord[zuord['set'] == x][['tn', 'short', 'session']], on=['tn', 'short'])
        tmp['set'] = x
        quest = tmp if quest is None else pd.concat([quest, tmp])
    quest = quest.dropna(subset=['e1']).astype({'e1': float})
    # era_b1 is the one era-B set whose RAW 4-point answers survive (fragebogen-3-old.xlsx,
    # pre-rescale). Prefer them: they are the instrument, and they repair one session whose
    # rescale was missed and later "fixed" to the wrong neighbour (raw 2 -> stored 1, not 3).
    raw_b1 = base / 'fragebogen-3-old.xlsx'
    if not TEACHER_LEGACY and raw_b1.exists():
        sh = pd.read_excel(raw_b1, sheet_name=['answers', 'tn2name'])
        raw = sh['answers'].merge(sh['tn2name'])[['name', 'e1']].dropna(subset=['e1'])
        raw = {r['name']: 1 + (float(r.e1) - 1) * 4 / 3 for _, r in raw.iterrows()}
        m = quest['set'].eq('era_b1') & quest['short'].isin(raw)
        quest.loc[m, 'e1'] = quest.loc[m, 'short'].map(raw)
        quest.loc[m, 'set'] = '_done'          # already on the 5-point range, skip de-round
    quest['e1'] = [_teacher_deround(r.e1, r.set) for r in quest.itertuples()]
    return quest.set_index('session')['e1'].to_dict()


_LOADERS = {
    'ticks': _ticks_target,
    'vaccines': _vaccines_target,
    'teacher': _teacher_target,
}


def load_target(dataset):
    """session -> expert overall rating, normalised to 1-5 (higher = better)."""
    return _LOADERS[dataset]()


def sessions(dataset):
    """Ordered session list for a dataset (from its .set file, else target keys)."""
    cfg = DATASETS[dataset]
    if cfg['set_file'] and cfg['set_file'].exists():
        return [s for s in cfg['set_file'].read_text().splitlines() if s]
    return sorted(load_target(dataset).keys())


if __name__ == '__main__':
    for ds in DATASETS:
        try:
            t = load_target(ds)
            vals = sorted(set(t.values()))
            print(f'{ds:9s} target n={len(t):3d}  range=[{min(t.values()):.2f},{max(t.values()):.2f}]  '
                  f'distinct={vals if len(vals)<=8 else len(vals)}')
        except Exception as e:
            print(f'{ds:9s} target ERR {e}')
