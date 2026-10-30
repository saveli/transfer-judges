"""Vaccines transcript repair via AU50 ground truth (phase 10d).

Vaccines transcripts come from diarising shared audio; the B/G speaker labels carry
attribution errors that plausibly cap the language judges there (the vacc fold stops
improving as judges strengthen). The corpus however has HUMAN speech-activity ground
truth: per-role AU50 annotations (a dedicated annotator, SSI format) on disk for all 53
sessions. This script:
  1. re-attributes every diarised transcript segment to the role whose AU50 speech
     intervals overlap it most (reporting the flip rate = measured diarisation error),
  2. rebuilds the merged B/G dialog text from the repaired attribution,
  3. re-runs the granite-30b theory judge on the repaired vaccines transcripts
     -> ml/llm_theory_granite30_au50fix/vaccines/,
  4. prints per-construct correlation with the expert target, repaired vs original.

Diagnostic first (one judge, one corpus); full integration (all judges, features,
DB annotator) only if the repair moves the needle.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import spearmanr
import discover_utils.data.handler.nova_db_handler as mh

import dsconfig as ds_cfg
import llm_features as lf

DATA = ds_cfg.NOVA_DATA / 'vaccines'
OUT = Path(__file__).resolve().parent / 'llm_theory_granite30_au50fix'
MODEL = 'granite4.1:30b'


def au50_intervals(ses, role):
    """(start_s, end_s) speech intervals from the SSI annotation data file."""
    f = DATA / ses / f'{role}.au50.annotation~'
    if not f.exists():
        return None
    iv = []
    for line in f.read_text().strip().splitlines():
        parts = line.split(';')
        if len(parts) >= 2:
            iv.append((float(parts[0]), float(parts[1])))
    return sorted(iv)


def overlap(seg, ivs):
    a, b = seg
    return sum(max(0.0, min(b, e) - max(a, s)) for s, e in ivs)


def remap_session(ah, db, ses):
    """Pool both roles' diarised segments, re-attribute by AU50 overlap.
    Returns (merged repaired dialog text, n_segments, n_flipped) or None."""
    gt = {r: au50_intervals(ses, r) for r in ('doctor', 'patient')}
    if any(v is None for v in gt.values()):
        return None
    segs = []
    for rol in ('doctor', 'patient'):
        try:
            d = ah.load(dataset=db, session=ses, scheme='transcript', role=rol,
                        annotator='annotator_main').data
        except Exception:
            continue
        if d is None or len(d) == 0:
            continue
        df = pd.DataFrame(d)
        for _, r in df.iterrows():
            segs.append(dict(f=float(r['from']) / 1000.0, t=float(r['to']) / 1000.0,
                             text=str(r['name']).strip(), orig=rol))
    if not segs:
        return None
    flipped = 0
    for s in segs:
        ov = {r: overlap((s['f'], s['t']), gt[r]) for r in gt}
        # keep original attribution when AU50 sees no speech at all (silence/noise seg)
        new = max(ov, key=ov.get) if max(ov.values()) > 0 else s['orig']
        s['new'] = new
        flipped += int(new != s['orig'])
    segs.sort(key=lambda s: s['f'])
    text = '\n'.join(('B: ' if s['new'] == 'doctor' else 'G: ') + s['text']
                     for s in segs if s['text'])
    return text, len(segs), flipped, segs


def main():
    load_dotenv('.env')
    ah = mh.AnnotationHandler(db_host=os.environ['dbHost'], db_port=int(os.environ['dbPort']),
                              db_user=os.environ['dbUser'], db_password=os.environ['dbPassword'])
    db = ds_cfg.DATASETS['vaccines']['db']
    (OUT / 'vaccines').mkdir(parents=True, exist_ok=True)
    tot_seg = tot_flip = 0
    for ses in ds_cfg.sessions('vaccines'):
        res = remap_session(ah, db, ses)
        if res is None:
            print(f'  {ses}: no AU50 GT, skip')
            continue
        text, n, fl, _segs = res
        tot_seg += n
        tot_flip += fl
        outf = OUT / 'vaccines' / f'{ses}.json'
        if outf.exists():
            continue
        scores = None
        for attempt in range(3):
            scores = lf.query_dialog(text, seed=attempt, model=MODEL,
                                     prompt=lf.PROMPT6, constructs=lf.CONSTRUCTS4)
            if scores and len(scores) == len(lf.CONSTRUCTS4):
                break
        if scores:
            outf.write_text(json.dumps(scores))
        print(f'  {ses}: {fl}/{n} segments flipped, judge {"ok" if scores else "FAIL"}')
    print(f'\nflip rate: {tot_flip}/{tot_seg} = {tot_flip / max(tot_seg, 1):.1%} '
          f'(measured diarisation role-attribution error)')

    # ---- before/after correlation with expert target --------------------------------
    target = ds_cfg.load_target('vaccines')
    old_dir = Path(__file__).resolve().parent / 'llm_theory_granite30' / 'vaccines'
    rows = []
    for f in (OUT / 'vaccines').glob('*.json'):
        ses = f.stem
        of = old_dir / f'{ses}.json'
        if ses in target and of.exists():
            rows.append((target[ses], json.loads(of.read_text()), json.loads(f.read_text())))
    y = np.array([r[0] for r in rows])
    print(f'\nper-construct Spearman with expert target (n={len(rows)}):')
    print(f'{"construct":20s} {"orig":>7s} {"au50fix":>8s}')
    for c in lf.CONSTRUCTS4:
        o = np.array([r[1][c] for r in rows])
        n_ = np.array([r[2][c] for r in rows])
        ro = spearmanr(y, o).correlation if np.std(o) > 0 else float('nan')
        rn = spearmanr(y, n_).correlation if np.std(n_) > 0 else float('nan')
        print(f'{c:20s} {ro:7.3f} {rn:8.3f}')


if __name__ == '__main__':
    main()
