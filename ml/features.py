"""Build the functionals feature matrix (assessed role) per dataset, from the DB.

Reads streams + annotations directly via discover_utils handlers (same path the
per-dataset export.py scripts use internally), so no CSV re-export is needed.

Per session, over the *assessed role* (doctor / teacher):
  - continuous streams opensmile(88), openface2 aureg(17): functionals
    mean/std/min/max/median per column.
  - emotiefflib (DISCOVER file-mode, phase 7; replaces libreface): 8-class expression
    time fractions (emotion_*) + face valence/arousal functionals (ef_*). Files
    <role>.emotiefflib_{expression,valence,arousal}.annotation in the session dir.
  - emoW2V V/A/D (DISCOVER file-mode, phase 7; replaces the local ml/emow2v CSV cache,
    which was assessed-only — its dlt_ V/A/D deltas were silently 0): per-role
    <role>.emow2v_{arousal,dominance,valence}.annotation. Vaccines has no usable
    per-role audio -> shared-role files masked by the role's diarised transcript
    segments (same route as sentiment).
  - sentiment (per-frame score): same functionals. Source per dataset (vaccines = shared role).
  - discrete annotations: gaze/smile -> rate/count/meandur.
  - transcript -> verbal aggregates.

Plus three low-dimensional additions (n-safe, no width blow-up):
  - cmp_*  : a compact interpretable summary (F0, loudness, sentiment, gaze/smile rate,
             key emotions, speech rate, V/A/D) — for the "does compact prosody carry signal"
             test vs the 440-dim opensmile dump.
  - dlt_*  : assessed-minus-foil deltas of that same compact summary — dyadic/interaction
             signal (covariance) without doubling feature width.
  - syn_*  : dyadic synchrony — windowed correlation + lead-lag of F0/loudness between
             assessed and foil (temporal coupling; not linearly implied by cmp+dlt).
  - mut_*  : mutual gaze / mutual smile — interval intersection of both roles' segment
             annotations (rate/count/meandur), ported from an internal analysis notebook.
  - tt_*   : turn-taking — response latencies, interruption fractions, speaking-time
             share, overlap, turn counts/lengths from both roles' transcript timestamps.

Output: ml/features/<dataset>.parquet  (index = session, cols = features + 'target').
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import discover_utils.data.handler.nova_db_handler as mh
from discover_utils.data.handler.file_handler import FileHandler

import dsconfig as ds_cfg

OUT = Path(__file__).resolve().parent / os.environ.get('KT_FEAT_DIR', 'features')
DATA = ds_cfg.NOVA_DATA

# eGeMAPSv02 functionals name -> column index of the 25 Hz windowed opensmile stream
_EG = {name: int(i) for i, name in
       (l.split('\t') for l in (Path(__file__).resolve().parent / 'egemaps88.txt')
        .read_text().strip().splitlines())}

# phase-11 prosody-v2 block (prior in-house evidence: variability/slope functionals dominate,
# means rank bottom; semitone F0 = built-in speaker normalization, fixes the raw-Hz
# sex confound of cmp_f0_mean)
PRO_COLS = {
    'f0sem_mean': 'F0semitoneFrom27.5Hz_sma3nz_amean',
    'f0sem_range': 'F0semitoneFrom27.5Hz_sma3nz_pctlrange0-2',
    'f0_riseslope': 'F0semitoneFrom27.5Hz_sma3nz_meanRisingSlope',
    'f0_fallslope': 'F0semitoneFrom27.5Hz_sma3nz_meanFallingSlope',
    'jitter': 'jitterLocal_sma3nz_amean',
    'shimmer': 'shimmerLocaldB_sma3nz_amean',
    'hnr': 'HNRdBACF_sma3nz_amean',
    'loud_std': 'loudness_sma3_stddevNorm',
    'loud_range': 'loudness_sma3_pctlrange0-2',
    'voiced_per_sec': 'VoicedSegmentsPerSec',
    'voiced_seglen': 'MeanVoicedSegmentLengthSec',
}


def _prosody2(opensmile):
    if opensmile is None:
        return {f'pro_{k}': np.nan for k in PRO_COLS}
    return {f'pro_{k}': float(np.nanmean(opensmile[:, _EG[c]])) for k, c in PRO_COLS.items()}
FH = FileHandler()
FRAME_HZ = 25.0
EMOTIONS = ['neutral', 'happiness', 'sadness', 'surprise', 'fear', 'disgust', 'anger', 'contempt']
AUREG = 'face.openface2[0,1,0].openface2_aureg[0,1,0]'

# per-dataset sentiment source: ('anno', scheme, annotator, field, role) | ('stream', name, col, role)
# 'role' None -> assessed role. vaccines sentiment only exists at role 'shared' (diarised-transcript
# sentiment; poor audio meant shared-audio + diarisation, so per-role sentiment rides the transcript).
SENTIMENT_SRC = {
    'ticks':    ('anno', 'sentiment', 'annotator_main', 'score', None),
    'vaccines': ('anno', 'sentiment', 'annotator_main', 'score', 'shared'),
    'teacher':  ('anno', 'sentiment', 'annotator_main', 'score', None),
}


def _annotator(dataset, scheme):
    """vaccines smile is auto-generated under annotator 'system' (n=53); 'annotator_main'
    covers only 2 vaccines sessions there. Everything else lives under 'annotator_main'."""
    return 'system' if dataset == 'vaccines' and scheme == 'smile' else 'annotator_main'


# Face-derived features are invalid for this session (detector runs, but the lower
# face is not usable in the video, so expression/smile output is unreliable). Gaze is
# kept. Values become NaN (imputed), not fake zeros.
# The session is given by its opaque predictions.csv index; real session names are
# not released. On your own data, list (dataset, session name, role) triples.
FACE_INVALID = {('vaccines', '032', 'doctor')}


def _functionals(arr, prefix):
    arr = np.asarray(arr, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    out = {}
    fns = {'mean': np.nanmean, 'std': np.nanstd, 'min': np.nanmin, 'max': np.nanmax, 'median': np.nanmedian}
    for j in range(arr.shape[1]):
        col = arr[:, j]
        col = col[np.isfinite(col)]
        for name, fn in fns.items():
            out[f'{prefix}_{j:02d}_{name}'] = float(fn(col)) if col.size else np.nan
    return out


def _rate(data, total_ms, active_id=0):
    if len(data) == 0 or total_ms <= 0:
        return 0.0
    dur = np.clip(data['to'] - data['from'], 0, None).astype(float)
    return float(dur[data['id'] == active_id].sum() / total_ms)


def _seg_stats(data, total_ms, prefix, active_id=0):
    if len(data) == 0 or total_ms <= 0:
        return {f'{prefix}_rate': 0.0, f'{prefix}_count': 0, f'{prefix}_meandur': 0.0}
    dur = np.clip(data['to'] - data['from'], 0, None).astype(float)
    mask = data['id'] == active_id
    on = dur[mask]
    return {f'{prefix}_rate': float(on.sum() / total_ms), f'{prefix}_count': int(mask.sum()),
            f'{prefix}_meandur': float(on.mean()) if on.size else 0.0}


def _emotion_fractions(data, total_ms):
    out = {f'emotion_{e}': 0.0 for e in EMOTIONS}
    out['emotion_rest'] = 0.0                            # emotiefflib id 8 = no face detected
    if len(data) == 0 or total_ms <= 0:
        return out
    dur = np.clip(data['to'] - data['from'], 0, None).astype(float)
    for eid in np.unique(data['id']):
        if 0 <= eid < len(EMOTIONS):
            out[f'emotion_{EMOTIONS[eid]}'] = float(dur[data['id'] == eid].sum() / total_ms)
        elif eid == len(EMOTIONS):
            out['emotion_rest'] = float(dur[data['id'] == eid].sum() / total_ms)
    return out


def _transcript_stats(data):
    keys = ['tr_duration', 'tr_segcount', 'tr_segdur_mean', 'tr_wordcount_mean',
            'tr_wordlen_mean', 'tr_wordpersec_mean', 'tr_statements', 'tr_questions']
    if len(data) == 0:
        return {k: 0.0 for k in keys}
    df = pd.DataFrame(data)
    df['text'] = df['name'].astype(str)
    segdur = (df['to'] - df['from']) / 1000.0
    words = df['text'].apply(lambda x: len(x.split()))
    wordlen = df['text'].apply(lambda x: np.mean([len(w) for w in x.strip('.?!').split()]) if x.strip('.?!').split() else 0.0)
    wps = words / segdur.replace(0, np.nan)
    return {'tr_duration': float((df['to'].max() - df['from'].min()) / 1000.0), 'tr_segcount': int(len(df)),
            'tr_segdur_mean': float(segdur.mean()), 'tr_wordcount_mean': float(words.mean()),
            'tr_wordlen_mean': float(wordlen.mean()), 'tr_wordpersec_mean': float(wps.mean()),
            'tr_statements': float(df['text'].apply(lambda x: x.replace('..', '.').count('.')).mean()),
            'tr_questions': float(df['text'].apply(lambda x: x.count('?')).mean())}


SL_KEYS = ['sl_seg_per_min', 'sl_words_per_min', 'sl_wordlen_mean', 'sl_question_ratio',
           'sl_speech_ratio', 'sl_segdur_mean', 'sl_unfinished_ratio']


def _shallow_ling(data, total_ms):
    """sl_* shallow linguistic block — port of the LING set (7 features),
    rate-normalised because sessions vary in length (the original used fixed 60 s windows):
    amount of talk, question/unfinished fractions, speech share, segment granularity.
    Core-4 mapping (LING_CORE): words_per_min, question_ratio, speech_ratio,
    segdur_mean."""
    if len(data) == 0 or total_ms <= 0:
        return {k: np.nan for k in SL_KEYS}
    df = pd.DataFrame(data).sort_values('from')
    txt = df['name'].astype(str).str.strip()
    words = txt.apply(lambda x: len(x.split()))
    wl = [len(w) for t in txt for w in t.strip('.?!…').split()]
    segdur_s = (df['to'] - df['from']).astype(float) / 1000.0
    # unfinished (LING def): trails off ('...') or lacks a .?! terminal and is not
    # continued by the same speaker within 2 s (per-role transcript, so "same speaker"
    # = this role's next segment)
    stripped = txt.str.rstrip()
    no_term = ~stripped.str[-1:].isin(list('.?!'))
    trail = stripped.str.endswith('...') | stripped.str.endswith('…')
    continued = (df['from'].shift(-1) - df['to']).fillna(np.inf) < 2000
    unfinished = trail | (no_term & ~continued)
    mins = total_ms / 60000.0
    return {'sl_seg_per_min': float(len(df) / mins),
            'sl_words_per_min': float(words.sum() / mins),
            'sl_wordlen_mean': float(np.mean(wl)) if wl else 0.0,
            'sl_question_ratio': float(stripped.str.endswith('?').mean()),
            'sl_speech_ratio': float(np.clip(segdur_s.sum() * 1000.0 / total_ms, 0, 1)),
            'sl_segdur_mean': float(segdur_s.mean()),
            'sl_unfinished_ratio': float(unfinished.mean())}


def _load_stream(sh, db, ses, rol, name, with_sr=False):
    try:
        s = sh.load(dataset=db, session=ses, name=name, role=rol)
        sr = getattr(s.meta_data, 'sample_rate', None) or FRAME_HZ
        return (s.data, float(sr)) if with_sr else s.data
    except Exception:
        return (None, FRAME_HZ) if with_sr else None


# phase-11: optional AU50-based repair of the diarised vaccines transcripts (16.1% of
# segments role-misattributed vs the human speech GT; see vacc_au50_remap.py). Enabled
# via KT_VACC_AU50=1 so the original feature matrix stays reproducible.
_AU50_REPAIR = os.environ.get('KT_VACC_AU50') == '1'
_REMAP_CACHE = {}


def _transcript_data(ah, db, ses, rol, dataset):
    if dataset == 'vaccines' and _AU50_REPAIR:
        if ses not in _REMAP_CACHE:
            from vacc_au50_remap import remap_session
            res = remap_session(ah, db, ses)
            segs = res[3] if res else []
            _REMAP_CACHE[ses] = {
                r: [{'from': s['f'] * 1000.0, 'to': s['t'] * 1000.0, 'name': s['text']}
                    for s in segs if s['new'] == r]
                for r in ('doctor', 'patient')}
        return _REMAP_CACHE[ses].get(rol, [])
    return _load_anno(ah, db, ses, rol, 'transcript', _annotator(dataset, 'transcript'))


def _load_anno(ah, db, ses, rol, scheme, annotator):
    try:
        d = ah.load(dataset=db, session=ses, scheme=scheme, role=rol, annotator=annotator).data
        return d if d is not None else []
    except Exception:
        return []


def _file_anno(db, ses, rol, name):
    """Load a DISCOVER file-mode annotation from the session dir (role case-insensitive)."""
    sesdir = DATA / db / ses
    if not sesdir.is_dir():
        return None
    want = f'{rol}.{name}.annotation'.lower()
    for p in sesdir.iterdir():
        if p.name.lower() == want:
            try:
                return FH.load(fp=p)
            except Exception:
                return None
    return None


def _cont_scores(anno):
    return np.asarray(anno.data['score'], float) if anno is not None and len(anno.data) else None


def _emow2v_vad(ah, db, ses, rol, dataset):
    """Nx3 [arousal, dominance, valence] (25 Hz, [0,1]) for one role. vaccines: the
    shared-audio series masked to the role's diarised transcript segments."""
    src_rol = 'shared' if dataset == 'vaccines' else rol
    series = [_cont_scores(_file_anno(db, ses, src_rol, f'emow2v_{n}'))
              for n in ('arousal', 'dominance', 'valence')]
    if any(s is None for s in series):
        return None
    n = min(len(s) for s in series)
    vad = np.stack([s[:n] for s in series], axis=1)
    if dataset == 'vaccines':
        tr = _transcript_data(ah, db, ses, rol, dataset)
        iv = _iv_list(tr)
        if not iv:
            return None
        mask = np.zeros(n, bool)
        for f, t in iv:
            mask[int(f / 1000 * FRAME_HZ):int(np.ceil(t / 1000 * FRAME_HZ))] = True
        vad = vad[mask]
    return vad if len(vad) else None


def _sentiment_series(sh, ah, db, ses, rol, dataset):
    kind, *rest = SENTIMENT_SRC[dataset]
    src_role = rest[-1] or rol
    if kind == 'anno':
        scheme, annotator, field, _ = rest
        d = _load_anno(ah, db, ses, src_role, scheme, annotator)
        return np.asarray(d[field], float) if len(d) else None
    name, col, _ = rest
    s = _load_stream(sh, db, ses, src_role, name)
    return np.asarray(s[:, col], float) if s is not None else None


def _compact(opensmile, sentiment, emo_data, gaze_data, smile_data, vad, ef_val, ef_aro, tr, total_ms):
    """Low-dim interpretable summary (used both as cmp_ features and for foil deltas)."""
    f0 = opensmile[:, 0] if opensmile is not None else np.array([np.nan])      # opensmile col 0 = F0-related
    loud = opensmile[:, 10] if opensmile is not None else np.array([np.nan])   # col 10 = loudness (per sessionstats)
    emo = _emotion_fractions(emo_data, total_ms)
    return {
        'f0_mean': float(np.nanmean(f0)), 'f0_std': float(np.nanstd(f0)),
        'loud_mean': float(np.nanmean(loud)),
        'sentiment_mean': float(np.nanmean(sentiment)) if sentiment is not None else np.nan,
        'gaze_rate': _rate(gaze_data, total_ms), 'smile_rate': _rate(smile_data, total_ms),
        'emo_happy': emo['emotion_happiness'], 'emo_neutral': emo['emotion_neutral'],
        'wps': tr['tr_wordpersec_mean'],
        'arousal': float(np.nanmean(vad[:, 0])) if vad is not None else np.nan,
        'dominance': float(np.nanmean(vad[:, 1])) if vad is not None else np.nan,
        'valence': float(np.nanmean(vad[:, 2])) if vad is not None else np.nan,
        'face_valence': float(np.nanmean(ef_val)) if ef_val is not None else np.nan,
        'face_arousal': float(np.nanmean(ef_aro)) if ef_aro is not None else np.nan,
    }


def _sync_pair(x, y, sr, prefix, win_s=15.0, max_lag_s=5.0):
    """Dyadic synchrony of one per-frame signal: windowed correlation (attunement level +
    stability) and global lead-lag (normalised cross-correlation peak within ±max_lag_s;
    negative lag = assessed leads). This is temporal coupling — NOT derivable from the
    session-level functionals or the cmp/dlt scalars."""
    keys = {f'syn_{prefix}_wmean': np.nan, f'syn_{prefix}_wstd': np.nan,
            f'syn_{prefix}_maxr': np.nan, f'syn_{prefix}_lag': np.nan}
    n = min(len(x), len(y))
    w = int(win_s * sr)
    if n < 2 * w:
        return keys
    x, y = np.asarray(x[:n], float), np.asarray(y[:n], float)
    rs = []
    for i in range(0, n - w + 1, w // 2):
        xs, ys = x[i:i + w], y[i:i + w]
        m = np.isfinite(xs) & np.isfinite(ys)
        if m.sum() >= 0.5 * w and np.std(xs[m]) > 0 and np.std(ys[m]) > 0:
            rs.append(np.corrcoef(xs[m], ys[m])[0, 1])
    xi = np.where(np.isfinite(x), x, np.nanmean(x)) - np.nanmean(x)
    yi = np.where(np.isfinite(y), y, np.nanmean(y)) - np.nanmean(y)
    denom = np.sqrt((xi ** 2).sum() * (yi ** 2).sum())
    if denom > 0:
        ml = int(max_lag_s * sr)
        cc = np.array([(xi[max(0, -l):n - max(0, l)] * yi[max(0, l):n - max(0, -l)]).sum() / denom
                       for l in range(-ml, ml + 1)])
        best = int(np.argmax(np.abs(cc)))
        keys[f'syn_{prefix}_maxr'] = float(cc[best])
        keys[f'syn_{prefix}_lag'] = float((best - ml) / sr)
    if rs:
        keys[f'syn_{prefix}_wmean'] = float(np.mean(rs))
        keys[f'syn_{prefix}_wstd'] = float(np.std(rs))
    return keys


def _synchrony(os_a, os_f, sr):
    """syn_* block from both roles' opensmile: F0 (col 0, semitone; 0 = unvoiced -> NaN)
    and loudness (col 10)."""
    f0a = np.where(os_a[:, 0] != 0, os_a[:, 0], np.nan)
    f0f = np.where(os_f[:, 0] != 0, os_f[:, 0], np.nan)
    out = _sync_pair(f0a, f0f, sr, 'f0')
    out.update(_sync_pair(os_a[:, 10], os_f[:, 10], sr, 'loud'))
    return out


def _iv_list(data, active_id=None):
    """Sorted (from, to) ms intervals. active_id filters discrete schemes (0 = active);
    None keeps all rows (transcript/free schemes have no meaningful id)."""
    if len(data) == 0:
        return []
    df = pd.DataFrame(data)
    if active_id is not None and 'id' in df.columns:
        df = df[df['id'] == active_id]
    return [(f, t) for f, t in sorted(zip(df['from'].astype(float), df['to'].astype(float))) if t > f]


def _intersect(A, B):
    """Pairwise intersections of two sorted interval lists."""
    out, j = [], 0
    for f, t in A:
        while j < len(B) and B[j][1] <= f:
            j += 1
        k = j
        while k < len(B) and B[k][0] < t:
            s, e = max(f, B[k][0]), min(t, B[k][1])
            if e > s:
                out.append((s, e))
            k += 1
    return out


MUT_KEYS = [f'mut_{s}_{m}' for s in ('gaze', 'smile') for m in ('rate', 'count', 'meandur')]
TT_KEYS = ['tt_lat_a_med', 'tt_lat_a_mean', 'tt_lat_f_med', 'tt_int_a_frac', 'tt_int_f_frac',
           'tt_share_a', 'tt_overlap_frac', 'tt_turns_per_min',
           'tt_turnlen_a_mean', 'tt_turnlen_a_std', 'tt_turnlen_f_mean']


def _mutual(raw_a, raw_f, total_ms):
    """mut_* block: both roles simultaneously gazing / smiling (interval intersection).
    Same construct as sessionstats.ipynb gazemutual/smilemutual (there via 40ms raster)."""
    out = {}
    for scheme in ('gaze', 'smile'):
        iv = _intersect(_iv_list(raw_a[scheme], 0), _iv_list(raw_f[scheme], 0))
        dur = np.array([e - s for s, e in iv], dtype=float)
        out[f'mut_{scheme}_rate'] = float(dur.sum() / total_ms) if total_ms > 0 else 0.0
        out[f'mut_{scheme}_count'] = int(len(iv))
        out[f'mut_{scheme}_meandur'] = float(dur.mean()) if dur.size else 0.0
    return out


def _turn_taking(tr_a, tr_f, total_ms):
    """tt_* block from both roles' transcript timestamps. Turns = consecutive same-speaker
    segments merged; floor transfers give response latencies (negative = starts while the
    other still speaks). a = assessed, f = foil."""
    out = {k: np.nan for k in TT_KEYS}
    A, F = _iv_list(tr_a), _iv_list(tr_f)
    if not A or not F:
        return out
    segs = sorted([(f, t, 'a') for f, t in A] + [(f, t, 'f') for f, t in F])
    turns = []
    for f, t, w in segs:
        if turns and turns[-1][2] == w:
            turns[-1][1] = max(turns[-1][1], t)
        else:
            turns.append([f, t, w])
    lat = {'a': [], 'f': []}
    for prev, cur in zip(turns, turns[1:]):
        lat[cur[2]].append((cur[0] - prev[1]) / 1000.0)
    la, lf = np.array(lat['a']), np.array(lat['f'])
    if la.size:
        out['tt_lat_a_med'] = float(np.median(la))
        out['tt_lat_a_mean'] = float(la.mean())
        out['tt_int_a_frac'] = float((la < 0).mean())
    if lf.size:
        out['tt_lat_f_med'] = float(np.median(lf))
        out['tt_int_f_frac'] = float((lf < 0).mean())
    dur_a = sum(t - f for f, t in A)
    dur_f = sum(t - f for f, t in F)
    if dur_a + dur_f > 0:
        out['tt_share_a'] = float(dur_a / (dur_a + dur_f))
    if total_ms > 0:
        out['tt_overlap_frac'] = float(sum(e - s for s, e in _intersect(A, F)) / total_ms)
        out['tt_turns_per_min'] = float(len(turns) / (total_ms / 60000.0))
    ta = np.array([(t - f) / 1000.0 for f, t, w in turns if w == 'a'])
    tf = np.array([(t - f) / 1000.0 for f, t, w in turns if w == 'f'])
    if ta.size:
        out['tt_turnlen_a_mean'] = float(ta.mean())
        out['tt_turnlen_a_std'] = float(ta.std())
    if tf.size:
        out['tt_turnlen_f_mean'] = float(tf.mean())
    return out


def _role_bundle(sh, ah, db, ses, rol, dataset):
    """Load everything needed for one role; returns (functionals, compact, (opensmile, sr)) or None."""
    opensmile, sr = _load_stream(sh, db, ses, rol, 'opensmile', with_sr=True)
    if opensmile is None:
        return None
    total_ms = opensmile.shape[0] / sr * 1000.0
    aureg = _load_stream(sh, db, ses, rol, AUREG)
    sentiment = _sentiment_series(sh, ah, db, ses, rol, dataset)
    # emotiefflib (file-mode, replaces libreface stream + libreface-derived emotion anno)
    expr = _file_anno(db, ses, rol, 'emotiefflib_expression')
    emo_data = expr.data if expr is not None and expr.data is not None else []
    ef_val = _cont_scores(_file_anno(db, ses, rol, 'emotiefflib_valence'))
    ef_aro = _cont_scores(_file_anno(db, ses, rol, 'emotiefflib_arousal'))
    gaze_data = _load_anno(ah, db, ses, rol, 'gaze', _annotator(dataset, 'gaze'))
    smile_data = _load_anno(ah, db, ses, rol, 'smile', _annotator(dataset, 'smile'))
    tr_data = _transcript_data(ah, db, ses, rol, dataset)
    vad = _emow2v_vad(ah, db, ses, rol, dataset)
    tr = _transcript_stats(tr_data)

    feat = {}
    feat.update(_functionals(opensmile, 'opensmile'))
    if aureg is not None:
        feat.update(_functionals(aureg, 'aureg'))
    if ef_val is not None and ef_aro is not None:
        n = min(len(ef_val), len(ef_aro))
        feat.update(_functionals(np.stack([ef_val[:n], ef_aro[:n]], axis=1), 'ef'))  # 00=valence, 01=arousal
    feat.update(_functionals(sentiment if sentiment is not None else np.array([np.nan]), 'sentiment'))
    feat.update(_emotion_fractions(emo_data, total_ms))
    feat.update(_seg_stats(gaze_data, total_ms, 'gaze'))
    feat.update(_seg_stats(smile_data, total_ms, 'smile'))
    feat.update(tr)
    feat.update(_shallow_ling(tr_data, total_ms))          # sl_* LING port
    if vad is not None:
        feat.update(_functionals(vad, 'emow2v'))

    compact = _compact(opensmile, sentiment, emo_data, gaze_data, smile_data, vad, ef_val, ef_aro, tr, total_ms)
    raw = {'gaze': gaze_data, 'smile': smile_data, 'tr': tr_data, 'total_ms': total_ms,
           'face_invalid': (dataset, ses, rol) in FACE_INVALID,
           'pro': _prosody2(opensmile)}
    if raw['face_invalid']:
        for k in list(feat):
            if k.startswith(('emotion_', 'ef_', 'smile_')):
                feat[k] = np.nan
        for k in ('emo_happy', 'emo_neutral', 'face_valence', 'face_arousal', 'smile_rate'):
            compact[k] = np.nan
    return feat, compact, (opensmile, sr), raw


def build(dataset, sh, ah):
    cfg = ds_cfg.DATASETS[dataset]
    db, rol, foil = cfg['db'], cfg['assessed_role'], cfg['foil_role']
    target = ds_cfg.load_target(dataset)
    export_dir = cfg['export_dir']
    avail = {p.name for p in export_dir.iterdir()} if export_dir.exists() else set(target)
    sess = [s for s in target if s in avail] if avail else list(target)

    rows = {}
    for ses in sess:
        bundle = _role_bundle(sh, ah, db, ses, rol, dataset)
        if bundle is None:
            print(f'  {dataset}/{ses}: no opensmile, skip')
            continue
        feat, cmp_a, (os_a, sr_a), raw_a = bundle
        feat.update({f'cmp_{k}': v for k, v in cmp_a.items()})            # compact interpretable
        feat.update(raw_a['pro'])                                         # prosody-v2 (phase 11)
        foil_b = _role_bundle(sh, ah, db, ses, foil, dataset)             # dyadic deltas + synchrony
        if foil_b is not None:
            cmp_f = foil_b[1]
            os_f = foil_b[2][0]
            raw_f = foil_b[3]
            feat.update({f'dlt_{k}': cmp_a[k] - cmp_f[k] for k in cmp_a})
            feat.update({f'dpr_{k[4:]}': raw_a['pro'][k] - raw_f['pro'][k] for k in raw_a['pro']})
            feat.update(_synchrony(os_a, os_f, sr_a))
            feat.update(_mutual(raw_a, raw_f, raw_a['total_ms']))
            if raw_a['face_invalid'] or raw_f['face_invalid']:
                feat.update({k: np.nan for k in MUT_KEYS if 'smile' in k})
            feat.update(_turn_taking(raw_a['tr'], raw_f['tr'], raw_a['total_ms']))
        else:
            feat.update({f'dlt_{k}': np.nan for k in cmp_a})
            feat.update({f'dpr_{k[4:]}': np.nan for k in raw_a['pro']})
            feat.update(_synchrony(os_a, os_a[:0], sr_a))                 # all-NaN syn block, stable columns
            feat.update({k: np.nan for k in MUT_KEYS + TT_KEYS})
        feat['target'] = target[ses]
        rows[ses] = feat

    mat = pd.DataFrame.from_dict(rows, orient='index').sort_index()
    OUT.mkdir(exist_ok=True)
    mat.to_parquet(OUT / f'{dataset}.parquet')
    print(f'{dataset}: {mat.shape[0]} sessions x {mat.shape[1]-1} features -> {dataset}.parquet')
    return mat


if __name__ == '__main__':
    load_dotenv('.env')
    dbconf = dict(db_host=os.environ['dbHost'], db_port=int(os.environ['dbPort']),
                  db_user=os.environ['dbUser'], db_password=os.environ['dbPassword'])
    sh = mh.StreamHandler(data_dir=str(DATA), **dbconf)
    ah = mh.AnnotationHandler(**dbconf)
    for dataset in ds_cfg.READY:
        build(dataset, sh, ah)
