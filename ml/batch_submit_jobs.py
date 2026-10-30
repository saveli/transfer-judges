"""Submit DISCOVER jobs (FILE MODE) for the phase-6 feature upgrades (venue gate).

File mode (uri-based I/O, no DB reads/writes) because the two modules disagree on
scales while sharing output names: emow2v emits V/A/D in [0,1], emotiefflib V/A in
[-1,1] — DB annotations would need colliding 'valence'/'arousal' schemes. In file
mode each module writes its own scheme header into the .annotation file, so no DB
schemes/annotators are touched at all.

Jobs (one per session x role x module; server serialises via ml_lock):
  - emotiefflib : <role>.face.mp4 + <role>.face_bounding_box.stream ->
                  expression (AffectNet-8 discrete) + valence/arousal (continuous).
                  Replaces libreface. All 3 datasets, both roles.
  - emow2v      : <role>.audio.wav -> arousal/dominance/valence (continuous, [0,1]).
                  ticks/teacher: both roles. vaccines: shared.audio.wav ONLY (per-role
                  audio is silenced there); features.py maps shared-audio frames to
                  roles via the diarised transcript segments (same route as sentiment).

Embedding outputs are NOT described -> not saved (avoids ~30 GB of streams).

The server runs in docker: data/nova is mounted at /app/data and the
project dir is NOT mounted. So job uris use the container namespace (/app/data/...)
and outputs land INSIDE the session dirs, next to the other per-session files:
  data/nova/<db>/<session>/<role>.<module>_<output>.annotation
Existing outputs skip their job (resume-safe).

Usage (project root): python ml/batch_submit_jobs.py [--dry-run] [--smoke]
  --smoke: submit only the first ticks session (both modules) to validate the setup.
"""
import json
import sys
from pathlib import Path

import requests
import urllib3

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsconfig as ds_cfg

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

URL = 'https://localhost:27014/process'
HEADERS = {'Content-type': 'application/x-www-form-urlencoded'}
DATA = ds_cfg.NOVA_DATA                  # host view (existence checks)
DATA_CTR = Path('/app/data')              # same dir inside the discover container

EMOTIEFFLIB_OUTS = ['expression', 'valence', 'arousal']
EMOW2V_OUTS = ['arousal', 'dominance', 'valence']


def _find(sesdir, filename):
    """Case-insensitive file lookup (vaccines has e.g. Doctor.face.mp4)."""
    lower = filename.lower()
    for p in sesdir.iterdir():
        if p.name.lower() == lower:
            return p
    return None


def _ctr(host_path):
    """Map a host path under DATA to the container namespace."""
    return DATA_CTR / Path(host_path).relative_to(DATA)


def _out_uris(module, db, ses, rol, out_ids):
    """Host paths of the outputs; container path via _ctr()."""
    return {o: DATA / db / ses / f'{rol}.{module}_{o}.annotation' for o in out_ids}


def emotiefflib_job(db, ses, sesdir, rol):
    video = _find(sesdir, f'{rol}.face.mp4')
    bb = _find(sesdir, f'{rol}.face_bounding_box.stream')
    if video is None or bb is None:
        print(f'  SKIP {db}/{ses}/{rol} emotiefflib: missing '
              f'{"video" if video is None else "bounding box"}')
        return None
    uris = _out_uris('emotiefflib', db, ses, rol, EMOTIEFFLIB_OUTS)
    data = [
        {'id': 'video', 'type': 'input', 'src': 'file:stream:video', 'uri': str(_ctr(video))},
        {'id': 'bounding_box', 'type': 'input', 'src': 'file:stream', 'uri': str(_ctr(bb))},
        {'id': 'expression', 'type': 'output', 'src': 'file:annotation:discrete',
         'uri': str(_ctr(uris['expression']))},
        {'id': 'valence', 'type': 'output', 'src': 'file:annotation:continuous',
         'uri': str(_ctr(uris['valence']))},
        {'id': 'arousal', 'type': 'output', 'src': 'file:annotation:continuous',
         'uri': str(_ctr(uris['arousal']))},
    ]
    return {'trainerFilePath': 'emotiefflib/emotiefflib.trainer',
            'leftContext': '0', 'rightContext': '0', 'frameSize': '0',
            'data': json.dumps(data),
            'options': json.dumps({'model_name': 'enet_b0_8_va_mtl', 'batch_size': 64,
                                   'face_detection_min_conf': 0.5, 'margin': 0.1}),
            'force': 'False', 'annoMinGap': '0.5', 'annoMinDur': '0.15'}, uris


def emow2v_job(db, ses, sesdir, rol):
    audio = _find(sesdir, f'{rol}.audio.wav')
    if audio is None:
        print(f'  SKIP {db}/{ses}/{rol} emow2v: missing audio')
        return None
    uris = _out_uris('emow2v', db, ses, rol, EMOW2V_OUTS)
    data = [{'id': 'audio', 'type': 'input', 'src': 'file:stream:audio', 'uri': str(_ctr(audio))}]
    data += [{'id': o, 'type': 'output', 'src': 'file:annotation:continuous', 'uri': str(_ctr(u))}
             for o, u in uris.items()]
    return {'trainerFilePath': 'emow2v/emow2v.trainer',
            'leftContext': '960', 'rightContext': '0', 'frameSize': '40',
            'data': json.dumps(data),
            'options': json.dumps({'batch_size': 250}),
            'force': 'False'}, uris


def build_jobs(smoke=False):
    jobs = []
    for dataset, cfg in ds_cfg.DATASETS.items():
        db = cfg['db']
        sessions = sorted(p.name for p in (DATA / db).iterdir() if p.is_dir())
        if smoke:
            if dataset != 'ticks':
                continue
            sessions = sessions[:1]
        emow2v_roles = ['shared'] if dataset == 'vaccines' else [cfg['assessed_role'], cfg['foil_role']]
        for ses in sessions:
            sesdir = DATA / db / ses
            for rol in (cfg['assessed_role'], cfg['foil_role']):
                j = emotiefflib_job(db, ses, sesdir, rol)
                if j:
                    jobs.append((f'{dataset}-emotiefflib-{ses}-{rol}', db, ses, *j))
            for rol in emow2v_roles:
                j = emow2v_job(db, ses, sesdir, rol)
                if j:
                    jobs.append((f'{dataset}-emow2v-{ses}-{rol}', db, ses, *j))
    return jobs


def main(dry_run=False, smoke=False):
    jobs = build_jobs(smoke=smoke)
    todo = [(jid, db, ses, job, uris) for jid, db, ses, job, uris in jobs
            if not all(u.exists() for u in uris.values())]
    print(f'{len(jobs)} jobs, {len(jobs) - len(todo)} already done, {len(todo)} to submit')
    if dry_run:
        for jid, *_ in todo[:20]:
            print(f'  would submit {jid}')
        if len(todo) > 20:
            print(f'  ... and {len(todo) - 20} more')
        return
    for i, (jid, db, ses, job, uris) in enumerate(todo):
        job = job | {'jobID': jid, 'dataset': db, 'sessions': json.dumps([ses])}
        req = requests.post(URL, headers=HEADERS, data=job, verify=False)
        if req.status_code != 200:
            print(f'canceled at {jid} ({i + 1}/{len(todo)}): {req.status_code} {req.text[:200]}')
            return
        if (i + 1) % 50 == 0 or i == len(todo) - 1:
            print(f'  submitted {i + 1}/{len(todo)}')
    print('done — jobs queue server-side (one at a time)')


if __name__ == '__main__':
    main(dry_run='--dry-run' in sys.argv, smoke='--smoke' in sys.argv)
