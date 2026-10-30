"""Re-run the ensemble judges on AU50-repaired vaccines transcripts (phase 10d).

The AU50 ground-truth speech annotations imply 16.1% of diarised vaccines segments
were role-misattributed; repairing the labels lifted the granite judge's vacc-fold
transfer by +.02. This reruns the remaining ensemble judges (llm2 generic, llm3
think, llm4 BGR+OSCE, llm6 anti-band) on the repaired transcripts, into *_au50fix
cache dirs (vaccines only; other corpora keep their mic-separated originals).
"""
from pathlib import Path

import llm_features as lf
from vacc_au50_remap import remap_session

ROOT = Path(__file__).resolve().parent


def au50_text_builder(ah, db, ses, assessed, foil):
    res = remap_session(ah, db, ses)
    return res[0] if res else ''


JOBS = [
    ('llm_dialog_au50fix', lf.MODEL2, dict(prompt=lf.PROMPT2, constructs=lf.CONSTRUCTS), False),
    ('llm_dialog_think_au50fix', lf.MODEL3, dict(prompt=lf.PROMPT2, constructs=lf.CONSTRUCTS), True),
    ('llm_theory_au50fix', lf.MODEL4, dict(prompt=lf.PROMPT4, constructs=lf.CONSTRUCTS4), False),
    ('llm_theory_v2_au50fix', lf.MODEL4, dict(prompt=lf.PROMPT6, constructs=lf.CONSTRUCTS4), False),
]

if __name__ == '__main__':
    for outdir, model, kw, think in JOBS:
        lf.run_dialog(datasets=['vaccines'], out_root=ROOT / outdir, model=model,
                      think=think, tag=outdir, text_builder=au50_text_builder, **kw)
    print('VACC REPAIR RERUNS DONE')
