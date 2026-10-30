# Configuration

## Environment variables

All of them, with defaults and the file that reads each, are in `.env.example` at the
repo root. In short: `KT_DATA_ROOT` (data root, default `data`), the NOVA
database login `dbHost` / `dbPort` / `dbUser` / `dbPassword` (loaded by
`load_dotenv('.env')`), the run switches `KT_FEAT_DIR`, `KT_VACC_AU50`,
`KT_TEACHER_LEGACY`, and `OLLAMA_HOST` for the LLM judges.

Expected layout under the data root (default paths shown):

```
data/nova/<db>/<session>/<role>.<stream>    NOVA session folders (audio, video, extractor outputs)
data/groundtruth/ticks/questionnaires/Ground_Truth.sav
data/groundtruth/vaccines/questionnaire/zusammengefuegte_Frageboegen.sav
```

## What you must edit to run on your own data

| File | Setting | Meaning |
|---|---|---|
| `ml/dsconfig.py` | `DATASETS[<name>]` | one entry per corpus: NOVA db name (`db`), the rated role (`assessed_role`), the interlocutor (`foil_role`), per-session export dir (`export_dir`), session list (`set_file`, or `None` to take the sessions from the target) |
| `ml/dsconfig.py` | `READY` | corpora that form the LODO folds |
| `ml/dsconfig.py` | `_LOADERS`, `_ticks_target`, `_vaccines_target`, `_teacher_target` | target loaders: session id -> expert overall rating on 1-5, higher = better. Questionnaire paths and item columns live here (`.sav` files under `GT_ROOT`; the teacher roster `teacher/zuordnung.xlsx` and `teacher/fragebogen-<fb>.xlsx`) |
| `ml/dsconfig.py` | `TEACHER_ERA_A`, `_DEROUND` | corpus-specific repair of a 4-to-5-point scale switch; drop for a new corpus |
| `ticks/all.set`, `vaccines/all.set` | one session id per line | session lists referenced by `set_file` |
| `ml/dsconfig.py` | session-id prefix of the single-actor corpus (`T_`), set labels (`era_*`) | placeholders in this release; they must match the ids and set labels in your own roster and questionnaire files |
| `ml/features.py` | `_annotator()`, `SENTIMENT_SRC` | NOVA annotator per scheme (`annotator_main` for transcripts, sentiment, gaze, smile; `system` for vaccines smile) and the sentiment source per corpus |
| `ml/llm_features.py`, `ml/vacc_au50_remap.py` | `annotator='annotator_main'` | annotator of the transcripts the judges read |
| `ml/features.py` | `FACE_INVALID` | `(dataset, session, role)` triples whose face-derived features are set to NaN |
| `ml/batch_submit_jobs.py` | `URL`, `DATA_CTR` | DISCOVER extraction server endpoint and its container view of the NOVA data dir |
| `ml/llm_features.py` | judge models and prompts | ollama model tags, sampling options, prompt templates |

## Pipeline order (session data -> predictions)

1. `ml/batch_submit_jobs.py`: DISCOVER jobs per session and role (emotiefflib face
   expression + valence/arousal; emoW2V audio arousal/dominance/valence). Outputs land in
   the NOVA session folders. `ml/emow2v.py` is the standalone emoW2V runner
   (`ml/emow2v/<domain>/<session>.csv`).
2. `ml/features.py`: one feature matrix per domain, `ml/$KT_FEAT_DIR/<domain>.parquet`
   (compact `cmp_`, dyadic deltas `dlt_`, prosody `pro_`/`dpr_`, synchrony, mutual
   gaze/smile, turn-taking, wide functionals). With `KT_VACC_AU50=1` the diarised
   vaccines transcripts are first repaired from the per-role speech annotation
   (`ml/vacc_au50_remap.py`).
3. `ml/llm_features.py`: LLM judges, one JSON of construct scores per session under
   `ml/llm_*/<domain>/`. The five judge runs of the reported systems: `run_dialog()`,
   `run_dialog(think=True)`, `run_theory()`, `run_theory_qwen36()` plus
   `run_theory_qwen36_seed(10)` and `run_theory_qwen36_seed(20)`, `run_theory_granite30()`.
4. Judge reruns on the repaired vaccines transcripts: `ml/run_vacc_repair.py`
   (`llm_dialog`, `llm_dialog_think`, `llm_theory` -> `*_au50fix/vaccines`) and
   `python ml/vacc_au50_remap.py` (granite -> `llm_theory_granite30_au50fix/vaccines`).
5. Assemble the judge caches that `ml/experiment.py` reads (no script, plain filesystem):
   `ml/llm_{dialog,dialog_think,theory,theory_granite30}_fix/<domain>` are symlinks to the
   original cache for the mic-separated domains and to the `*_au50fix` dir for vaccines;
   `ml/llm_theory_qwen36_mean/<domain>/<session>.json` is the per-construct mean of the
   three qwen runs (`llm_theory_qwen36`, `_s10`, `_s20`).
6. `ml/experiment.py` `load_matrices()` joins features, target and judge scores and adds
   the per-domain z copies; `feature_cols()` defines the two reported feature sets
   (`lowdim_all23479c_pro_z` = stack, `llm23479c_z` = language-only). The fixed
   PCA-10 + elastic-net LODO fit with inductive standardisation is `predict.py` at the
   release root.
