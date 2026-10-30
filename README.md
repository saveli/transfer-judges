# Cross-domain transfer of counseling-quality ratings

Code from recorded counseling sessions to predictions of the expert overall rating, and
the predictions themselves, for three German corpora of simulated counseling (`ticks`,
`vaccines`, `teacher`; n = 57 / 53 / 85 sessions). Each domain is predicted by a model
trained only on the other two (leave-one-domain-out).

## Pipeline

1. **Features** (`ml/batch_submit_jobs.py`, `ml/emow2v.py`, `ml/features.py`): face and
   speech emotion, prosody, gaze, smile, turn-taking and dyadic synchrony per session,
   aggregated to one feature matrix per domain. `ml/vacc_au50_remap.py` repairs the
   speaker attribution of the diarised `vaccines` transcripts.
2. **LLM judges** (`ml/llm_features.py`, `ml/run_vacc_repair.py`): four local ollama
   judges score each transcript on instrument-anchored constructs. The prompts are the
   `PROMPT*` constants in `ml/llm_features.py`; models and digests in `ollama_models.md`.
3. **Matrices** (`ml/experiment.py` `load_matrices`, `ml/dsconfig.py`): features, judge
   scores and the expert target per domain, plus per-domain standardised copies.
4. **Fixed configuration, LODO predictions** (`predict.py`): median imputation, variance
   filter, standardisation, PCA-10, elastic net; nothing is tuned. Training domains are
   standardised per domain; the held-out domain is standardised with the pooled
   statistics of the training domains (inductive: no statistic of the test domain is
   used). Two systems: `stack` (judges + compact multimodal
   features: per-channel summaries and assessed-minus-interlocutor deltas) and `language-only` (the judges alone).

The step-by-step order, every setting to edit for your own corpora and the judge-cache
layout are in `ml/CONFIG.md`; environment variables (data root, NOVA database login,
run switches, ollama host) in `.env.example`.

## Run on your own data

```bash
cp .env.example .env            # fill in, see ml/CONFIG.md for what else to edit
PYTHONPATH=ml python ml/batch_submit_jobs.py
PYTHONPATH=ml python ml/features.py
PYTHONPATH=ml python ml/llm_features.py     # plus the judge runs listed in ml/CONFIG.md
uv run python predict.py --out my_predictions.csv
```

Feature extraction and judging need the NOVA/DISCOVER stack, a GPU and an ollama server
(environment: `env/pipeline-requirements.txt`). `predict.py` and `evaluate.py` run on
CPU in the locked environment (`uv sync`; `pyproject.toml`, `uv.lock`).

## Predictions

`predictions.csv`: `domain, session_idx, system, y, pred`, one row per session and system.
`y` is the expert overall rating (target), `pred` the out-of-fold prediction of the model
trained on the other two domains under the inductive protocol above. `session_idx` is an
opaque per-domain index (`001..NNN`, random order); it does not encode or map back to any
session identifier, and the two systems share it within a domain.

Verify the reported transfer correlations without any data:

```bash
uv run python evaluate.py      # Spearman rho per held-out domain and the mean
```

Expected: `stack` mean .533 (ticks .485, vaccines .517, teacher .598), `language-only`
mean .523 (.485, .486, .597).

## Not released

- The audio/video recordings and transcripts (consent terms exclude redistribution).
- Everything per session derived from them: feature matrices, extraction outputs, judge
  outputs.
- Any mapping from `session_idx` to sessions, participants or actors.

## License

MIT, see `LICENSE`. Third-party models and extractors keep their own licenses.
