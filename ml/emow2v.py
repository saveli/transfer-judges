"""emoW2V — dimensional speech emotion (arousal / dominance / valence) per session.

Standalone HF inference (the cml/emow2v nova module wraps the same audeering model;
running it directly avoids nova-server orchestration and lets us batch all wavs on GPU).
Model: audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim — already in the HF cache.
Output logits order = [arousal, dominance, valence], range ~0..1.

Runs on the assessed-role audio (<role>.audio.wav) per session, windowed, batched on GPU.
Writes per-frame V/A/D series to ml/emow2v/<dataset>/<session>.csv; features.py turns
those into functionals (mean/std/min/max/median) for the Phase-2 re-prediction.
"""
import os
import sys
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
import torch
import torch.nn as nn
from pathlib import Path
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

sys.path.insert(0, str(Path(__file__).resolve().parent))
import dsconfig as ds_cfg

MODEL = 'audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim'
SR = 16000
WIN_S, HOP_S = 3.0, 1.5          # 3 s windows, 1.5 s hop -> per-frame V/A/D series
BATCH = 32
OUT = Path(__file__).resolve().parent / 'emow2v'
DATA = ds_cfg.NOVA_DATA


class RegressionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        x = self.dropout(x)
        x = torch.tanh(self.dense(x))
        return self.out_proj(self.dropout(x))


class EmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        self.init_weights()

    def forward(self, input_values):
        h = torch.mean(self.wav2vec2(input_values)[0], dim=1)
        return self.classifier(h)


def _windows(y):
    w, h = int(WIN_S * SR), int(HOP_S * SR)
    if len(y) < w:
        return [y]
    return [y[i:i + w] for i in range(0, len(y) - w + 1, h)]


def _load_16k_mono(path):
    y, sr = sf.read(str(path), dtype='float32')
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        from math import gcd
        g = gcd(SR, sr)
        y = resample_poly(y, SR // g, sr // g).astype('float32')
    return y


def process_audio(path, model, proc, device):
    y = _load_16k_mono(path)
    wins = _windows(y)
    vad = []
    for i in range(0, len(wins), BATCH):
        batch = wins[i:i + BATCH]
        x = proc([w for w in batch], sampling_rate=SR, return_tensors='pt', padding=True)
        with torch.no_grad():
            out = model(x['input_values'].to(device)).cpu().numpy()  # (b, 3) = a,d,v
        vad.append(out)
    return np.concatenate(vad) if vad else np.empty((0, 3))


def run(datasets=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    proc = Wav2Vec2Processor.from_pretrained(MODEL)
    model = EmotionModel.from_pretrained(MODEL).to(device).eval()

    for dataset in (datasets or ds_cfg.READY):
        cfg = ds_cfg.DATASETS[dataset]
        rol = cfg['assessed_role']
        (OUT / dataset).mkdir(parents=True, exist_ok=True)
        sess = ds_cfg.sessions(dataset)
        done = 0
        for ses in sess:
            wav = DATA / cfg['db'] / ses / f'{rol}.audio.wav'
            outf = OUT / dataset / f'{ses}.csv'
            if outf.exists():
                done += 1
                continue
            if not wav.exists():
                print(f'  {dataset}/{ses}: no {wav.name}')
                continue
            vad = process_audio(wav, model, proc, device)
            pd.DataFrame(vad, columns=['arousal', 'dominance', 'valence']).to_csv(outf, index=False)
            done += 1
            if done % 10 == 0:
                print(f'  {dataset}: {done}/{len(sess)}')
        print(f'{dataset}: {done}/{len(sess)} sessions -> ml/emow2v/{dataset}/')


if __name__ == '__main__':
    run()
