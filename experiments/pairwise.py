# pairwise_clap_similarity.py
"""
Compute **pairwise cosine similarity** (or Euclidean distance) between
matching‑named audio files in two directories using LAION‑CLAP embeddings.

Example
-------
python pairwise_clap_similarity.py dirA dirB --csv out.csv

Assumes each directory contains e.g.  ``basic_0_0.wav`` and
``reword_0_0.wav`` with identical basenames. Any files missing from one side
are ignored with a warning.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Dict

import torch
import torchaudio
from transformers import ClapProcessor, ClapModel
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "laion/clap-htsat-unfused"
SAMPLE_RATE = 48_000  # training SR for CLAP

# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------

def load_clip() -> tuple[ClapProcessor, ClapModel]:
    proc = ClapProcessor.from_pretrained(MODEL_NAME)
    model = ClapModel.from_pretrained(MODEL_NAME).to(DEVICE).eval()
    model.requires_grad_(False)
    return proc, model


def get_embedding(path: Path, proc: ClapProcessor, model: ClapModel) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
        sr = SAMPLE_RATE
    inputs = proc(audios=wav.mean(0).numpy(), sampling_rate=sr, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        emb = model.get_audio_features(**inputs).squeeze()
    return emb.cpu()

# ---------------------------------------------------------------------------
# Main pairwise routine
# ---------------------------------------------------------------------------

def pairwise_similarity(dir_a: Path, dir_b: Path) -> pd.DataFrame:
    audio_exts = {".wav", ".flac", ".mp3", ".ogg"}
    files_a = {p.name: p for p in dir_a.iterdir() if p.suffix.lower() in audio_exts}
    files_b = {p.name: p for p in dir_b.iterdir() if p.suffix.lower() in audio_exts}

    common = sorted(set(files_a) & set(files_b))
    if not common:
        sys.exit("No matching filenames between the two directories.")

    missing = sorted(set(files_a) ^ set(files_b))
    for name in missing:
        print(f"[warn] {name} missing from one of the dirs", file=sys.stderr)

    proc, model = load_clip()

    rows: List[Dict[str, float]] = []
    for name in common:
        emb_a = get_embedding(files_a[name], proc, model)
        emb_b = get_embedding(files_b[name], proc, model)

        cos = torch.nn.functional.cosine_similarity(emb_a, emb_b, dim=0).item()
        l2  = torch.dist(emb_a, emb_b).item()
        rows.append({"file": name, "cosine_similarity": cos, "l2_distance": l2})
        print(f"{name:30}  cos={cos:+.3f}  L2={l2:.3f}")

    return pd.DataFrame(rows)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pairwise CLAP cosine similarity between matching audio files")
    ap.add_argument("dir_a", type=Path, help="First directory of audio files")
    ap.add_argument("dir_b", type=Path, help="Second directory of audio files (same filenames)")
    ap.add_argument("--csv", type=Path, help="Optional path to write CSV results")

    args = ap.parse_args()
    df = pairwise_similarity(args.dir_a, args.dir_b)

    if args.csv:
        df.to_csv(args.csv, index=False)
        print(f"\nCSV written to {args.csv}")
