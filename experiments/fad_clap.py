#!/usr/bin/env python3
"""
compute_fad_clap.py
===================

Compute Fréchet Audio Distance (FAD) between a *reference* directory
and a *candidate* (evaluation) directory using CLAP embeddings,
courtesy of the fadtk toolkit.

USAGE
-----
python compute_fad_clap.py /path/to/ref /path/to/cand \
    --model clap-2023         # or clap-laion-audio / clap-laion-music
    --workers 8               # CPU workers for I/O
    --device cuda             # "cuda" or "cpu"
    --cache_only              # (optional) just pre‑compute embeddings

Dependencies
------------
pip install fadtk torch soundfile
"""

from __future__ import annotations
import argparse, time, sys
from pathlib import Path

import torch
from fadtk.fad import FrechetAudioDistance                 # core scorer
from fadtk.model_loader import get_all_models              # registry of embed models
from fadtk.fad_batch import cache_embedding_files          # pre‑compute embeds

def pick_model(name: str):
    """Return the requested CLAP model loader from fadtk."""
    for m in get_all_models():
        if m.name == name:
            return m
    raise ValueError(
        f"Model '{name}' not found. "
        "Use one of: " + ", ".join(sorted({m.name for m in get_all_models()}))
    )

def main():
    parser = argparse.ArgumentParser("Compute FAD with CLAP embeddings")
    parser.add_argument("reference", type=Path, help="Directory with reference WAV/MP3 files")
    parser.add_argument("candidate", type=Path, help="Directory with candidate WAV/MP3 files")
    parser.add_argument("--model", default="clap-2023",
                        help="CLAP variant registered in fadtk (default: clap-2023)")
    parser.add_argument("--workers", type=int, default=8,
                        help="CPU workers for I/O/feature extraction")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda:6" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--cache_only", action="store_true",
                        help="Just compute & store .npy embedding files, exit afterwards")
    args = parser.parse_args()

    # --- sanity checks -------------------------------------------------------
    if not args.reference.is_dir() or not args.candidate.is_dir():
        sys.exit("Both reference and candidate paths must be directories containing audio.")
    if args.device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA not available – either install a GPU‑enabled PyTorch build or run with --device cpu.")

    # --- grab the CLAP embedder ---------------------------------------------
    model = pick_model(args.model)
    model.device = torch.device(args.device)               # override default device

    print(f"[info] Using model '{model.name}' on {model.device}.")
    t0 = time.time()

    # 1) Pre‑compute / load cached embeddings
    for d in (args.reference, args.candidate):
        cache_embedding_files(d, model, workers=args.workers)

    if args.cache_only:
        print(f"[done] Cached embeddings for both folders in {(time.time()-t0):.1f}s. Bye!")
        return

    # 2) Compute the FAD score
    fad = FrechetAudioDistance(model, audio_load_worker=args.workers, load_model=False)
    score = fad.score(args.reference, args.candidate)      # plain FAD (not FAD‑∞ or per‑track)

    print(f"[result] FAD({args.model}) = {score:.4f}  "
          f"(elapsed {(time.time()-t0):.1f}s)")

if __name__ == "__main__":
    main()
