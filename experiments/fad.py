# fad_score.py — v2
"""
Compute Frechet Audio Distance (FAD) with **fadtk** and auto‑handle the
annoying embedding‑cache step so you don’t crash with “No files provided”.

Usage
-----
python fad_score.py /ref/dir  /gen/dir             # plain FAD
python fad_score.py /ref/dir  /gen/dir  --indiv    # per‑file FAD (CSV)
python fad_score.py /ref/dir  /gen/dir  --inf      # FAD∞ extrapolation
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Union

from fadtk.model_loader import get_all_models
from fadtk.fad import FrechetAudioDistance
from fadtk.fad_batch import cache_embedding_files  # ��� generates *.npy caches

# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #

def _pick_model(name: str) -> "ModelLoader":
    models: Dict[str, any] = {m.name: m for m in get_all_models()}
    if name not in models:
        raise ValueError(f"Model {name!r} not found; try one of {list(models)}")
    return models[name]


def _ensure_embeddings(directory: Path, model) -> None:
    """Generate cached *.npy embeddings if they’re missing."""
    if not directory.is_dir():
        raise NotADirectoryError(directory)
    emb_dir = directory / "embeddings" / model.name
    if any(emb_dir.glob("*.npy")):
        return  # already done
    print(f"[cache] computing embeddings for {directory} …")
    cache_embedding_files(directory, model)


def fad_score(
    ref_dir: Union[str, Path],
    eval_dir: Union[str, Path],
    model_name: str = "clap-2023",
    fad_inf: bool = False,
    individual: bool = False,
):
    """Do‑all wrapper around fadtk with implicit embedding caching."""
    ref_dir, eval_dir = Path(ref_dir), Path(eval_dir)
    model = _pick_model(model_name)

    # 1) Guarantee embeddings exist -----------------------------------------
    _ensure_embeddings(ref_dir, model)
    _ensure_embeddings(eval_dir, model)

    # 2) Fire up FAD ---------------------------------------------------------
    fad = FrechetAudioDistance(model, load_model=False)  # model already cached

    if fad_inf:
        return fad.score_inf(ref_dir, list(eval_dir.glob("*.*")))
    if individual:
        csv_out = f"fad_individual_{model_name}.csv"
        return fad.score_individual(ref_dir, eval_dir, csv_out)

    return fad.score(ref_dir, eval_dir)

# --------------------------------------------------------------------------- #
# CLI                                                                        #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compute Frechet Audio Distance")
    ap.add_argument("reference", help="Directory of reference audio files")
    ap.add_argument("evaluation", help="Directory of generated/eval audio files")
    ap.add_argument("-m", "--model", default="clap-2023")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--inf", action="store_true", help="Compute FAD∞ extrapolation")
    g.add_argument("--indiv", action="store_true", help="Per‑file FAD to CSV")

    args = ap.parse_args()

    result = fad_score(
        args.reference,
        args.evaluation,
        model_name=args.model,
        fad_inf=args.inf,
        individual=args.indiv,
    )

    print("\nRESULT:", result)

