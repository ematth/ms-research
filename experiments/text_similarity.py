#!/usr/bin/env python
"""
prompt_similarity.py  –  Row-wise prompt–prompt similarity with Stable Audio embeddings
"""
import argparse
import os

import pandas as pd
import torch
import torch.nn.functional as F
from diffusers import StableAudioPipeline
from tqdm.auto import tqdm


def load_prompts(csv_path: str, column: str | None = None) -> list[str]:
    """Read a CSV and return a list[str] containing the prompt column."""
    df = pd.read_csv(csv_path)
    if column is None:
        column = df.columns[0]                # default: first column
    if column not in df.columns:
        raise KeyError(f"'{column}' not found in {csv_path}")
    return df[column].astype(str).tolist()


@torch.inference_mode()
def batch_encode(pipe: StableAudioPipeline,
                 prompts: list[str],
                 batch_size: int = 8,
                 device: str = "cuda") -> torch.Tensor:
    """Encode prompts → pooled embeddings."""
    pooled_chunks: list[torch.Tensor] = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Encoding"):
        batch = prompts[i:i + batch_size]
        # encode_prompt returns (B, T, D)  :contentReference[oaicite:0]{index=0}
        token_embeds = pipe.encode_prompt(
            batch,
            device=device,
            do_classifier_free_guidance=False,  # we just want plain text embeddings
            negative_prompt=None
        )
        pooled = token_embeds.mean(dim=1)      # simple mean-pool over tokens
        pooled_chunks.append(pooled)

    return torch.cat(pooled_chunks, dim=0)      # (N, D)


def main(csv_a: str,
         csv_b: str,
         out_csv: str = "similarities.csv",
         column: str | None = None,
         batch_size: int = 8):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1️⃣  Load pipeline (weights are ~ 1 GB – grab coffee)
    pipe = StableAudioPipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        low_cpu_mem_usage=False,
    ).to(device)
    pipe.enable_model_cpu_offload()            # optional VRAM saver

    # 2️⃣  Load prompts
    prompts_a = load_prompts(csv_a, column)
    prompts_b = load_prompts(csv_b, column)

    if len(prompts_a) != len(prompts_b):
        raise ValueError("CSV files must have the *same* number of rows.")

    # 3️⃣  Embeddings
    emb_a = batch_encode(pipe, prompts_a, batch_size, device)
    emb_b = batch_encode(pipe, prompts_b, batch_size, device)

    # 4️⃣  Cosine similarity (row-wise)
    sims = F.cosine_similarity(emb_a, emb_b, dim=-1).cpu().numpy()

    # 5️⃣  Save results
    df_out = pd.DataFrame({
        "prompt_A": prompts_a,
        "prompt_B": prompts_b,
        "cosine_similarity": sims,
    })
    df_out.to_csv(out_csv, index=False)
    print(f"✔ Similarities written to {os.path.abspath(out_csv)}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Compute row-wise cosine similarity between two CSVs of prompts "
                    "using Stable Audio Open embeddings.")
    p.add_argument("csv_a", help="First CSV file")
    p.add_argument("csv_b", help="Second CSV file")
    p.add_argument("--column", default=None,
                   help="Name of the prompt column (defaults to the first column)")
    p.add_argument("--out_csv", default="similarities.csv",
                   help="Output CSV path")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for embedding encoding")
    args = p.parse_args()
    main(args.csv_a, args.csv_b, args.out_csv, args.column, args.batch_size)
