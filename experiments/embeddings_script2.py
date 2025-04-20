#!/usr/bin/env python3
"""
Extract Stable Audio embeddings from an input WAV and use them to
generate new audio with StableAudioPipeline.

Requires:
  pip install diffusers==0.27.0 torch torchaudio soundfile
  (plus an NVIDIA GPU or a lot of patience)
"""

import torch, torchaudio, soundfile as sf
from diffusers import StableAudioPipeline
import pandas as pd
from tqdm import tqdm
import sys, os
import numpy as np


global NUM_WAVEFORMS
NUM_WAVEFORMS: int = 1

global DEVICE
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:1")
print('Using device:', DEVICE)

# ---------- utility helpers ----------
def load_pipe(dtype=torch.float16, device=DEVICE):
    model_id = "stabilityai/stable-audio-open-1.0"
    pipe = StableAudioPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
    ).to(device)
    #pipe.eval()  # we’re only inferencing
    return pipe


def read_wav(path: str, target_sr: int):
    wav, sr = torchaudio.load(path)  # (C, S)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    if wav.shape[0] == 1:           # ensure stereo
        wav = wav.repeat(2, 1)
    return wav.unsqueeze(0)         # -> (B=1, C, S)


def encode_audio(pipe, wav):
    # Ensure dtype matches model weights (usually fp16)
    wav = wav.to(pipe.device, dtype=pipe.vae.dtype)
    with torch.no_grad():
        latents = pipe.vae.encode(wav).latent_dist.mode()
    return latents


def decode_latents(pipe, latents):
    with torch.no_grad():
        wav = pipe.vae.decode(latents).sample
    return wav


def remix(pipe, wav, prompt, *, steps=200, seed=42, seconds=10.0):
    wav = wav.to(pipe.device, dtype=pipe.vae.dtype)  # <- added
    g   = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt=prompt,
        num_inference_steps=steps,
        guidance_scale=1.0,
        audio_start_in_s=0.0,
        audio_end_in_s=seconds,
        num_waveforms_per_prompt=NUM_WAVEFORMS,
        generator=g,
        initial_audio_waveforms=wav,
        initial_audio_sampling_rate=pipe.vae.config.sampling_rate,
    ).audios


# ---------- main workflow ----------
if __name__ == "__main__":

    pipe = load_pipe()

    data = pd.read_csv('sound_prompts2.csv')
    print('data loaded')

    line: str = ''

    for index in range(1000):
        line = str(data['prompt_text'][index]) # update prompt text every three files.

        for k in range(3):
            # Load input wav
            try:
                input_wav = f"/mnt/data2/evanmm3/reword/sound_{index}_{k}.wav"   
                src_wav = read_wav(input_wav, pipe.vae.config.sampling_rate)
            except:
                print(f"File not found: {input_wav}, moving on...")
                continue

            print(f'({index})\n{input_wav}: {line}')

            # save latents
            latents = encode_audio(pipe, src_wav)                
            torch.save(latents.cpu(), f"/mnt/data2/evanmm3/ti_latent2/latent_{index}_{k}.pt")

            # create variation
            variations = remix(pipe, src_wav, line, seed=42*k)
            print(variations.squeeze(0).T.shape)
            output = variations.squeeze(0).T.to(torch.float32).cpu().numpy()
            sf.write(f"/mnt/data2/evanmm3/ti_reword/sound_{index}_{k}.wav", output, pipe.vae.config.sampling_rate)

    print("Done!")
