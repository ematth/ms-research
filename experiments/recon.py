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

global DEVICE
if torch.cuda.is_available():
    DEVICE = torch.device("cuda:6")
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


# ---------- main workflow ----------
if __name__ == "__main__":

    pipe = load_pipe()

    data = pd.read_csv('sound_prompts.csv')
    print('data loaded')

    line: str = ''

    diff = np.zeros((3000, 2))

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
            
            # Load respective latent
            latent = torch.load(f'/mnt/data2/evanmm3/ti_latent2/latent_{index}_{k}.pt').to(DEVICE)
            # decode latent to wav
            latent_wav = decode_latents(pipe, latent)
            # compute MSE between the two wav files
            length = min(src_wav.shape[2], latent_wav.shape[2])
            src_wav = src_wav[:, :, :length].to(DEVICE)
            latent_wav = latent_wav[:, :, :length].to(DEVICE)
            mse = torch.nn.functional.mse_loss(src_wav, latent_wav)
            l2 = torch.dist(src_wav, latent_wav)
            # save the results
            diff[index*3 + k][0] = mse.item()
            diff[index*3 + k][1] = l2.item()
            print(diff[index*3 + k])

    # save the results to a csv file
    df = pd.DataFrame(diff, columns=['mse', 'l2'])
    df.to_csv('ti_recon2.csv', index=False, header=True)
    print("Done!")
