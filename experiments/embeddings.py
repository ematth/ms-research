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


# ---------- utility helpers ----------
def load_pipe(dtype=torch.float16, device="cuda"):
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


def remix(pipe, wav, *, steps=200, seed=42, seconds=10.0):
    wav = wav.to(pipe.device, dtype=pipe.vae.dtype)  # <- added
    g   = torch.Generator(device=pipe.device).manual_seed(seed)
    return pipe(
        prompt='flowing water',
        num_inference_steps=steps,
        guidance_scale=1.0,
        audio_start_in_s=0.0,
        audio_end_in_s=seconds,
        num_waveforms_per_prompt=1,
        generator=g,
        initial_audio_waveforms=wav,
        initial_audio_sampling_rate=pipe.vae.config.sampling_rate,
    ).audios


# ---------- main workflow ----------
if __name__ == "__main__":
    INPUT_WAV = "sounds/sound_0_1.wav"

    pipe = load_pipe()                                   # 1️⃣
    src_wav = read_wav(INPUT_WAV, pipe.vae.config.sampling_rate)

    latents = encode_audio(pipe, src_wav)                # 2️⃣
    torch.save(latents.cpu(), "input_latents.pt")

    recon = decode_latents(pipe, latents).cpu()          # 3️⃣
    # ----------------- save reconstructed -----------------
    recon_32 = recon.squeeze().T.to(torch.float32).cpu().numpy()
    sf.write("sounds/reconstructed.wav", recon_32,
            pipe.vae.config.sampling_rate)

    variation = remix(pipe, src_wav)[0].cpu()            # 4️⃣
    # ----------------- save variation -----------------
    var_32 = variation.T.to(torch.float32).cpu().numpy()
    sf.write("sounds/variation.wav", var_32,
            pipe.vae.config.sampling_rate)

    print("Done – check reconstructed.wav and variation.wav")
