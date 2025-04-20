import torch
from diffusers.models.transformers.stable_audio_transformer import StableAudioDiTModel
import numpy as np
import soundfile as sf
from torchaudio.functional import resample
import torchaudio

from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
import copy

import sys
sys.path.append('./src/')
from StableAudioFeaturePipeline import StableAudioFeaturePipeline


def main(initial_audio: str):

    transformer = StableAudioDiTModel.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        subfolder="transformer",
        torch_dtype=torch.float16,
        num_readout_blocks=20
    )

    pipe = StableAudioFeaturePipeline.from_pretrained(
        "stabilityai/stable-audio-open-1.0",
        torch_dtype=torch.float16,
        transformer=transformer,
    )

    pipe.enable_model_cpu_offload()

    prompt = "The sound of a hammer hitting a wooden surface."
    # negative_prompt = "Low quality."

    # set the seed for generator
    generator = torch.Generator("cuda:0").manual_seed(0)

    # x, sr = sf.read(initial_audio)
    x, sr = torchaudio.load(initial_audio, normalize=True)
    x = x.to("cuda:0")

    # x = x.mean(dim=0)  # convert to mono
    # x = x / torch.max(torch.abs(x))  # normalize to [-1, 1]

    # x = resample(x, orig_freq=sr, new_freq=pipe.vae.sampling_rate).to("cuda:0")

    print(x.unsqueeze(1).shape, pipe.vae.sampling_rate)

    # print('SR STUFF:', 2097152, pipe.vae.sampling_rate)

    # x = x.resize(100, x.shape[0])

    # run the generation

    # temp = pipe.__call__(
    #     prompt,
    #     # negative_prompt=negative_prompt,
    #     num_inference_steps=200,
    #     audio_end_in_s=10.0,
    #     num_waveforms_per_prompt=1,
    #     generator=generator,
    #     initial_audio_waveforms=x.unsqueeze(1).type(pipe.vae.dtype),
    #     initial_audio_sampling_rate=pipe.vae.sampling_rate
    # )

    feats, _  = pipe.extract_feats(
        prompt,
        # negative_prompt=negative_prompt,
        num_inference_steps=200,
        audio_end_in_s=10.0,
        num_waveforms_per_prompt=1,
        generator=generator,
        initial_audio_waveforms=x.unsqueeze(1).type(pipe.vae.dtype),
        initial_audio_sampling_rate=pipe.vae.sampling_rate,
        timestep_to_collect=0
    )

    print(f'feats -> {feats.shape}')
    print("TEST COMPLETE")

if __name__ == "__main__":
    main(initial_audio=sys.argv[1])