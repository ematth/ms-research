from transformers import pipeline
import soundfile as sf
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, MusicgenForConditionalGeneration
import numpy as np
import sys
import torch

# device = 1;
# torch.cuda.device(1)
# torch.cuda.set_device(1)

MAX_TOKENS: int = 1503 # 30 seconds of tokens

def musicgen_small(
        text_prompt: str | list[str] | list[list[str]] = "white noise", 
        time: int = 15, 
        outname: str = 'output',
        num_outputs: int = 2) -> list[str]:

    # Processor and model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-small", low_cpu_mem_usage=True)
    print('processor complete')
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-small")
    print('model complete')

    print('preprocessing')

    inputs = processor( # only text prompt
        text=text_prompt,
        padding=True,
        return_tensors="pt",
    ).to('cuda:0')

    model.to('cuda:1')

    for k in inputs.keys():
        inputs[k] = inputs[k].to(f'cuda:1')

    print('generating...')
    time_tokens = lambda t: t * 50

    names = []
    for _ in range(num_outputs):
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=time_tokens(time), use_cache=True)

        sampling_rate = model.config.audio_encoder.sampling_rate
        output_name = f"sounds/{outname}-{np.random.randint(0, 1000, None)}.wav"
        names.append(output_name)
        sf.write(output_name, audio_values[0].T.cpu().numpy(), sampling_rate)

    return names


# musicgen_medium
def musicgen_medium(
        text_prompt: str | list[str] | list[list[str]] = "white noise", 
        time: int = 15, 
        outname: str = 'output',
        num_outputs: int = 2) -> list[str]:

    # Processor and model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-medium", low_cpu_mem_usage=True)
    print('processor complete')
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-medium")
    print('model complete')

    print('preprocessing')

    inputs = processor( # only text prompt
        text=text_prompt,
        padding=True,
        return_tensors="pt",
    ).to('cuda:0')

    model.to('cuda:1')

    for k in inputs.keys():
        inputs[k] = inputs[k].to(f'cuda:1')

    print('generating...')
    time_tokens = lambda t: t * 50

    names = []
    for _ in range(num_outputs):
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=time_tokens(time), use_cache=True)

        sampling_rate = model.config.audio_encoder.sampling_rate
        output_name = f"sounds/{outname}-{np.random.randint(0, 1000, None)}.wav"
        names.append(output_name)
        sf.write(output_name, audio_values[0].T.cpu().numpy(), sampling_rate)

    return names


def musicgen_large(
        text_prompt: str | list[str] | list[list[str]] = "white noise", 
        time: int = 15, 
        outname: str = 'output',
        num_outputs: int = 2) -> list[str]:

    # Processor and model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-large", low_cpu_mem_usage=True)
    print('processor complete')
    model = MusicgenForConditionalGeneration.from_pretrained("facebook/musicgen-large")
    print('model complete')

    print('preprocessing')

    inputs = processor( # only text prompt
        text=text_prompt,
        padding=True,
        return_tensors="pt",
    ).to('cuda:0')

    model.to('cuda:1')

    for k in inputs.keys():
        inputs[k] = inputs[k].to(f'cuda:1')

    print('generating...')
    time_tokens = lambda t: t * 50

    names = []
    for _ in range(num_outputs):
        audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=time_tokens(time), use_cache=True)

        sampling_rate = model.config.audio_encoder.sampling_rate
        output_name = f"sounds/{outname}-{np.random.randint(0, 1000, None)}.wav"
        names.append(output_name)
        sf.write(output_name, audio_values[0].T.cpu().numpy(), sampling_rate)

    return names


# StableAudio Open 1.0 (Audiosparx 1.0)
def stableaudio_open(
        text_prompt: str | list[str] | list[list[str]] = "white noise", 
        time: int = 15, 
        outname: str = 'output',
        num_outputs: int = 2) -> list[str]:

    from diffusers import StableAudioPipeline
    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    generator = torch.Generator("cuda").manual_seed(0)

    names = []
    for _ in range(num_outputs):
        audio = pipe(
            prompt=text_prompt,
            negative_prompt="low quality",
            num_inference_steps=200,
            audio_end_in_s=time,
            num_waveforms_per_prompt=3,
            generator=generator,
        ).audios

        output_name = f"sounds/{outname}-{np.random.randint(0, 1000, None)}.wav"
        names.append(output_name)
        sf.write("{output_name}.wav", audio[0].T.float().cpu().numpy(), pipe.vae.sampling_rate)

    return names


# AudioLDM
def audioldm(
        text_prompt: str | list[str] | list[list[str]] = "white noise", 
        time: int = 15, 
        outname: str = 'output',
        num_outputs: int = 2) -> list[str]:
    from diffusers import AudioLDMPipeline
    import scipy

    repo_id = "cvssp/audioldm-s-full-v2"
    pipe = AudioLDMPipeline.from_pretrained(repo_id, torch_dtype=torch.float16)
    pipe = pipe.to("cuda")

    names = []
    for _ in range(num_outputs):
        prompt = "Techno music with a strong, upbeat tempo and high melodic riffs"
        audio = pipe(text_prompt, num_inference_steps=100, audio_length_in_s=time).audios[0]

        # save the audio sample as a .wav file
        output_name = f"sounds/{outname}-{np.random.randint(0, 1000, None)}"
        names.append(output_name)
        sf.write(f"{output_name}.wav", audio, 16000)
        #scipy.io.wavfile.write(f"{output_name}.wav", rate=16000, data=audio)

    return names


# Tango
# def tango(
#         text_prompt: str | list[str] | list[list[str]] = "white noise", 
#         time: int = 15, 
#         outname: str = 'output',
#         num_outputs: int = 2) -> list[str]:

#     from tango import Tango

#     tango = Tango("declare-lab/tango")

#     prompt = "An audience cheering and clapping"
#     audio = tango.generate(prompt)
#     sf.write(f"{prompt}.wav", audio, samplerate=16000)
#     IPython.display.Audio(data=audio, rate=16000)

#     names = []
#     for _ in range(num_outputs):
#         audio = pipe(
#             prompt=text_prompt,
#             negative_prompt="low quality",
#             num_inference_steps=200,
#             audio_end_in_s=time,
#             num_waveforms_per_prompt=3,
#             generator=generator,
#         ).audios

#         output_name = f"sounds/{outname}-{np.random.randint(0, 1000, None)}.wav"
#         names.append(output_name)
#         sf.write("{output_name}.wav", audio[0].T.float().cpu().numpy(), pipe.vae.sampling_rate)

#     return names


# Make-an-Audio 2
