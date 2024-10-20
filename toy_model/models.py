from transformers import pipeline
import soundfile as sf
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, MusicgenForConditionalGeneration
import numpy as np
import sys

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


# StableAudio Open


# StableAudio 1.0


# StableAudio 2.0

