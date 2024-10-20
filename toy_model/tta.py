from transformers import pipeline
import soundfile as sf
# from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, MusicgenForConditionalGeneration
import models
import numpy as np
import sys

MODELS: list[str] = ['musicgen_small', 'musicgen_medium', 'musicgen_large', 'stableaudio_open']

if __name__ == '__main__':
    model: str
    try:
        model = sys.argv[1]
    except: 
        print(f'Failure in model parameter... falling back on {MODELS[0]}')
    model = MODELS[0]

    if not (model in MODELS):
        raise ValueError(f'Model \"{model}\" does not exist.')

    print(f'model: {model}')

    text_prompt: str = ''
    try:
        text_prompt: str | list[str] | list[list[str]] = sys.argv[2]
    except:
        ValueError(f'Failure in provided text prompt')

    if not (len(text_prompt) > 0):
        raise ValueError(f'Unusable or no text prompt provided.')

    print(f'prompt: {text_prompt}')

    names = getattr(models, model)(text_prompt=text_prompt)
    print(f'generated {names}.')

    