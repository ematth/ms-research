from transformers import pipeline
import soundfile as sf
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, MusicgenForConditionalGeneration
from models import musicgen
import numpy as np
import sys


if __name__ == '__main__':
    try:
        model: str = sys.argv[1]
        model in ['musicgen', 'musicgenmelody', 'stableaudio'] == True 
    except: 
        ValueError(f'Failure in provided model type')

    try:
        text_prompt: str | list[str] | list[list[str]] = sys.argv[2]
        (len(text_prompt) > 0) == True
    except:
        ValueError(f'Failure in provided text prompt')

    names = getattr(musicgen, model)(text_prompt=text_prompt)
    print(f'generated {names}.')

    