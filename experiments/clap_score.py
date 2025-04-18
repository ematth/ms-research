
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
import fadtk


def compute_clap_score(audio_file: str, text_file: str) -> float:
    # Load the audio file
    audio = fadtk.audio.load_audio(audio_file)

    # Load the text file
    with open(text_file, 'r') as f:
        text = f.read()

    # Compute the CLAP score
    score = fadtk.score.compute_clap_score(audio, text)

    return score


if __name__ == '__main__':
    compute_clap_score(
        audio_file='/mnt/data2/evanmm3/reword2/reword2_0_0.wav',
        text_file='/mnt/data2/evanmm3/reword2/reword2_0_0.txt'
    )