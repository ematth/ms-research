from transformers import pipeline
import soundfile as sf
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration, MusicgenForConditionalGeneration
from diffusers import StableAudioPipeline
import numpy as np
import sys, torch, torch.nn as nn
from tqdm import tqdm
import pandas as pd


def basic_generation(prompts: str) -> bool:
    MAX_TOKENS: int = 1500 # 30 seconds of tokens

    # Initialize pipe
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)

    pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
    pipe = pipe.to(device)

    generator = torch.Generator(device).manual_seed(0)

    print('pipe and generator loaded')

    # Load prompt data
    data = pd.read_csv(prompts)
    print('data loaded')

    NUM_WAVEFORMS: int = 3

    for index, line in enumerate(tqdm(data['prompt_text'][0:int(n:=sys.argv[2] if not None else 2)], desc="Generating audio", unit="file")):
        print(f'{index}: {line}')
        # run the generation
        audio = pipe(
            prompt=line,
            negative_prompt='Low quality.',
            num_inference_steps=200,
            audio_end_in_s=10.0,
            num_waveforms_per_prompt=NUM_WAVEFORMS,
            generator=generator,
        ).audios

        for i in range(NUM_WAVEFORMS):
            output = audio[i].T.float().cpu().numpy()
            #sf.write(f"/mnt/data2/evanmm3/basic/basic_{index}_{i}.wav", output, pipe.vae.sampling_rate)
            sf.write(f"sounds/{index}_{i}.wav", output, pipe.vae.sampling_rate)



if __name__ == "__main__":
    prompts = sys.argv[1]
    basic_generation(prompts)
    print('Audio generation completed.')