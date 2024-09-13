from transformers import pipeline
import scipy
import soundfile as sf
from transformers import AutoProcessor, MusicgenMelodyForConditionalGeneration
import torch

# device = 1;
# torch.cuda.device(1)
# torch.cuda.set_device(1)

MAX_TOKENS: int = 1503 # 30 seconds of tokens

def model_script(audio_name: str = '', time: int = 5, text_prompt: list[str] = ["classical piano"], outname: str = 'output') -> None:

    # Seconds to tokens relation
    time_tokens = lambda t: t * 50

    # load sound, if audio prompt given
    if len(audio_name) < 1:
        rate, data = scipy.io.wavfile.read(f'sounds/{audio_name}.wav')
        data = torch.Tensor(data)[rate * time:, :]#.cuda(1)
        print(rate, data.shape)

    # Processor and model
    processor = AutoProcessor.from_pretrained("facebook/musicgen-melody", low_cpu_mem_usage=True)
    print('processor complete')
    model = MusicgenMelodyForConditionalGeneration.from_pretrained("facebook/musicgen-melody")
    print('model complete')

    print('preprocessing')

    if len(audio_name) < 1:
        inputs = processor( # only text prompt
            text=text_prompt,
            padding=True,
            return_tensors="pt",
        ).to('cuda:0')
    else:
        inputs = processor( # text and audio prompts
            audio=data,
            sampling_rate=rate,
            text=text_prompt,
            padding=True,
            return_tensors="pt",
        ).to('cuda:0')
        

    print('postprocessing')

    model.to('cuda:1')

    for k in inputs.keys():
        inputs[k] = inputs[k].to(f'cuda:1')

    print('generating...')
    audio_values = model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=time_tokens(time), use_cache=True)
    print('generated')

    sampling_rate = model.config.audio_encoder.sampling_rate
    output_name = f"sounds/{outname}.wav"
    sf.write(output_name, audio_values[0].T.cpu().numpy(), sampling_rate)
    return output_name


if __name__ == '__main__':
    model_script()