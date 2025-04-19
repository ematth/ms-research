import torch
from transformers import ClapProcessor, ClapModel
from diffusers import StableAudioPipeline
import soundfile as sf
import accelerate

# 1. Load CLAP and extract embedding
print('1')
clap_model = ClapModel.from_pretrained("laion/clap-htsat-fused").to("cuda")
processor = ClapProcessor.from_pretrained("laion/clap-htsat-fused")

# 2. Load StableAudio pipeline :contentReference[oaicite:0]{index=0}
print('2')
pipe = StableAudioPipeline.from_pretrained("stabilityai/stable-audio-open-1.0", torch_dtype=torch.float16)
pipe = pipe.to("cuda")
pipe.enable_sequential_cpu_offload()
# save some GPU memory if you need to
#pipe.enable_vae_slicing()

generator = torch.Generator("cuda").manual_seed(0)

# 3. Read your input .wav and get its CLAP embedding
print('3')
audio_array, sr = sf.read("sounds/sound_0_1.wav")
inputs = processor(audios=audio_array, sampling_rate=sr, return_tensors="pt")
inputs = {k: v.to("cuda") for k, v in inputs.items()}
with torch.no_grad():
    audio_features = clap_model.get_audio_features(**inputs)
audio_embeds = audio_features.audio_embeds  # shape [1, embed_dim]

# 4. Generate new audio conditioned on that embedding
print('4')
out = pipe(
    prompt_embeds=audio_embeds,     # use your CLAP embedding directly
    num_inference_steps=200,         # quality/speed trade‑off
    audio_end_in_s=10.0,
    num_waveforms_per_prompt=1,      # number of audio samples to generate
    generator=generator,            # length in seconds
)

# out.audios is a list of tensors; take the first one
generated = out.audios[0].T.float().cpu().numpy()

# 5. Save it
sf.write("sounds/generated.wav", generated, samplerate=pipe.vae.sampling_rate)
