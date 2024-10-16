# Toy Model Notes:

- Model: Facebook/musicgen-small/large [link](https://huggingface.co/facebook/musicgen-small)
    - Condition with prior melody, using text to differentiate style-wise.
- Evaluation metric: AudioSimilarity [link](https://github.com/markstent/audio-similarity/blob/main/audio_similarity/audio_similarity.py)
    - Temporal, Rhythmic, Chromatic, Spectral comparisons between baseline audio and conditionally-generated audio.


### Misc. Thoughts:
- FRECHET audio distance
- music clap
- KL distance (bw probabilities)
- mixing style transfer


### TODO:
- Musicgen, StableAudio setup
    - Find more Text-to-Audio models?
- Data collection
    - Discord bot, website for scalability
    - Manually collect 100+ generated results
    - Auto generate files from text prompt database, then manually pick, (myself, Mechanical Turk)
- Store results in /mnt/ drive mapping
- Text-to-Text Neural Network.
    - i.e. prompt to prompt, goal of prompt improvement given CLAP, KL distance, FRECHET.
