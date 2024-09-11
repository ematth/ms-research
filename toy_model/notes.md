# Toy Model Notes:

- Model: Facebook/musicgen-small/large [link](https://huggingface.co/facebook/musicgen-small)
    - Condition with prior melody, using text to differentiate style-wise.
- Evaluation metric: AudioSimilarity [link](https://github.com/markstent/audio-similarity/blob/main/audio_similarity/audio_similarity.py)
    - Temporal, Rhythmic, Chromatic, Spectral comparisons between baseline audio and conditionally-generated audio.


Misc. Thoughts:
- FRECHET audio distance

- music clap

- KL distance (bw probabilities)

- mixing style transfer
