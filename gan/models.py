import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm
from librosa.filters import mel
from librosa.util import normalize
import numpy as np
from scipy.io.wavfile import read
import random

WAV_CONST = 32768.0

### REFERENCE: 
# MELGAN
# https://github.com/descriptinc/melgan-neurips/tree/master
# HIFI-GAN
# https://github.com/jik876/hifi-gan/tree/master
###

mel_banks = {}
windows = {}

class MelDataset(nn.Module):
    def __init__(
            self,
            # Training Files
            audio_files,
            # Segment size
            segment_size: int = 8192,
            # Number of FFT components
            num_fft: int = 1024,
            # Hop size of the FFT window
            hop_size: int = 256,
            # Type of window to use
            window: function = torch.hann_window,
            # Length of the FFT window
            win_length: int = 1024,
            # Sample rate of the input sound, (typically 22050 or 44100)
            sample_rate: int = 44100,
            # number of channels in the MEL spectrogram
            num_mel_chan: int = 80,
            # Mel: Minimum frequency (Hz)
            fmin: float = 0.0,
            # Mel: Maximum frequency (Hz)
            fmax: float = None,
            # Shuffle provided audio files
            shuffle: bool = False
        ) -> None:
        
        super().__init__()
        random.seed(1337)
        if shuffle: random.shuffle(self.audio_files)
        mel_basis = mel(sample_rate, num_fft, num_mel_chan, fmin, fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window(win_length).float())
        self.num_fft = num_fft
        self.hop_size = hop_size
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.num_mel_chan = num_mel_chan
        self.audio_files = audio_files
        self.segment_size = segment_size

    def __getitem__(self, index) -> torch.Tensor:
        filename = self.audio_files[index]
        ## assume no caching yet
        sample_rate, audio = read(filename)
        audio /= WAV_CONST
        audio = normalize(audio) * 0.95 # assume no fine-tuning, yet
        audio = torch.floatTensor(audio).unsqueeze(0)

        # assume no fine-tuning, yet
        if audio.size(1) >= self.segment_size:
            max_audio_start = audio.size(1) - self.segment_size
            audio_start = random.randint(0, max_audio_start)
            audio = audio[:, audio_start:audio_start + self.segment_size]
        else:
            audio = F.pad(audio, (0, self.segment_size - audio.size(1)), 'constant')

        # Mel spectrogram conversion
        global mel_banks, windows
        if self.fmax not in mel_banks:
            mel = mel(self.sample_rate, self.n_fft, self.num_mels, self.fmin, self.fmax)
            mel_banks[f'{str(self.fmax)}_{str(audio.device)}'] = torch.from_numpy(mel).float().to(audio.device)
            windows[str(audio.device)] = torch.hann_window(self.win_length).to(audio.device)

        PAD = (self.num_fft - self.hop_size) // 2
        audio = F.pad(audio.unsqueeze(1), (PAD, PAD), mode='reflect')
        audio = audio.unsqueeze(1)

        spectrogram = torch.stft(audio, 
                                self.num_fft, 
                                hop_length=self.hop_size,
                                win_length=self.win_length,
                                window=self.window,
                                center=False,
                                pad_mode='reflect',
                                normalized=False,
                                onesided=True)
        spectrogram = torch.sqrt(spectrogram.pow(2).sum(-1)+(1e-9))
        spectrogram = torch.matmul(mel_banks[f'{str(self.fmax)}_{str(audio.device)}'], spectrogram)
        spectrogram = torch.log10(torch.clamp(spectrogram, min=1e-5)) # dynamic range compression. Maybe use log10?

        return (spectrogram.squeeze(), audio.squeeze(0), filename)

    def __len__(self):
        return len(self.audio)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2),
            nn.ReflectionPad1d(padding=dilation),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=3, dilation=dilation)),
            nn.LeakyReLU(negative_slope=0.2),
            weight_norm(nn.Conv1d(dim, dim, kernel_size=1))
        )
        self.shortcut = weight_norm(nn.Conv1d(dim, dim, kernel_size=1))

    def forward(self, x): return self.shortcut(x) + self.block(x)


# TODO: Generator akin to Mel-GAN for music.
class Generator(nn.Module):
    def __init__():
        pass

    def forward():
        pass


# TODO: Discriminator akin to Mel-GAN for music.
class Discriminator(nn.Module):
    def __init__():
        pass

    def forward():
        pass

def feature_loss():
    pass


def generator_loss():
    pass


def discriminator_loss():
    pass