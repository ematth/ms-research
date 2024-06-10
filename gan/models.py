import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.utils import weight_norm
from librosa.filters import mel
import numpy as np

### REFERENCE: 
# MELGAN
# https://github.com/descriptinc/melgan-neurips/tree/master
###


class Audio2Mel(nn.Module):
    def __init__(
            self,
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
            fmax: float = None
        ) -> None:
        
        super().__init__()
        mel_basis = mel(sample_rate, num_fft, num_mel_chan, fmin, fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.register_buffer("window", window(win_length).float())
        self.num_fft = num_fft
        self.hop_size = hop_size
        self.win_length = win_length
        self.sample_rate = sample_rate
        self.num_mel_chan = num_mel_chan

    def forward(self, audio) -> torch.Tensor:
        """Compute the MEL spectrogram for an audio file.

        Args:
            audio (torch.Tensor): audio tensor (C x S) containing C channels of S samples each.

        Returns:
            torch.Tensor: MEL spectrogram representation of the audio.
        """

        PAD = (self.num_fft - self.hop_size) // 2 # Padding to apply to audio Tensor.
        audio = F.pad(
            input=audio, 
            pad=(PAD, PAD), 
            mode='reflect'
            ).squeeze(1)
        fft = torch.stft(
            input=audio,
            n_fft=self.num_fft,
            hop_length=self.hop_size,
            win_length=self.win_length,
            window=self.window,
            center=False,
            normalized=False
        )
        real, imag = fft.unbind(-1) # unbind complex Tensor into Re, Im parts.
        mel_out = torch.matmul(self.mel_basis, torch.sqrt((real ** 2) + (imag ** 2)))
        log_mel = torch.log10(torch.clamp(mel_out, min=1e-5, max=None))
        return log_mel


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


# TODO: DiscriminatorS (multi-scale Discriminator) akin to Mel-GAN for music.
class DiscriminatorS(nn.Module):
    def __init__():
        pass

    def forward():
        pass


# TODO: DiscriminatorP (multi-period Discriminator) akin to HiFi-GAN for music.
class DiscriminatorP(nn.Module):
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