import torch
from torch import nn
from torch.nn import functional as F
# from torchvision.models import resnet18, ResNet18_Weights
from torch import Tensor


class CRNN_Network(torch.nn.Module):
    def __init__(self):
        super(CRNN_Network, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...)
        self.pool1 = nn.MaxPool2d()
        self.conv2 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...)
        self.pool2 = nn.MaxPool2d()
        self.conv3 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...)
        self.conv4 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...)
        self.pool3 = nn.MaxPool2d()
        self.conv5 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...)
        self.batch_norm1 = nn.BatchNorm2d()
        self.conv6 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...)
        self.batch_norm2 = nn.BatchNorm2d()
        self.pool4 = nn.MaxPool2d()
        self.conv7 = nn.Conv2d(in_channels=..., out_channels=..., kernel_size=...)
        # Map to sequence
        self.lstm1 = nn.LSTM(bidirectional=True)
        self.lstm2 = nn.LSTM(bidirectional=True)
        # Transcription
        self.dense = nn.Linear()

        self.layers = nn.Sequential(
            self.conv1,
            self.pool1,
            self.conv2,
            self.pool2,
            self.conv3,
            self.conv4,
            self.pool3,
            self.conv5,
            self.batch_norm1,
            self.conv6,
            self.batch_norm2,
            self.pool4,
            self.conv7,
            self.lstm1,
            self.lstm2,
            self.dense
        )

    def forward(self, input):
        return self.layers(input)