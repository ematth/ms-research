import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet18, ResNet18_Weights
from torch import Tensor


class CRNN_Network(torch.nn.Module):
    def __init__(self):
        super(CRNN_Network, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d()
        self.pool1 = nn.MaxPool2d()
        self.conv2 = nn.Conv2d()
        self.pool2 = nn.MaxPool2d()
        self.conv3 = nn.Conv2d()
        self.conv4 = nn.Conv2d()
        self.pool3 = nn.MaxPool2d()
        self.conv5 = nn.Conv2d()
        self.batch_norm1 = nn.BatchNorm2d()
        self.conv6 = nn.Conv2d()
        self.batch_norm2 = nn.BatchNorm2d()
        self.pool4 = nn.MaxPool2d()
        self.conv7 = nn.Conv2d()
        # Map to sequence
        self.lstm1 = nn.LSTM(bidirectional=True)
        self.lstm2 = nn.LSTM(bidirectional=True)
        # Transcription
        self.dense = nn.Linear()

    def forward(input):
        input = 6