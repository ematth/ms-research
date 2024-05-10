import torch
from torch import nn
from torch.nn import functional as F
# from torchvision.models import resnet18, ResNet18_Weights
from torch import Tensor


class CRNN_Network(torch.nn.Module):
    def __init__(self, in_chans):
        super(CRNN_Network, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(in_channels=in_chans, out_channels=64, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.pool1 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.pool2 = nn.MaxPool2d(kernel_size=(2,2), stride=2)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.pool3 = nn.MaxPool2d(kernel_size=(1,2), stride=2)
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.batch_norm1 = nn.BatchNorm2d()
        self.conv6 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False)
        self.batch_norm2 = nn.BatchNorm2d()
        self.pool4 = nn.MaxPool2d(kernel_size=(1,2), stride=2)
        self.conv7 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(2,2), stride=1, padding=0, bias=False)
        # Map to sequence
        self.lstm1 = nn.LSTM(hidden_size=256, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size=256, bidirectional=True)
        # Transcription
        self.dense = nn.Linear(in_features=256, out_features=256)

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