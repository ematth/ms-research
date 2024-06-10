import torch
import torch.utils.data

class MusicNet(torch.utils.data.Dataset):
    def __init__(self, data):
        self.audio = data

    def __getitem__(self, index):
        pass

    def __len__(self):
        return len(self.audio)

