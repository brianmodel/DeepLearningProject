import torch
from torch import nn
from torchaudio import transforms
from utils.audio_utils import AudioHelper, AudioTransformations

from utils import SAMPLE_RATE

class CNNPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.spec = transforms.MelSpectrogram(SAMPLE_RATE, n_fft=1024, hop_length=None, n_mels=64)
    
    def forward(self, audio):
        # rechan = AudioHelper.rechannel(audio, 1)
        # sgram = AudioTransformations.spectrogram(rechan, n_mels=64, n_fft=1024, hop_len=None)
        # return sgram
        spec = self.spec(audio)
        if self.training:
            spec = AudioTransformations.spectrogram_augment(spec)
        return spec
    
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.pipeline = CNNPipeline()
        self.conv = nn.Conv2d(1, 8, kernel_size=(5,5), stride=(3,3), padding=(2,2))
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(4,4)

    def forward(self, x):
        x = self.pipeline(x)

        if len(x.shape) == 3:
            x = torch.unsqueeze(x, dim=1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.bn(x)
        x = self.pool(x)
        
        x = x.view((x.shape[0], -1))
        return x
