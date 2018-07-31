import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F


class observe_cnn(nn.Module):
    def __init__(self, input_channel, hidden_channel, filter_size):
        super(observe_cnn, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, filter_size * filter_size * input_channel, kernel_size=5, stride=1, padding=2),
            )

    def forward(self, input):
        x = self.layer1(input)
        filts = self.layer2(x)
        return filts

class reference_cnn(nn.Module):
    def __init__(self, input_channel, hidden_channel, filter_size):
        super(DecoderRNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, hidden_size, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),)
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, filter_size * filter_size * input_channel, kernel_size=5, stride=1, padding=2),
            )

    def forward(self, input):
        x = self.layer1(input)
        filts = self.layer2(input)
        return filts
