
import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

import librosa

class AudioDataset():

    def __init__(self):

        speakers = ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yeweler']

        # all of the file names
        self.files = []

        # their corresponding labels
        self.labels = []

        for speaker in speakers:
            for digit in range(10):
                for trial in range(50):
                    self.files.append(f'{digit}_{speaker}_{trial}.wav')
                    self.labels.append(digit)

    def __getitem__(self, index):

        filename = f'SpokenDigitRecognition/recordings/{self.files[index]}'
        label = self.labels[index]

        audio = librosa.load(filename, sr=8000)

        # because audio is a tuple (waveform: array, int: sample rate), we call audio[0]
        melspec = librosa.feature.melspectrogram(y = audio[0], sr=8000)

        return melspec, label

    def __len__(self):
        return len(self.labels)
