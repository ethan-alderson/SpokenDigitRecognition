
import numpy as np
import torch
from torch.utils.data import DataLoader

# Just to visualize spectrograms
import matplotlib.pyplot as plt

# opens audio files in python format, used here to convert audio into spectrograms
import librosa

import pandas as pd

from utils import PreprocessingUtils
from audiodataset import AudioDataset


# IGNORE SCALING WARNINGS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# establish a utility instance with padding length 15
u = PreprocessingUtils(15)

# separate the data into training and testing
train, test = u.split_data()

train_dataset = AudioDataset("SpokenDigitRecognition/Data/train_data.csv")
test_dataset = AudioDataset("SpokenDigitRecognition/Data/test_data.csv")

train_loader = DataLoader(train_dataset, batch_size=5)
test_loader = DataLoader(test_dataset, batch_size=5)

# print the length of a single batch (5)
print(len(next(iter(train_loader))[0]))