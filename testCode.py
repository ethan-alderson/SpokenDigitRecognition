
import numpy as np
import torch
from torch.utils.data import DataLoader

# Just to visualize spectrograms
import matplotlib.pyplot as plt

# opens audio files in python format, used here to convert audio into spectrograms
import librosa

import pandas as pd

import preprocessingutils as ppu

# 80% for training and 20% for testing
files = pd.read_csv("train.csv")
train = files.iloc[:int(3000*0.8)]
test  = files.iloc[int(3000*0.8)]
train.to_csv('train_data.csv')
test.to_csv('test_data.csv')