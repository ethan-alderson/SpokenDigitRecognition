
import numpy as np
import torch
from torch.utils.data import DataLoader

# Just to visualize spectrograms
import matplotlib.pyplot as plt

# opens audio files in python format, used here to convert audio into spectrograms
import librosa


audio = librosa.load("./recordings/0_george_0.wav")


