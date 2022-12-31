
import numpy as np
import torch
from torch.utils.data import DataLoader

# Just to visualize spectrograms
import matplotlib.pyplot as plt

# opens audio files in python format, used here to convert audio into spectrograms
import librosa

# grab my data set class
import SpokenDigitRecognition.audiodataset as audiodataset

audio = librosa.load("SpokenDigitRecognition/recordings/0_george_0.wav", sr=8000)
waveform = audio[0]

# -- plot the waveform
# plt.plot(waveform)
# plt.show()

melspec = librosa.feature.melspectrogram(y = waveform, sr=8000)

plt.imshow(librosa.power_to_db(melspec, ref=np.max))

dataset = audiodataset.AudioDataset()


plt.imshow(librosa.power_to_db(dataset.__getitem(10)[0], ref=np.max))
plt.show()