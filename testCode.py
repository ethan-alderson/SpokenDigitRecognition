
import numpy as np
import torch
from torch.utils.data import DataLoader

# Just to visualize spectrograms
import matplotlib.pyplot as plt

# opens audio files in python format, used here to convert audio into spectrograms
import librosa

import pandas as pd

# grab my data set class
import audiodataset



audio = librosa.load("SpokenDigitRecognition/recordings/0_george_0.wav", sr=8000)
waveform = audio[0]

# -- plot the waveform
# plt.plot(waveform)
# plt.show()

melspec = librosa.feature.melspectrogram(y = waveform, sr=8000) 
plt.imshow(librosa.power_to_db(melspec, ref=np.max))

dataset = audiodataset.AudioDataset()
loader = DataLoader(dataset, batch_size=5)

# AUDIO_LENGTH = 15

# # PREPROCESSING UTILITY

# def trim_and_pad(audio: np.array):
#     """Trims or pads the audio files for each input matrix to be the same size

#     Args:
#         audio (np.array): numpy array of the audio file data

#     Returns:
#         np.array: trimmed and padded audio 
#     """
#     length_diff = AUDIO_LENGTH - audio.shape[1]

#     if length_diff > 0:
#         audio = np.pad(audio, pad_width=[(0,0), (0, length_diff + 1)], mode = 'constant', constant_values = 0)
        
#     return audio[:, :AUDIO_LENGTH]
    

# def load_audio(filename):
#     """ Loads audio and converts it into a spectrogram

#     Args:
#         filename (str): The name of the file
#     """

#     audio = librosa.load(f'SpokenDigitRecognition/recordings/{filename}', sr=8000)
#     return trim_and_pad(librosa.feature.melspectrogram(y = audio[0], sr=8000, n_fft=16384))
