
import librosa
import numpy as np
import pandas as pd

class PreprocessingUtils:

    def __init__(self, AUDIO_LENGTH):
        
        self.AUDIO_LENGTH = AUDIO_LENGTH
    
# PREPROCESSING UTILITY

    def trim_and_pad(self, audio: np.array):
        """Trims or pads the audio files for each input matrix to be the same size

        Args:
            audio (np.array): numpy array of the audio file data

        Returns:
            np.array: trimmed and padded audio 
        """
        length_diff = self.AUDIO_LENGTH - audio.shape[1]

        if length_diff > 0:
            audio = np.pad(audio, pad_width=[(0,0), (0, length_diff + 1)], mode = 'constant', constant_values = 0)
        
        return audio[:, :self.AUDIO_LENGTH]
    
    def load_audio(self, filename):
        """ Loads audio and converts it into a spectrogram

        Args:
            filename (str): The name of the file
        """

        audio = librosa.load(f'SpokenDigitRecognition/Recordings/{filename}', sr=8000)
        return self.trim_and_pad(librosa.feature.melspectrogram(y = audio[0], sr=8000, n_fft=16384))


    def split_data(self):
        # 80% for training and 20% for testing
        files = pd.read_csv("SpokenDigitRecognition/Data/train.csv")
        train = files.iloc[:int(3000*0.8)]
        test  = files.iloc[int(3000*0.8):]
        train.to_csv('SpokenDigitRecognition/Data/train_data.csv')
        test.to_csv('SpokenDigitRecognition/Data/test_data.csv')

        return train, test