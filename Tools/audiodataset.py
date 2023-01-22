

import pandas as pd

from Tools.utils import PreprocessingUtils

class AudioDataset():

    def __init__(self, file_csv):

        filenames_and_labels = pd.read_csv(file_csv)
        self.files = filenames_and_labels["file_name"]
        self.labels = filenames_and_labels["label"]

    def __getitem__(self, index):

        u = PreprocessingUtils(15)

        label = self.labels.iloc[index]
        melspec = u.load_audio(self.files.iloc[index])

        return melspec, label

    def __len__(self):
        return len(self.labels)
