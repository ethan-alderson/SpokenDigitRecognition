
from torch.utils.data import DataLoader

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

