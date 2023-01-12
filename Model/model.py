
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


# torch imports for classifer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# gpu stuff for speed
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using {device} device')

# inherits the PyTorch nn module, 
class AudioClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # the four convolutional layers of the network
        # powers of two are for aesthetic reasons only
        self.conv1 = nn.Sequential(
            # 1 in 32 features out
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=8, stride=1, padding=2),
            # rectify linear unit, if output is positive it produces the output, or else it produces 0,
            # TLDR ReLU makes training easier and performance better
            nn.ReLU(),
            # batch normalization keeps the mean of the layer's inputs 0 to prevent skew
            # shape of normalization should be the same as the number of output channels
            nn.BatchNorm2d(32),
            # Concentrates the outputs in pairs of two, takes the max of each pair, concentrating and shrinking the data
            nn.MaxPool2d(kernel_size=2)
        )

        # note that the # of output channels of one layer matches the # of input channels of the next 
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=8, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2)
            )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2)
            )

        self.conv4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=256, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2)
            )

        # takes any parameter at flattens it to one dimension
        self.flatten=nn.Flatten()

        # 4608 is the size of the flattened data, final output is 10, for digits 0-9
        self.linear1 = nn.Linear(in_features=4608, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=64)
        self.linear3 = nn.Linear(in_features=64, out_features=32)
        self.linear4 = nn.Linear(in_features=32, out_features=10)
    
        # a probability function that outputs the highest float of the final outputs, i.e. the digit it is guessing
        self.output = nn.Softmax(dim = 1)

    # tells the network to move forward with the data, inherited by nn.module
    def forward(self, input_data):
        x1 = self.conv1(input_data)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x = self.flatten(x4)

        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        logits = self.linear4(x)

        output = self.output(logits)

        return output

model = AudioClassifier()
model.to(device)
print(model)