
import torch
from torch.utils.data import DataLoader

from Model.model import AudioClassifier
from Model.train_model import train_model
from Model.test_model import test_model


from Tools.utils import PreprocessingUtils
from Tools.audiodataset import AudioDataset

# IGNORE SCALING WARNINGS
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# establish a utility object with standardization length 10
u = PreprocessingUtils(15)

# separate the data into training and testing
training, testing = u.split_data()

train_dataset = AudioDataset("SpokenDigitRecognition/Datasheets/train_data.csv")
test_dataset = AudioDataset("SpokenDigitRecognition/Datasheets/test_data.csv")

train_loader = DataLoader(train_dataset, batch_size=5)
test_loader = DataLoader(test_dataset, batch_size=5)
model = AudioClassifier().cuda()

# How fast the model learns, was previously too large and overshot local minima
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

cost = torch.nn.CrossEntropyLoss()

# baseline 25 epoch training showing >90% accuracy
for i in range(25):
    print(i + 1)
    train_model(train_loader, model, cost, optimizer, 'cuda') 
    
# disables the dropout layers
model.prepToTest()

test_model(test_loader, model, 'cuda')