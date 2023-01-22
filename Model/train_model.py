
import torch

# takes in the dataloader, the model, the cost function, and the optimizer object
def train_model(dataloader, model, loss, optimizer, device):
    model.train()
    # length of the data in the data set
    size = len(dataloader.dataset)

    cost = torch.nn.CrossEntropyLoss()

    # X: audio, Y: label
    for batch, (X, Y) in enumerate(dataloader):
        X = X.view(-1,1,128,15).float()
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        # prediction for the particular audio file
        pred = model(X)

        # Loss/Error
        loss = cost(pred, Y)

        # backpropagation
        loss.backward()

        optimizer.step()

        # PROGRESS MESSAGE
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}   [{current:>5d}/{size:>5d}]')


