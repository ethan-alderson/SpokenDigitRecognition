
import torch

def test_model(dataloader, model, device):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0

    cost = torch.nn.CrossEntropyLoss()

    # keep the model static via torch.no_grad
    with torch.no_grad():
        for batch, (X, Y) in enumerate(dataloader):
            X = X.view(-1, 1, 128, 15).float()
            X, Y = X.to(device), Y.to(device)
            pred = model(X)

            test_loss += cost(pred, Y).item()
            correct += (pred.argmax(1) == Y).type(torch.float).sum().item()

        test_loss /= size
        correct /= size

        print(f'\nTest Error:\nacc: {(100 * correct):0.1f}%, avg loss: {test_loss:>8f}\n')