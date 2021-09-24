from typing import Tuple
import pathlib
import os
import torch
import torchvision
from torchvision import transforms
import numpy as np


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv_1 = torch.nn.Conv2d(3, 6, 5)
        self.pool_1 = torch.nn.MaxPool2d(2, 2)
        self.pool_2 = torch.nn.MaxPool2d(2, 2)
        self.conv_2 = torch.nn.Conv2d(6, 16, 5)
        self.fc_1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc_2 = torch.nn.Linear(120, 84)
        self.fc_3 = torch.nn.Linear(84, 10)
        self.relu_1 = torch.nn.ReLU()
        self.relu_2 = torch.nn.ReLU()
        self.relu_3 = torch.nn.ReLU()
        self.relu_4 = torch.nn.ReLU()

    def forward(self, x):
        x = self.pool_1(self.relu_1(self.conv_1(x)))
        x = self.pool_2(self.relu_2(self.conv_2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = self.relu_3(self.fc_1(x))
        x = self.relu_4(self.fc_2(x))
        x = self.fc_3(x)
        return x

    
def load_datasets() -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Load datasets and make loaders."""
    transformer = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./sample_data', train=True, transform=transformer, download=True)
    test_set = torchvision.datasets.CIFAR10(root='./sample_data', train=False, transform=transformer, download=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True,
                                               pin_memory=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=12, pin_memory=True, num_workers=4)

    return train_loader, test_loader


def get_classes() -> dict:
    return {0: 'plane', 1: 'car', 2: 'bird', 3: 'cat', 4: 'deer',
            5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}


def load_pretrained_model(path: str = "../../tutorials/assets/test_model",
                          **kwargs):

    # Load model architecture.
    model = Net()

    if pathlib.Path(path).is_file():

        # Load model weights.
        model.load_state_dict(torch.load(path))

    else:

        print(f"Cannot find a torch model at {path}, given current {os.getcwd()} so" \
              " continue to train model.")

        # Load data.
        train_loader, test_loader = load_datasets()

        # Train and evaluate model.
        model = train_model(model=model.to(kwargs.get("device", None)),
                            train_data=train_loader,
                            test_data=test_loader,
                            device=kwargs.get("device", None),
                            epochs=20,
                            criterion=torch.nn.CrossEntropyLoss().to(kwargs.get("device", None)),
                            optimizer=torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9),
                            evaluate=True)

        # Save model.
        torch.save(model.state_dict(), path)

    # Model to GPU and eval mode.
    model.to(kwargs.get("device", None))
    model.eval()

    # Check test set performance.
    predictions, labels = evaluate_model(model=model, data=test_loader, device=kwargs.get("device", None))
    test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())
    print(f"Model test accuracy: {(100 * test_acc):.2f}%")

    return model


def train_model(model,
                train_data: torchvision.datasets,
                test_data: torchvision.datasets,
                device: torch.device,
                criterion: torch.nn,
                optimizer: torch.optim,
                epochs: int = 20,
                evaluate: bool = False):
    """Train torch model."""

    model.train()

    for epoch in range(epochs):

        for images, labels in train_data:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        # Evaluate model!
        if evaluate:
            predictions, labels = evaluate_model(model, test_data, device)
            test_acc = np.mean(np.argmax(predictions.cpu().numpy(), axis=1) == labels.cpu().numpy())

        print(f"Epoch {epoch + 1}/{epochs} - test accuracy: {(100 * test_acc):.2f}% and CE loss {loss.item():.2f}")

    return model


def evaluate_model(model, data, device):
    """Evaluate torch model."""
    model.eval()
    logits = torch.Tensor().to(device)
    targets = torch.LongTensor().to(device)

    with torch.no_grad():
        for images, labels in data:
            images, labels = images.to(device), labels.to(device)
            logits = torch.cat([logits, model(images)])
            targets = torch.cat([targets, labels])

    return torch.nn.functional.softmax(logits, dim=1), targets
