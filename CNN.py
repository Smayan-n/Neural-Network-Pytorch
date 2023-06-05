import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from DataHandler import DataHandler
import time


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def getOutput(self, input):
        input = torch.from_numpy(input).float()
        input = input.view(1, 1, 28, 28)

        output = self.forward(input)

        return output

    def getOutputDistribution(self, output):
        percents = F.softmax(output, dim=1)
        return percents.detach().numpy()[0]

    def load(self, path):
        """loads model from given path"""
        self.load_state_dict(torch.load(path, map_location=torch.device("cpu")))

    def save(self, path):
        """saves model to given path"""
        torch.save(self.state_dict(), path)

    def trainNetwork(self, dataset, device, epochs=3, learn_rate=1):
        self.train()

        # optimizer to update gradients
        optimizer = torch.optim.Adadelta(self.parameters(), lr=learn_rate)

        # gamma is factor of learn rate decay after [step_size] epochs
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

        # training loop
        losses = []
        batch_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        log_interval = 50
        for epoch in range(epochs):
            for batch, (X, y) in enumerate(dataset):
                # NOTE: X is whole batch of data, y is whole batch of labels
                X, y = X.to(device), y.to(device)

                # clear gradients before each batch
                optimizer.zero_grad()

                # forward pass
                output = self(X)

                # calculate loss
                loss = F.nll_loss(output, y)
                batch_loss += loss.item()

                # backward pass and adjust weights
                loss.backward()
                optimizer.step()

                # training accuracy
                for idx, i in enumerate(output):
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

                # performance metrics
                if (batch + 1) % log_interval == 0:
                    print(
                        f"Epoch: {epoch + 1}    Batch: {batch + 1}/{len(dataset)}    Loss: {(batch_loss/log_interval):.4f}   Train Accuracy: {(correct / total):.4f}   Time: {(time.time() - start_time):.4f}"
                    )
                    losses.append(batch_loss / log_interval)
                    batch_loss = 0
                    correct = 0
                    total = 0
                    start_time = time.time()

            # step scheduler
            scheduler.step()

        return losses

    def evaluateNetwork(self, dataset, device):
        """evaluates network performance on given dataset"""
        self.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataset:
                X, y = data
                X, y = X.to(device), y.to(device)

                output = self.forward(X)
                # print(output)
                for idx, i in enumerate(output):
                    # print(torch.argmax(i), y[idx])
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

        return round(correct / total, 3)
