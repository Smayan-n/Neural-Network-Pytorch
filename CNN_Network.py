import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from DataHandler import DataHandler
import time


class CNN_Network(nn.Module):
    def __init__(self):
        super(CNN_Network, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 32, 3)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, 3)
        self.bn3 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.bn5 = nn.BatchNorm2d(64)
        self.conv6 = nn.Conv2d(64, 64, 3)
        self.bn6 = nn.BatchNorm2d(64)
        self.maxpool2 = nn.MaxPool2d(2)

        input_dims = self.calc_input_dims()

        self.fc1 = nn.Linear(input_dims, 10)

    def calc_input_dims(self):
        batch_data = T.zeros((1, 1, 28, 28))
        batch_data = self.conv1(batch_data)
        # batch_data = self.bn1(batch_data)
        batch_data = self.conv2(batch_data)
        # batch_data = self.bn2(batch_data)
        batch_data = self.conv3(batch_data)

        batch_data = self.maxpool1(batch_data)
        batch_data = self.conv4(batch_data)
        batch_data = self.conv5(batch_data)
        batch_data = self.conv6(batch_data)
        batch_data = self.maxpool2(batch_data)

        return int(np.prod(batch_data.size()))

    def forward(self, batch_data):

        batch_data = self.conv1(batch_data)
        batch_data = self.bn1(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv2(batch_data)
        batch_data = self.bn2(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv3(batch_data)
        batch_data = self.bn3(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool1(batch_data)

        batch_data = self.conv4(batch_data)
        batch_data = self.bn4(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv5(batch_data)
        batch_data = self.bn5(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.conv6(batch_data)
        batch_data = self.bn6(batch_data)
        batch_data = F.relu(batch_data)

        batch_data = self.maxpool2(batch_data)

        batch_data = batch_data.view(batch_data.size()[0], -1)

        classes = self.fc1(batch_data)
        return classes

    def load(self, path):
        """loads model from given path"""
        self.load_state_dict(T.load(path))

    def save(self, path):
        """saves model to given path"""
        T.save(self.state_dict(), path)

    def getOutput(self, input):
        """returns network output for given 2D numpy array"""

        # convert np array to a tensor
        input = T.from_numpy(input.astype(np.float32))
        # reshape vector
        input = input.view(1, 1, 28, 28)

        # forward pass
        output = self.forward(input)
        return output

    def getFinalPrediction(self, output):
        return T.argmax(output).item()

    def getOutputDistribution(self, output):
        """returns the percentage distribution of predicted outputs"""

        percents = F.softmax(output, dim=1)
        return percents.detach().numpy()[0]

    def evaluateNetwork(self, dataset):
        """evaluates network performance on given dataset"""

        correct = 0
        total = 0
        with T.no_grad():
            for data in dataset:
                X, y = data
                # # randomly transform images
                # X = X.view(128, 28, 28)
                # X = DataHandler.apply_random_transformation(
                #     X,
                # )

                # X = X.to(T.float32)
                output = self.forward(X)

                # print(output)
                for idx, i in enumerate(output):
                    # print(torch.argmax(i), y[idx])
                    if T.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

        return round(correct / total, 3)
