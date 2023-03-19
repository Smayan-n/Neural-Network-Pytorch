import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import time
from DataHandler import DataHandler
from HandWrittenRecognizerNetwork import HandWrittenRecognizerNetwork
from DigitNet import Net
from LetterNet import LetterNet, getData


# net = HandWrittenRecognizerNetwork(784, 128, 27)
# net.load("trained models/letter_recognizer_model.pt")

# trainset, testset = DataHandler.get_hand_drawn_letters(128)
# print("Trainset Accuracy: ", net.evaluateNetwork(trainset))
# print("TestSet Accuracy: ", net.evaluateNetwork(testset))

# net = HandWrittenRecognizerNetwork(784, 128, 10)
# net.load("trained models/digit_recognizer_model_best.pt")

# tensor = torch.zeros(1, 784)
# output = net.forward(tensor)
# # print(output)
# print(net.getOutputDistribution(output))

# net = CNN_Network()
# net.load("cnn_model.pt")

# trainset, testset = DataHandler.get_hand_drawn_letters(128)
# print("Trainset Accuracy: ", net.evaluateNetwork(trainset))
# print("TestSet Accuracy: ", net.evaluateNetwork(testset))

# model = Net()
# model.load("cnn_model.pt")
# trainset, testset = DataHandler.get_hand_drawn_digits(128)
# print("Trainset Accuracy: ", model.evaluateNetwork(trainset, "cpu"))
# print("Testset Accuracy: ", model.evaluateNetwork(testset, "cpu"))


# model = LetterNet()
# model.load("cnn_model_letters3.pt")
# trainset, testset = getData(128)
# print(trainset.dataset.tensors[0][0])
# print("Trainset Accuracy: ", model.evaluateNetwork(trainset, "cpu"))
# print("Testset Accuracy: ", model.evaluateNetwork(testset, "cpu"))

arr = np.ones((28, 28))
print(arr)
resized = np.pad(arr, ((2, 2), (2, 2)), "constant", constant_values=0)
print(resized)
