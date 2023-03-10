import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import time
from DataHandler import DataHandler
from HandWrittenRecognizerNetwork import HandWrittenRecognizerNetwork
from CNN_Network import CNN_Network


# net = HandWrittenRecognizerNetwork(784, 128, 27)
# net.load("trained models/letter_recognizer_model.pt")

# trainset, testset = DataHandler.get_hand_drawn_letters(128)
# print("Trainset Accuracy: ", net.evaluateNetwork(trainset))
# print("TestSet Accuracy: ", net.evaluateNetwork(testset))

net = HandWrittenRecognizerNetwork(784, 128, 10)
net.load("trained models/digit_recognizer_model_best.pt")

tensor = torch.zeros(1, 784)
output = net.forward(tensor)
# print(output)
print(net.getOutputDistribution(output))

# net = CNN_Network()
# net.load("cnn_model.pt")

# trainset, testset = DataHandler.get_hand_drawn_letters(128)
# print("Trainset Accuracy: ", net.evaluateNetwork(trainset))
# print("TestSet Accuracy: ", net.evaluateNetwork(testset))
