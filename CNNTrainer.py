import os
import torch, time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from DataHandler import DataHandler
from HandWrittenRecognizerNetwork import HandWrittenRecognizerNetwork
from Utility import plot_cost_graph
from CNN import CNN

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ", device)

# data
batch_size = 64
trainset, testset = DataHandler.get_hand_drawn_digits(batch_size, augment_data=True)


# neural network
net = CNN()
net = net.to(device)

# load network
# net.load("trained models/cnn_digit_model.pt")

# train model
# make sure to add more args to to DataLoader if you use gpu
losses = net.trainNetwork(trainset, device, epochs=14, learn_rate=1)

# save model
net.save("trained models/cnn_digit_model_experiment.pt")

# plot cost graph
plot_cost_graph(losses)

# eval model on trainset and testset
print("Trainset Accuracy: ", net.evaluateNetwork(trainset, device))
print("Testset Accuracy: ", net.evaluateNetwork(testset, device))
# 79.1% accuracy
