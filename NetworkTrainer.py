import os
import torch, time
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from DataHandler import DataHandler
from HandWrittenRecognizerNetwork import HandWrittenRecognizerNetwork
from Utility import plot_cost_graph
import string

# get device the network will run on
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # data
# batch_size = 128
# trainset, testset = DataHandler.get_hand_drawn_letters(batch_size)


# # neural network
# net = HandWrittenRecognizerNetwork(784, 128, 27)

# # load network
# net.load("trained models/letter_recognizer_model2.pt")

# # train model
# losses = net.trainNetwork(trainset, 10, batch_size, 0.0002)

# # save model
# net.save("trained models/letter_recognizer_model.pt")

# # plot cost graph
# plot_cost_graph(losses)

# # eval model on trainset and testset
# print("Trainset Accuracy: ", net.evaluateNetwork(trainset))
# print("Testset Accuracy: ", net.evaluateNetwork(testset))
# # 87.8% - model is best yet


# data
batch_size = 128
trainset, testset = DataHandler.get_hand_drawn_digits(batch_size, augment_data=True)


# neural network
net = HandWrittenRecognizerNetwork(784, 128, 10)

# load network
net.load("trained models/digit_recognizer_experiment.pt")

# train model
losses = net.trainNetwork(trainset, 10, batch_size, 0.0002)

# save model
net.save("trained models/digit_recognizer_experiment.pt")

# plot cost graph
plot_cost_graph(losses)

# eval model on trainset and testset
print("Trainset Accuracy: ", net.evaluateNetwork(trainset))
print("Testset Accuracy: ", net.evaluateNetwork(testset))
# 81.3% accuracy
