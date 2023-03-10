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


class CNN(nn.Module):
    def __init__(self, lr, epochs, batch_size, num_classes=10):
        super(CNN, self).__init__()
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.loss_history = []
        self.acc_history = []
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
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

        self.fc1 = nn.Linear(input_dims, self.num_classes)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)

        self.loss = nn.CrossEntropyLoss()
        self.to(self.device)
        self.get_data()

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
        batch_data = T.tensor(batch_data).to(self.device)

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

    def get_data(self):

        (
            self.train_data_loader,
            self.test_data_loader,
        ) = DataHandler.get_hand_drawn_digits(self.batch_size)

    def _train(self):
        self.train()
        for i in range(self.epochs):
            ep_loss = 0
            ep_acc = []
            start_time = time.time()
            for j, (input, label) in enumerate(self.train_data_loader):
                # apply transformation
                input = input.view(self.batch_size, 28, 28)
                input = DataHandler.apply_random_transformation(input)
                input = input.view(self.batch_size, 1, 28, 28)
                input = input.to(T.float32)

                self.optimizer.zero_grad()
                label = label.to(self.device)

                prediction = self.forward(input)

                loss = self.loss(prediction, label)
                prediction = F.softmax(prediction, dim=1)
                classes = T.argmax(prediction, dim=1)
                wrong = T.where(
                    classes != label,
                    T.tensor([1.0]).to(self.device),
                    T.tensor([0.0]).to(self.device),
                )
                acc = 1 - T.sum(wrong) / self.batch_size

                ep_acc.append(acc.item())
                self.acc_history.append(acc.item())
                ep_loss += loss.item()
                loss.backward()
                self.optimizer.step()
                print(
                    f"Batch: {j}    Loss: {loss.item():.4f}    Accuracy: {acc.item():.4f}"
                )
            print(
                "Finish epoch ",
                i,
                "total loss %.3f" % ep_loss,
                "accuracy %.3f" % np.mean(ep_acc),
                f"Time: {(time.time() - start_time):.4f}",
            )
            self.loss_history.append(ep_loss)

    def _test(self):
        self.eval()

        ep_loss = 0
        ep_acc = []
        for j, (input, label) in enumerate(self.test_data_loader):
            label = label.to(self.device)
            prediction = self.forward(input)
            loss = self.loss(prediction, label)
            prediction = F.softmax(prediction, dim=1)
            classes = T.argmax(prediction, dim=1)
            wrong = T.where(
                classes != label,
                T.tensor([1.0]).to(self.device),
                T.tensor([0.0]).to(self.device),
            )
            acc = 1 - T.sum(wrong) / self.batch_size

            ep_acc.append(acc.item())

            ep_loss += loss.item()

        print("total loss %.3f" % ep_loss, "accuracy %.3f" % np.mean(ep_acc))


if __name__ == "__main__":
    network = CNN(lr=0.001, batch_size=128, epochs=5)
    network._train()

    # save network
    T.save(network.state_dict(), "cnn_model.pth")

    plt.plot(network.loss_history)
    plt.show()
    plt.plot(network.acc_history)
    plt.show()
    network._test()
