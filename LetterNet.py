import torch, torchvision, copy
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import scipy, cv2, random
from scipy import ndimage
from sklearn.model_selection import train_test_split
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR


def transformData(dataset):

    dataset.data = torch.flip(dataset.data, [2])
    dataset.data = torch.rot90(dataset.data, 1, [1, 2])

    # reshape data to shape [batchsize, 1, 28, 28]
    # dataset.data = dataset.data.view([-1, 1, 28, 28])

    # subtract 1 from targets to make them 0 indexed(remove the 'N/A' class)
    dataset.classes = dataset.classes[1:]
    dataset.targets = copy.deepcopy(dataset.targets) - 1

    return dataset


def getData(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomAffine(
                degrees=(-30, 30),
                translate=(0.32, 0.32),
                scale=(0.5, 1),
                shear=(-15, 15, -15, 15),
            ),
        ]
    )

    dataset_train = datasets.EMNIST(
        root="./datasets",
        split="letters",
        train=True,
        download=True,
        transform=transform,
    )
    dataset_test = datasets.EMNIST(
        root="./datasets",
        split="letters",
        train=False,
        download=True,
        transform=transform,
    )

    dataset_train = transformData(dataset_train)
    dataset_test = transformData(dataset_test)

    # rotate and flip images

    kwargs = {"num_workers": 1, "pin_memory": True}
    # kwargs = {}
    trainset = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    testset = DataLoader(
        dataset_test, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )

    print(trainset.dataset.data.shape, trainset.dataset.targets.shape)
    print(testset.dataset.data.shape, testset.dataset.targets.shape)

    return trainset, testset


def getData2(batch_size=64):
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.RandomAffine(
                degrees=(-30, 30),
                translate=(0.32, 0.32),
                scale=(0.5, 1),
                shear=(-15, 15, -15, 15),
            ),
        ]
    )

    dataset = datasets.EMNIST(
        root="./datasets",
        split="letters",
        download=True,
        transform=transform,
    )

    # rotate and flip images
    dataset.data = torch.flip(dataset.data, [2])
    dataset.data = torch.rot90(dataset.data, 1, [1, 2])

    # reshape data to shape [batchsize, 1, 28, 28]
    dataset.data = dataset.data.view([-1, 1, 28, 28])

    # subtract 1 from targets to make them 0 indexed(remove the 'N/A' class)
    dataset.classes = dataset.classes[1:]
    dataset.targets = copy.deepcopy(dataset.targets) - 1

    # plt.hist(dataset.data[:10, :, :, :].view(1, -1).detach(), 40)
    # plt.show()
    # dataset.data = dataset.data / torch.max(dataset.data)

    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # x, y = next(iter(dataloader))
    # grid = torchvision.utils.make_grid(x, pad_value=1)
    # torchvision.utils.save_image(grid, "grid.png")

    dataset.data = dataset.data.float()
    print(type(dataset.data))
    print(type(dataset.targets))

    # split data into train and test
    train_data, test_data, train_labels, test_labels = train_test_split(
        dataset.data, dataset.targets, test_size=0.1
    )

    train_data = TensorDataset(train_data, train_labels)
    test_data = TensorDataset(test_data, test_labels)

    # kwargs = {"num_workers": 1, "pin_memory": True}
    kwargs = {}
    trainset = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )
    testset = DataLoader(
        test_data, batch_size=batch_size, shuffle=True, drop_last=True, **kwargs
    )

    print(trainset.dataset.tensors[0].shape, trainset.dataset.tensors[1].shape)
    print(testset.dataset.tensors[0].shape, testset.dataset.tensors[1].shape)

    return trainset, testset


class LetterNet(nn.Module):
    def __init__(self):
        super(LetterNet, self).__init__()
        # self.conv1 = nn.Conv2d(1, 6, 3, padding=1)
        # self.bnorm1 = nn.BatchNorm2d(6)

        # self.conv2 = nn.Conv2d(6, 6, 3, padding=1)
        # self.bnorm2 = nn.BatchNorm2d(6)

        # self.fc1 = nn.Linear(7 * 7 * 6, 50)
        # self.fc2 = nn.Linear(50, 26)

        # self.conv1 = nn.Conv2d(1, 32, 3, 1)
        # self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout2d(0.5)
        # self.fc1 = nn.Linear(9216, 128)
        # self.fc2 = nn.Linear(128, 26)

        # self.layer1 = nn.Sequential(
        #     nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
        #     nn.BatchNorm2d(6),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.layer2 = nn.Sequential(
        #     nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
        #     nn.BatchNorm2d(16),
        #     nn.ReLU(),
        #     nn.MaxPool2d(kernel_size=2, stride=2),
        # )
        # self.fc = nn.Linear(400, 120)
        # self.relu = nn.ReLU()
        # self.fc1 = nn.Linear(120, 84)
        # self.relu1 = nn.ReLU()
        # self.fc2 = nn.Linear(84, 26)

        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 26)

    def forward(self, x):
        # x = F.max_pool2d(self.conv1(x), 2)
        # x = F.leaky_relu(self.bnorm1(x))

        # x = F.max_pool2d(self.conv2(x), 2)
        # x = F.leaky_relu(self.bnorm2(x))

        # nUnits = x.shape.numel() / x.shape[0]
        # x = x.view(-1, int(nUnits))

        # x = F.leaky_relu(self.fc1(x))
        # x = self.fc2(x)

        # return x
        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.conv2(x)
        # x = F.max_pool2d(x, 2)
        # x = self.dropout1(x)
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.dropout2(x)
        # x = self.fc2(x)
        # return x

        # out = self.layer1(x)
        # out = self.layer2(out)
        # out = out.reshape(out.size(0), -1)
        # out = self.fc(out)
        # out = self.relu(out)
        # out = self.fc1(out)
        # out = self.relu1(out)
        # out = self.fc2(out)
        # return out

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def getOutput(self, input):

        input = torch.from_numpy(input).float()
        # input = input.view(1, 1, 28, 28)
        input = input.view(1, 784)

        output = self(input)

        return output

    def getOutputDistribution(self, output):
        percents = F.softmax(output, dim=1)
        return percents.detach().numpy()[0]

    def load(self, path):
        """loads model from given path"""
        self.load_state_dict(torch.load(path))

    def evaluateNetwork(self, dataset, device):
        """evaluates network performance on given dataset"""

        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataset:
                X, y = data
                X, y = X.to(device), y.to(device)

                output = self(X)
                # print(output)
                for idx, i in enumerate(output):
                    # print(torch.argmax(i), y[idx])
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

        return round(correct / total, 3)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    lossfun = nn.CrossEntropyLoss()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        data = data.view(-1, 784)

        optimizer.zero_grad()
        output = model(data)
        loss = lossfun(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(args, model, device, test_loader):
    model.eval()
    lossfun = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            data = data.view(-1, 784)

            output = model(data)
            test_loss += lossfun(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate step gamma (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="disables CUDA training"
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )

    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="For Saving the current Model",
    )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)

    train_loader, test_loader = getData(args.batch_size)

    model = LetterNet().to(device)
    # model.load("cnn_model.pt")
    # optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # scheduler = StepLR(optimizer, step_size=1, gamma=0.2)
    for epoch in range(1, 7 + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        # scheduler.step()

    torch.save(model.state_dict(), "cnn_model_letters_linear.pt")
    # 1 is also pretty good
    # 2 - 56%


if __name__ == "__main__":
    main()
    pass
