import torch, time
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from DataHandler import DataHandler

# Neural Net for recognizing hand-written digits and letters
class HandWrittenRecognizerNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()
        """initializes the network's layers"""
        # hidden layers - (number of inputs/neurons, number of outputs/neurons)
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        # output layer
        self.layer4 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """performs a forward pass through the network"""

        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return F.log_softmax(x, dim=1)

    def load(self, path):
        """loads model from given path"""
        self.load_state_dict(torch.load(path))

    def save(self, path):
        """saves model to given path"""
        torch.save(self.state_dict(), path)

    def getOutput(self, input):
        """returns network output for given 2D numpy array"""

        # convert np array to a tensor
        input = torch.from_numpy(input.astype(np.float32))
        # reshape vector
        input = input.view(-1, 784)

        # forward pass
        output = self.forward(input)
        return output

    def getFinalPrediction(self, output):
        """returns the predicted value from given network output"""
        letter_map = {i: chr(i + 96) for i in range(1, 27)}  # for lowercase chars
        letter_map.update(
            {i: chr(i + 38) for i in range(27, 53)}
        )  # for uppercase chars
        pred = torch.argmax(output).item()

        return pred if pred < 10 else letter_map[pred]

    def getOutputDistribution(self, output):
        """returns the percentage distribution of predicted outputs"""

        percents = F.softmax(output, dim=1)
        return percents.detach().numpy()[0]

    def trainNetwork(self, dataset, epochs=3, batch_size=64, learn_rate=0.001):

        # loss/cost function also known as criterion
        loss_fn = nn.NLLLoss()
        # optimizer to update gradients
        optimizer = torch.optim.Adam(self.parameters(), lr=learn_rate)

        # training loop
        losses = []
        batch_loss = 0
        correct = 0
        total = 0
        start_time = time.time()
        for epoch in range(epochs):
            for batch, (X, y) in enumerate(dataset):
                # NOTE: X is whole batch of data, y is whole batch of labels

                # randomly transform images
                # X = X.view(batch_size, 28, 28)
                # X = DataHandler.apply_random_transformation(
                #     X,
                #     scale_ranges=[0.65, 1.25],
                #     translation_ranges=[-4, 4],
                #     rotation_ranges=[-40, 40],
                # )

                # X = X.to(torch.float32)

                # clear gradients before each batch
                optimizer.zero_grad()

                # forward pass
                output = self.forward(X.view(-1, 28 * 28))

                # calculate loss
                loss = loss_fn(output, y)
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
                if (batch + 1) % 50 == 0:
                    print(
                        f"Epoch: {epoch + 1}    Batch: {batch + 1}    Loss: {(batch_loss/50):.4f}   Train Accuracy: {(correct / total):.4f}   Time: {(time.time() - start_time):.4f}"
                    )
                    losses.append(batch_loss / 50)
                    batch_loss = 0
                    correct = 0
                    total = 0
                    start_time = time.time()

        return losses

    def evaluateNetwork(self, dataset):
        """evaluates network performance on given dataset"""

        correct = 0
        total = 0
        with torch.no_grad():
            for data in dataset:
                X, y = data
                # randomly transform images
                # X = X.view(128, 28, 28)
                # X = DataHandler.apply_random_transformation(
                #     X,
                #     scale_ranges=[0.65, 1.25],
                #     translation_ranges=[-4, 4],
                #     rotation_ranges=[-40, 40],
                # )
                # X = X.to(torch.float32)

                output = self.forward(X.view(-1, 784))
                # print(output)
                for idx, i in enumerate(output):
                    # print(torch.argmax(i), y[idx])
                    if torch.argmax(i) == y[idx]:
                        correct += 1
                    total += 1

        return round(correct / total, 3)
