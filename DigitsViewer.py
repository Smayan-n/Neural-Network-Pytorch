from keras.datasets import mnist
import numpy as np
import sys, cv2

from PyQt5 import QtCore
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from DataHandler import DataHandler


class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Visualize Digits")
        self.setGeometry(100, 100, COLS * CELL_SIZE, ROWS * CELL_SIZE + RELIEF)
        self.setStyleSheet("background-color: grey")

        # pixels which are labels
        self.pixels = [[QLabel(self) for i in range(COLS)] for j in range(ROWS)]

        # digit label
        self.digit_label = QLabel(self)
        self.digit_label.setFont(FONT1)
        self.digit_label.setGeometry(10, 0, 500, RELIEF)

        self.dilate_val = 1
        self.dilate_slider = QSlider(Qt.Horizontal, self)
        self.dilate_slider.setMinimum(1)
        self.dilate_slider.setMaximum(10)
        self.dilate_slider.setSingleStep(1)
        self.dilate_slider.setGeometry(10, 100, 500, 50)
        self.dilate_slider.valueChanged.connect(
            lambda: self.changeDilate(self.dilate_slider.value())
        )

        # previous and next buttons
        self.prev_digit_btn = QPushButton(self)
        self.prev_digit_btn.setText("Prev")
        self.prev_digit_btn.setFont(FONT1)
        self.prev_digit_btn.setGeometry(250, 10, 120, 80)
        self.prev_digit_btn.setStyleSheet(f"background-color: {BLUE}")
        self.prev_digit_btn.clicked.connect(lambda: self.changeDigit(-1))

        self.next_digit_btn = QPushButton(self)
        self.next_digit_btn.setText("Next")
        self.next_digit_btn.setFont(FONT1)
        self.next_digit_btn.setGeometry(400, 10, 120, 80)
        self.next_digit_btn.setStyleSheet(f"background-color: {RED}")
        self.next_digit_btn.clicked.connect(lambda: self.changeDigit(1))

        self.prev_digit_btn = QPushButton(self)
        self.prev_digit_btn.setText("Transform")
        self.prev_digit_btn.setFont(FONT1)
        self.prev_digit_btn.setGeometry(550, 10, 120, 80)
        self.prev_digit_btn.setStyleSheet(f"background-color: {GREEN}")
        self.prev_digit_btn.clicked.connect(self.transformDigit)

        self.createGrid()

        # load MNIST data
        # (self.training_data, self.training_labels), (
        #     self.testing_data,
        #     self.testing_labels,
        # ) = mnist.load_data()
        self.trainset, self.testset = DataHandler.get_hand_drawn_digits(
            1, augment_data=True
        )
        self.training_data = self.testset.dataset.data
        self.training_labels = self.testset.dataset.targets

        # print(self.training_data[0])

        # draw the first digit
        self.current_digit = 0
        self.drawDigit(
            self.training_data[self.current_digit],
            self.training_labels[self.current_digit],
        )

    def createGrid(self):
        """creates a grid with the pre-defined 2D pixels array"""
        x = 0
        y = RELIEF

        for row in range(ROWS):
            for col in range(COLS):
                pixel = self.pixels[row][col]

                # alligns text in label to the center
                pixel.setAlignment(QtCore.Qt.AlignCenter)
                pixel.setFont(FONT1)

                # determines placement and size of label
                pixel.setGeometry(x, y, CELL_SIZE, CELL_SIZE)
                x += CELL_SIZE

                pixel.setStyleSheet("background-color: black")
            x = 0
            y += CELL_SIZE

    def drawDigit(self, digit, label):
        # digit is a 2D numpy array

        # reset pixels
        self.createGrid()

        # update label
        self.digit_label.setText(f"Digit Label: {label}")

        # digit dilation
        digit = cv2.dilate(
            digit.view(28, 28).numpy(),
            np.ones((self.dilate_val, self.dilate_val), np.uint8),
            iterations=1,
        )
        # color pixels
        for r in range(digit.shape[0]):
            for c in range(digit.shape[1]):
                color = digit[r][c]
                self.pixels[r][c].setStyleSheet(
                    f"background-color: rgb({color}, {color}, {color})"
                )

    def changeDigit(self, flag):
        # if flag == 1, next digit, else prev
        if flag == 1:
            self.current_digit += 1
        elif flag == -1:
            self.current_digit -= 1

        self.drawDigit(
            self.training_data[self.current_digit],
            self.training_labels[self.current_digit],
        )

    def transformDigit(self):
        curr_digit = self.training_data[self.current_digit : self.current_digit + 1]
        transformed_digit = DataHandler.apply_random_transformation(curr_digit)
        self.drawDigit(transformed_digit[0], self.training_labels[self.current_digit])

    def changeDilate(self, val):
        self.dilate_val = val
        self.changeDigit(0)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


ROWS = 28
COLS = 28
CELL_SIZE = 25
RELIEF = 100

FONT1 = QFont("Arial", 20)

RED = "rgb(255, 0, 0)"
GREEN = "rgb(0, 180, 0)"
LIGHT_GREEN = "rgb(0, 255, 0)"
BLUE = "rgb(0, 0, 255)"


if __name__ == "__main__":
    main()

# load training and testing digits
# (training_data, training_labels), (testing_data, testing_labels) = mnist.load_data()

# print(training_data.shape)
# print(training_labels[0])

# for i in range(10):
#     print(training_labels[i])

# py_arr = [
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
#     [[2, 2], [2, 2]],
# ]
# arr = np.array(py_arr)
# print(arr.shape)
