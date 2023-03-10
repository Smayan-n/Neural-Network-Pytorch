from keras.datasets import mnist
import numpy as np
import sys, random, cv2

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

        self.letter_map = {i: chr(i + 96) for i in range(1, 27)}  # for lowercase chars
        self.letter_map.update(
            {i: chr(i + 38) for i in range(27, 53)}
        )  # for uppercase chars

        # digit label
        self.digit_label = QLabel(self)
        self.digit_label.setFont(FONT1)
        self.digit_label.setGeometry(10, 0, 500, RELIEF)

        # previous and next buttons
        self.prev_digit_btn = QPushButton(self)
        self.prev_digit_btn.setText("Next")
        self.prev_digit_btn.setFont(FONT1)
        self.prev_digit_btn.setGeometry(250, 10, 120, 80)
        self.prev_digit_btn.setStyleSheet(f"background-color: {BLUE}")
        self.prev_digit_btn.clicked.connect(lambda: self.changeLetter(1))

        self.next_digit_btn = QPushButton(self)
        self.next_digit_btn.setText("Transform")
        self.next_digit_btn.setFont(FONT1)
        self.next_digit_btn.setGeometry(450, 10, 150, 80)
        self.next_digit_btn.setStyleSheet(f"background-color: {GREEN}")
        self.next_digit_btn.clicked.connect(self.transformLetter)

        self.dilate_val = 1
        self.dilate_slider = QSlider(Qt.Horizontal, self)
        self.dilate_slider.setMinimum(1)
        self.dilate_slider.setMaximum(10)
        self.dilate_slider.setSingleStep(1)
        self.dilate_slider.setGeometry(10, 100, 500, 50)
        self.dilate_slider.valueChanged.connect(
            lambda: self.changeDilate(self.dilate_slider.value())
        )

        self.createGrid()

        self.testset = DataHandler.get_hand_drawn_letters(
            1, augment_data=True, only_test=True
        )
        self.training_data = self.testset.dataset.data
        self.training_labels = self.testset.dataset.targets

        # draw the first digit
        # index
        self.current_letter = 0
        self.drawLetter(
            self.training_data[self.current_letter],
            self.training_labels[self.current_letter],
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

    def drawLetter(self, digit, label):
        # digit is a 2D numpy array

        # reset pixels
        self.createGrid()

        # update label
        self.digit_label.setText(f"Letter: {self.letter_map[label.item()]}")

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

    def changeLetter(self, flag=1):
        if flag == 1:
            self.current_letter = random.randint(0, len(self.training_data) - 1)
        self.drawLetter(
            self.training_data[self.current_letter],
            self.training_labels[self.current_letter],
        )

    def transformLetter(self):
        curr_letter = self.training_data[self.current_letter : self.current_letter + 1]
        transformed_digit = DataHandler.apply_random_transformation(
            curr_letter, scale_ranges=[0.6, 1.3]
        )
        self.drawLetter(transformed_digit[0], self.training_labels[self.current_letter])

    def changeDilate(self, val):
        self.dilate_val = val
        self.changeLetter(0)


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
