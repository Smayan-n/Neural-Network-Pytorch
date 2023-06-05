import numpy as np
import sys, math, cv2
import torch, pyautogui

sys.path.append(
    "C:\\Smayan's Files\\Programming\\Python\\AI\\Neural Networks\\Neural Networks Pytorch"
)
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from Utility import interpolate, lerp
from HandWrittenRecognizerNetwork import HandWrittenRecognizerNetwork
from DigitNet import Net
from CNN import CNN


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("Hand-Written Digit Recognizer")
        self.setGeometry(100, 100, COLS * PIXEL_SIZE + RELIEF_RIGHT, ROWS * PIXEL_SIZE)
        self.setFixedSize(COLS * PIXEL_SIZE + RELIEF_RIGHT, ROWS * PIXEL_SIZE)
        self.setStyleSheet("background-color: rgb(38, 36, 48)")

        # Load in the Neural Network
        # digit_recognizer_model.pt: 96.4% accuracy on testset - overfitted model
        # digit_recognizer_model_poor.pt: 93.7% accuracy on testset - not good on my digits - trained w/o transformations
        # digit_recognizer_model_best.pt: 94.9% - best model on my digits
        # self.model = HandWrittenRecognizerNetwork(784, 128, 10)
        # self.model.load("./trained models/digit_recognizer_model_best.pt")

        # CNN model is best
        self.model = CNN()
        self.model.load("./trained models/cnn_digit_model.pt")

        # inputs
        self.mouse_clicked = False
        self.curr_cursor_pos = None
        self.prev_cursor_pos = None

        # # pixels which are labels
        self.pixels = [[QLabel(self) for i in range(COLS)] for j in range(ROWS)]
        # array to store center coordinates of each pixel
        self.pixel_centers = [[None for i in range(COLS)] for j in range(ROWS)]

        self.createGrid()
        self.initUI()

    def createGrid(self):
        """creates a grid with the pre-defined 2D pixels array"""
        x = 0
        y = 0

        for row in range(ROWS):
            for col in range(COLS):
                pixel = self.pixels[row][col]

                # determines placement and size of pixel
                pixel.setGeometry(x, y, PIXEL_SIZE, PIXEL_SIZE)
                pixel.setStyleSheet("background-color: black")

                # calculate center of pixel
                center_x = x + PIXEL_SIZE // 2
                center_y = y + PIXEL_SIZE // 2
                self.pixel_centers[row][col] = (center_x, center_y)

                x += PIXEL_SIZE

            x = 0
            y += PIXEL_SIZE

    def updatePixels(self):
        """loop function"""

        # interpolate points between cursor positions and draw pixels
        if self.prev_cursor_pos and self.curr_cursor_pos:
            for x, y in interpolate(
                (self.prev_cursor_pos.x(), self.prev_cursor_pos.y()),
                (self.curr_cursor_pos.x(), self.curr_cursor_pos.y()),
                step_size=0.075,
            ):
                self.drawPixel(x, y)

    def drawPixel(self, x, y):
        """draws a pixel at the given coordinate"""

        # original: 255, 180, 128, 20
        max_intensity = 255
        mid1_intensity = 180
        mid2_intensity = 128
        min_intensity = 20

        # gets widget that that cursor is currently hovering over
        pixel = qApp.widgetAt(QPoint(x, y))

        if type(pixel) == QLabel and pixel.text() == "":
            # for current pixel
            intensity = self.getPixelIntensity(
                max_intensity, mid1_intensity, pixel, x, y
            )
            pixel.setStyleSheet(
                f"background-color: rgb({intensity}, {intensity}, {intensity})"
            )

            # get surrounding pixels - right, down, left, up
            surrounding_pixels = self.getSurroundingPixels(pixel)

            # do the same thing for surrounding pixels
            for pixel in surrounding_pixels:
                intensity = self.getPixelIntensity(
                    mid2_intensity, min_intensity, pixel, x, y, surrounding_pixel=True
                )
                pixel.setStyleSheet(
                    f"background-color: rgb({intensity}, {intensity}, {intensity})"
                )

    def getPixelIntensity(
        self, min_intensity, max_intensity, pixel, x, y, surrounding_pixel=False
    ):
        """return pixel's white intensity based on cursor pos"""

        # position relative to window
        cursor_pos = self.mapFromGlobal(QPoint(x, y))
        mx, my = cursor_pos.x(), cursor_pos.y()

        pixel_indices = self.getPixelIndices(pixel)
        pixel_center = self.pixel_centers[pixel_indices[0]][pixel_indices[1]]

        # get distance from cursor to pixel center
        dist_from_center = np.sqrt(
            (mx - pixel_center[0]) ** 2 + (my - pixel_center[1]) ** 2
        )
        if surrounding_pixel:
            dist_from_center = dist_from_center - PIXEL_SIZE / 2

        # get current pixel intensity
        intensity = (
            pixel.palette()
            .color(pixel.backgroundRole())
            .red()  # all channels are the same
        )

        # What the intensity should be
        new_intensity = int(
            lerp(min_intensity, max_intensity, dist_from_center / (PIXEL_SIZE / 2))
        )
        # for surrounding pixels
        if surrounding_pixel:
            if new_intensity > min_intensity:
                new_intensity = min_intensity

        # only change intensity if it is more than current intensity
        if intensity < new_intensity:
            intensity = new_intensity

        return intensity

    def getPixelIndices(self, pixel):
        """returns the 2D indices of the given pixel"""
        row = 0
        col = 0
        for i, x in enumerate(self.pixels):
            if pixel in x:
                row = i
                col = x.index(pixel)

        return row, col

    def getSurroundingPixels(self, pixel):
        """returns adjacent pixels to the given pixel"""
        surrounding_pixels = []
        row, col = self.getPixelIndices(pixel)
        # get surrounding pixels
        # NOTE: fix the bounds problem
        try:
            surrounding_pixels.append(self.pixels[row][col + 1])
            surrounding_pixels.append(self.pixels[row + 1][col])
            surrounding_pixels.append(self.pixels[row][col - 1])
            surrounding_pixels.append(self.pixels[row - 1][col])
        except IndexError:
            pass

        return surrounding_pixels

    def predictDigit(self):
        """passes the current grid into neural network"""
        output = self.model.getOutput(self.getPixelArray())
        output_probs = self.model.getOutputDistribution(output)
        # display predictions
        self.displayPredictions(output_probs)

    def displayPredictions(self, output_probs):
        """changes the text of the prediction labels to the model's predicted probabilities"""

        for label in self.prediction_labels:
            digit = np.argmax(output_probs)
            highest_prob = output_probs[digit]
            label.setText(f"{DIGIT_LABELS[digit]}:\t    {highest_prob*100:.2f} %")

            # set probability to -1 so that it is not selected again
            output_probs[digit] = -1

    def getPixelArray(self):
        """returns 2D numpy array of the grid"""
        image = np.zeros((ROWS, COLS))
        for row in range(ROWS):
            for col in range(COLS):
                pixel = self.pixels[row][col]
                pixel_intensity = pixel.palette().color(pixel.backgroundRole()).red()
                # normalize pixel intensity
                # normalized_value = (pixel_intensity / 255 - 0.1307) / 0.3081
                image[row][col] = pixel_intensity

        # dilate image to make digits thicker
        dilated = cv2.dilate(image, np.ones((2, 2), np.uint8), iterations=1)
        normalized = (image / 255 - 0.1307) / 0.3081
        return normalized

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_clicked = True
            self.drawPixel(event.globalX(), event.globalY())

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.mouse_clicked = False
            # reset mouse positions
            self.curr_cursor_pos = None
            self.prev_cursor_pos = None

    def mouseMoveEvent(self, event):
        if self.mouse_clicked:
            self.curr_cursor_pos = QCursor.pos()
            self.updatePixels()
            self.predictDigit()
            self.prev_cursor_pos = self.curr_cursor_pos

    def clearGrid(self):
        """sets all pixels in grid to black"""
        # reset mouse positions
        self.curr_cursor_pos = None
        self.prev_cursor_pos = None
        for row in range(ROWS):
            for col in range(COLS):
                self.pixels[row][col].setStyleSheet("background-color: black")

    def initUI(self):
        # clear btn
        self.clear_btn = QPushButton(self)
        self.clear_btn.setText("Clear")
        self.clear_btn.setFont(FONT1)
        self.clear_btn.setGeometry(875, 50, 120, 80)
        self.clear_btn.setStyleSheet(f"background-color: {RED}")
        self.clear_btn.clicked.connect(self.clearGrid)

        # prediction labels
        self.prediction_labels = []
        for i in range(10):
            label = QLabel(self)
            label.setText(f"{DIGIT_LABELS[i]}:\t    0.00 %")
            label.setFont(FONT1)
            label.setGeometry(875, 170 + i * 55, 450, 55)
            label.setStyleSheet(
                "color: white; padding: 10px; font-weight:bold"
                if i == 0
                else "padding: 10px; color: rgb(118, 118, 118)"
            )
            self.prediction_labels.append(label)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec_()


DIGIT_LABELS = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
]

ROWS = 28
COLS = 28
PIXEL_SIZE = 30
RELIEF_RIGHT = 500

FONT1 = QFont("Arial", 20)

RED = "rgb(255, 0, 0)"
GREEN = "rgb(0, 180, 0)"
LIGHT_GREEN = "rgb(0, 255, 0)"
BLUE = "rgb(0, 0, 255)"


if __name__ == "__main__":
    main()
