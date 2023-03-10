from HandWrittenRecognizerNetwork import HandWrittenRecognizerNetwork
import torch


def convert():
    model = HandWrittenRecognizerNetwork(784, 128, 10)
    model.load("trained models/digit_recognizer_model_best.pt")
    model.eval()
    dummy_input = torch.zeros(1, 784)
    torch.onnx.export(
        model, dummy_input, "digit_recognizer_onnx_model.onnx", verbose=True
    )


convert()
