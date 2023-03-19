from HandWrittenRecognizerNetwork import HandWrittenRecognizerNetwork
import torch
from DigitNet import Net


def convert():
    # model = HandWrittenRecognizerNetwork(784, 128, 10)
    # model.load("trained models/digit_recognizer_model_best.pt")
    # model.eval()
    # dummy_input = torch.zeros(1, 784)
    # torch.onnx.export(
    #     model, dummy_input, "digit_model.onnx", verbose=True
    # )

    # model = HandWrittenRecognizerNetwork(784, 128, 27)
    # model.load("trained models/letter_recognizer_model.pt")
    # model.eval()
    # dummy_input = torch.zeros(1, 784)
    # torch.onnx.export(model, dummy_input, "letter_model.onnx", verbose=True)

    model = Net()
    model.load("cnn_model.pt")
    model.eval()
    dummy_input = torch.zeros(1, 1, 28, 28)
    torch.onnx.export(
        model,
        dummy_input,
        "./trained onnx models/cnn_digit_model.onnx",
        verbose=True,
        opset_version=9,
    )

    # import onnx

    # model = onnx.load("./trained onnx models/onnx_model.onnx")
    # for node in model.graph.node:
    #     print(node.name, node.op_type)


convert()
