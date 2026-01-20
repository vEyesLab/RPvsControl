import torch
import torch.nn as nn
import onnx
import os

class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.fc_layers = nn.Sequential(
            nn.ReLU(),                     # 0
            nn.Linear(64 * 37 * 37, 128),   # 1
            nn.ReLU(),                     # 2
            nn.Identity(),                 # 3
            nn.Linear(128, 1)               # 4
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x

# Parameters
IMAGE_SIZE = 150   # REAL TRAINING DIMENSION

# List of .pt models to convert
pt_models = [
    "model_SGD_LR0.001.pt",
    "model_SGD_LR0.0005.pt",
    "model_Adam_LR0.001.pt",
    "model_Adam_LR0.0005.pt"
]

# Conversion loop
for pt_model in pt_models:
    print(f"\nConverting {pt_model} to ONNX...")

    # Create the model and load weights
    model = CNNModel()
    state_dict = torch.load(pt_model, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    # ONNX file name
    onnx_model_name = os.path.splitext(pt_model)[0] + ".onnx"

    # Dummy input
    dummy_input = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        onnx_model_name,
        input_names=["input"],
        output_names=["output"],
        opset_version=11
    )

    print(f"ONNX export completed successfully: {onnx_model_name}")

    # ONNX validation
    onnx_model = onnx.load(onnx_model_name)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid")
    print("Input names:", [i.name for i in onnx_model.graph.input])
