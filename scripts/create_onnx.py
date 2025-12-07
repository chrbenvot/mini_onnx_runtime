# This is the script to generate the file,actual file is generated elsewhere(to not make a separate venv and bloat the project)
import torch
import torch.nn as nn
import torch.onnx

# 1. Define a Tiny Model (The same structure we will implement)
class TinyNet(nn.Module):
    def __init__(self):
        super(TinyNet, self).__init__()
        # Conv: 1 input channel (grayscale), 2 output channels, 3x3 kernel
        self.conv1 = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        # Flatten: 2 channels * 28 * 28 = 1568
        self.fc = nn.Linear(2*28*28, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = torch.flatten(x, 1) # Flatten logic
        x = self.fc(x)
        return x

# 2. Export
model = TinyNet()
model.eval() # Set to inference mode (removes dropout/batchnorm randomness)

# Create dummy input (Batch Size 1, 1 Channel, 28x28 Image)
dummy_input = torch.randn(1, 1, 28, 28)

print("Exporting 'model.onnx'...")
torch.onnx.export(
    model, 
    dummy_input, 
    "../model.onnx",  # Save to root folder or data/
    export_params=True,
    opset_version=13, # Standard version
    input_names=['Input3'], 
    output_names=['Output3']
)
print("Done!")
