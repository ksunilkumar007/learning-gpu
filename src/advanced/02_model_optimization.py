import torch
import torch.nn as nn
import time

device = torch.device("cuda")

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

# Load pretrained model
model = CNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth"))
model.eval()

# Benchmark — run 1000 inferences and measure time
dummy_input = torch.randn(1, 1, 28, 28).to(device)

# Warmup
for _ in range(10):
    _ = model(dummy_input)

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    _ = model(dummy_input)
torch.cuda.synchronize()
end = time.time()

baseline_ms = (end - start) / 1000 * 1000
print(f"Baseline inference: {baseline_ms:.4f} ms per image")

# TorchScript — compile model into optimized representation
scripted_model = torch.jit.script(model)

# Warmup
for _ in range(10):
    _ = scripted_model(dummy_input)

# Benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    _ = scripted_model(dummy_input)
torch.cuda.synchronize()
end = time.time()

scripted_ms = (end - start) / 1000 * 1000
print(f"TorchScript inference: {scripted_ms:.4f} ms per image")
print(f"Speedup: {baseline_ms / scripted_ms:.2f}x")

# Save the scripted model
torch.jit.save(scripted_model, "models/mnist_cnn_scripted.pt")
print("Scripted model saved to models/mnist_cnn_scripted.pt")




# Quantization — convert float32 weights to int8
# This runs on CPU (PyTorch dynamic quantization works on CPU)
model_cpu = CNN()
model_cpu.load_state_dict(torch.load("models/mnist_cnn.pth"))
model_cpu.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model_cpu,
    {nn.Linear},       # quantize Linear layers
    dtype=torch.qint8  # use 8-bit integers
)

# Check model size difference
import os
torch.save(model_cpu.state_dict(), "models/mnist_cnn_fp32.pth")
torch.save(quantized_model.state_dict(), "models/mnist_cnn_int8.pth")

fp32_size = os.path.getsize("models/mnist_cnn_fp32.pth") / 1024
int8_size = os.path.getsize("models/mnist_cnn_int8.pth") / 1024

print(f"\nModel size comparison:")
print(f"  float32: {fp32_size:.1f} KB")
print(f"  int8:    {int8_size:.1f} KB")
print(f"  Reduction: {(1 - int8_size/fp32_size):.1%} smaller")

# Benchmark quantized on CPU
dummy_cpu = torch.randn(1, 1, 28, 28)

start = time.time()
for _ in range(1000):
    _ = quantized_model(dummy_cpu)
end = time.time()

quantized_ms = (end - start) / 1000 * 1000
print(f"\nQuantized CPU inference: {quantized_ms:.4f} ms per image")
