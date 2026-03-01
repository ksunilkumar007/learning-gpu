import torch
import torch.nn as nn
import time

# --Architecture-------
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

def benchmark(model, device, batch_size, runs=100):
    model = model.to(device)
    model.eval()
    dummy = torch.randn(batch_size, 1, 28, 28).to(device)

    # Warmup
    for _ in range(10):
        _ = model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(runs):
        with torch.no_grad():
            _ = model(dummy)
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = (time.time() - start) / runs * 1000
    return elapsed

# Load model
cpu = torch.device("cpu")
gpu = torch.device("cuda")

model_cpu = CNN()
model_cpu.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=cpu))

model_gpu = CNN()
model_gpu.load_state_dict(torch.load("models/mnist_cnn.pth", map_location=gpu))

print(f"{'Batch Size':>12} | {'CPU (ms)':>10} | {'GPU (ms)':>10} | {'Speedup':>10}")
print("-" * 52)

for batch_size in [1, 8, 32, 128, 512, 1024]:
    cpu_ms = benchmark(model_cpu, cpu, batch_size)
    gpu_ms = benchmark(model_gpu, gpu, batch_size)
    speedup = cpu_ms / gpu_ms
    print(f"{batch_size:>12} | {cpu_ms:>10.2f} | {gpu_ms:>10.2f} | {speedup:>9.1f}x")
