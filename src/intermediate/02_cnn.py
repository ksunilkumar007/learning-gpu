import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda")

# A single convolution layer
conv = nn.Conv2d(
    in_channels=1,    # 1 input channel (grayscale)
    out_channels=8,   # learn 8 different filters
    kernel_size=3,    # each filter is 3x3 pixels
    padding=1         # keep output same size as input
)

# Simulate one batch of one grayscale 28x28 image
x = torch.randn(1, 1, 28, 28)  # [batch, channels, height, width]
output = conv(x)

print("Input shape: ", x.shape)
print("Output shape:", output.shape)
print("Filters learned:", conv.weight.shape)

# MaxPooling — shrinks the image by taking the max value in each 2x2 block
pool = nn.MaxPool2d(kernel_size=2)  # 2x2 blocks

pooled = pool(output)
print("After pooling:", pooled.shape)  # should be [1, 8, 14, 14]

# Full CNN for MNIST
class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Convolutional layers — extract features
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)   # [1,28,28] → [8,28,28]
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # [8,14,14] → [16,14,14]
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2)                               # halves the size

        # Fully connected layers — classify
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # 16 channels, 7x7 after 2 pools
        self.fc2 = nn.Linear(64, 10)           # 10 digit classes

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # conv → relu → pool
        x = self.pool(self.relu(self.conv2(x)))  # conv → relu → pool
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = CNN().to(device)
print(model)
print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))

# Load MNIST
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train for 5 epochs
for epoch in range(5):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        prediction = model(images)
        loss = loss_fn(prediction, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/5 | Loss: {avg_loss:.4f}")

# Test accuracy
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        predicted = outputs.argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy: {correct/total:.2%}")
print(f"Correct: {correct} / {total}")
