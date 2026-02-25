import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda")

# Download and prepare MNIST dataset
transform = transforms.ToTensor()  # converts images to tensors

train_data = datasets.MNIST(
    root="data",        # where to save
    train=True,         # training set
    download=True,      # download if not present
    transform=transform
)

test_data = datasets.MNIST(
    root="data",
    train=False,        # test set
    download=True,
    transform=transform
)

print("Training samples:", len(train_data))
print("Test samples:", len(test_data))
print("Image shape:", train_data[0][0].shape)
print("Label example:", train_data[0][1])

# DataLoader - feeds data in batches during training
train_loader = DataLoader(
    train_data,
    batch_size=32,   # process 32 images at a time
    shuffle=True     # shuffle order every epoch
)

test_loader = DataLoader(
    test_data,
    batch_size=32,
    shuffle=False    # no need to shuffle test data
)

# Peek at one batch
images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)  # [32, 1, 28, 28]
print("Batch label shape:", labels.shape)  # [32]
print("Labels in batch:", labels)

# Flatten the 28x28 image into a 784-long vector
# then pass through fully connected layers
class MNISTNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()          # [1, 28, 28] → [784]
        self.layer1 = nn.Linear(784, 128)    # 784 inputs → 128 neurons
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(128, 64)     # 128 → 64 neurons
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(64, 10)      # 64 → 10 outputs (one per digit)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        x = self.layer3(x)                  # no sigmoid — CrossEntropyLoss handles it
        return x

model = MNISTNet().to(device)
loss_fn = nn.CrossEntropyLoss()             # for multi-class (10 digits)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(model)
print("\nTotal parameters:", sum(p.numel() for p in model.parameters()))

# Training loop
for epoch in range(5):  # 5 passes through all 60,000 images
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


# Test accuracy on 10,000 unseen images
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        predicted = outputs.argmax(dim=1)  # pick digit with highest score

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

accuracy = correct / total
print(f"\nTest Accuracy: {accuracy:.2%}")
print(f"Correct: {correct} / {total}")    
