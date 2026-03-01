import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda")

# CNN architecture
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

# Train and save the CNN on MNIST
transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)

cnn = CNN().to(device)
optimizer = torch.optim.Adam(cnn.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("Training CNN to save...")
for epoch in range(3):
    cnn.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        pred = cnn(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/3 | Loss: {total_loss/len(train_loader):.4f}")

torch.save(cnn.state_dict(), "models/mnist_cnn.pth")
print("CNN saved to models/mnist_cnn.pth\n")

# Load the pretrained model
model = CNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth"))

print("Loaded model parameters:")
for name, param in model.named_parameters():
    print(f"  {name:30s} | shape: {str(param.shape):25s} | requires_grad: {param.requires_grad}")

# Freeze all layers except the final classifier
for name, param in model.named_parameters():
    if "fc2" not in name:        # freeze everything except last layer
        param.requires_grad = False

print("After freezing:")
for name, param in model.named_parameters():
    print(f"  {name:30s} | requires_grad: {param.requires_grad}")

# Count trainable vs frozen parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
frozen    = sum(p.numel() for p in model.parameters() if not p.requires_grad)
print(f"\nTrainable parameters: {trainable}")
print(f"Frozen parameters:    {frozen}")


# Replace last layer for a new task — classify only digits 0-4 (5 classes)
model.fc2 = nn.Linear(64, 5).to(device)

# Create dataset with only digits 0-4
def filter_classes(dataset, classes):
    idx = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return torch.utils.data.Subset(dataset, idx)

train_subset = filter_classes(train_data, classes=[0, 1, 2, 3, 4])
train_subset_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

# Only optimize the new last layer
optimizer = torch.optim.Adam(model.fc2.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

print("Transfer learning — retraining last layer only:")
for epoch in range(3):
    model.train()
    total_loss = 0
    for images, labels in train_subset_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/3 | Loss: {total_loss/len(train_subset_loader):.4f}")

# Test accuracy on digits 0-4
test_data = datasets.MNIST(root="data", train=False, transform=transform)
test_subset = filter_classes(test_data, classes=[0, 1, 2, 3, 4])
test_loader = DataLoader(test_subset, batch_size=32, shuffle=False)

model.eval()
correct = total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        predicted = model(images).argmax(dim=1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

print(f"\nTest Accuracy (digits 0-4): {correct/total:.2%}")
print(f"Correct: {correct} / {total}")
