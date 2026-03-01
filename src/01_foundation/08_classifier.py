import torch
import torch.nn as nn

device = torch.device("cuda")

# Create two groups of data
# Class 0: points centered around (2, 2)
# Class 1: points centered around (6, 6)
torch.manual_seed(42)

class0 = torch.randn(50, 2) + 2.0   # 50 points near (2,2)
class1 = torch.randn(50, 2) + 6.0   # 50 points near (6,6)

# Combine into one dataset
X = torch.cat([class0, class1]).to(device)
y = torch.cat([torch.zeros(50), torch.ones(50)]).to(device)

print("X shape:", X.shape)   # 100 points, 2 features each
print("y shape:", y.shape)   # 100 labels (0 or 1)
print("Class 0 samples:", (y == 0).sum().item())
print("Class 1 samples:", (y == 1).sum().item())

# Classifier network
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)  # 2 inputs (x,y coords) → 16 neurons
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(16, 1)  # 16 neurons → 1 output (0 or 1)
        self.sigmoid = nn.Sigmoid()      # squash output to 0-1 (probability)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        x = self.sigmoid(x)             # output = probability of being class 1
        return x

model = Classifier().to(device)
loss_fn = nn.BCELoss()                  # Binary Cross Entropy — for 0/1 problems
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Model ready:")
print(model)

# Training loop
for epoch in range(1000):
    model.train()

    # Forward pass
    prediction = model(X).squeeze()     # squeeze removes extra dimension
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        # Calculate accuracy
        predicted_class = (prediction > 0.5).float()  # above 0.5 = class 1
        accuracy = (predicted_class == y).float().mean()
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Accuracy: {accuracy.item():.2%}")


# Test on new unseen points
model.eval()
with torch.no_grad():
    test_points = torch.tensor([
        [2.0, 2.0],   # should be class 0
        [6.0, 6.0],   # should be class 1
        [1.5, 1.8],   # should be class 0
        [5.5, 6.2],   # should be class 1
    ]).to(device)

    probs = model(test_points).squeeze()

    print("\nPredictions on new points:")
    for point, prob in zip(test_points, probs):
        predicted = 1 if prob > 0.5 else 0
        print(f"  Point {point.tolist()} → Class {predicted} (confidence: {prob.item():.2%})")        
