import torch
import torch.nn as nn

device = torch.device("cuda")

# Simple model we'll save and reload
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 8)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

# Train quickly on y = 2x + 1
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]).to(device)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]]).to(device)

model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(1000):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print(f"Training done | Final loss: {loss.item():.4f}")

# Save the model weights
torch.save(model.state_dict(), "models/simplenet.pth")
print("Model saved to models/simplenet.pth")

# Load the model back
loaded_model = SimpleNet().to(device)
loaded_model.load_state_dict(torch.load("models/simplenet.pth"))
loaded_model.eval()

print("\nModel loaded from models/simplenet.pth")

# Test the loaded model
with torch.no_grad():
    test_inputs = torch.tensor([[6.0], [7.0], [10.0]]).to(device)
    predictions = loaded_model(test_inputs)

    print("\nPredictions from loaded model (y = 2x + 1):")
    for x, pred in zip([6, 7, 10], predictions):
        print(f"  x={x} → predicted={pred.item():.2f}, expected={2*x+1}")
