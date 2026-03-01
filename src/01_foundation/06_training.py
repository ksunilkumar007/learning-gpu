import torch
import torch.nn as nn

device = torch.device("cuda")

# We'll teach the network: y = 2x + 1
# Give it X, it should learn to predict y

X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]).to(device)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]]).to(device)

print("Input X:", X.shape)
print("Target y:", y.shape)

# Define the model
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 8)   # 1 input → 8 neurons
        self.layer2 = nn.Linear(8, 1)   # 8 neurons → 1 output

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

model = SimpleNet().to(device)

# Loss function — measures how wrong the prediction is
loss_fn = nn.MSELoss()

# Optimizer — adjusts weights to reduce loss
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

print("Model ready on:", next(model.parameters()).device)

# Training loop
for epoch in range(1000):
    # 1. Forward pass — make a prediction
    prediction = model(X)

    # 2. Calculate loss — how wrong are we?
    loss = loss_fn(prediction, y)

    # 3. Zero gradients — clear previous step
    optimizer.zero_grad()

    # 4. Backward pass — figure out blame
    loss.backward()

    # 5. Update weights — adjust parameters
    optimizer.step()

    # Print progress every 200 epochs
    if epoch % 200 == 0:
        print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f}")

# Test the model
model.eval()  # switch off training mode
with torch.no_grad():  # don't track gradients during testing
    test_inputs = torch.tensor([[6.0], [7.0], [10.0]]).to(device)
    predictions = model(test_inputs)

    print("\nTesting learned rule (y = 2x + 1):")
    for x, pred in zip([6, 7, 10], predictions):
        print(f"  x={x} → predicted={pred.item():.2f}, expected={2*x+1}")        
