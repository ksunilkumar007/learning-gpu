import torch
import torch.nn as nn

device = torch.device("cuda")
torch.manual_seed(42)

# Simple dataset
X = torch.randn(100, 4).to(device)
y = (X.sum(dim=1) > 0).float().to(device)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.fc2(self.relu(self.fc1(x)))).squeeze()

model = SimpleNet().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # high lr
loss_fn = nn.BCELoss()

print("Learning rate schedule:")
for epoch in range(5):
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    lr = optimizer.param_groups[0]['lr']
    print(f"  Epoch {epoch+1} | Loss: {loss.item():.4f} | LR: {lr:.6f}")


# StepLR — reduce LR by gamma every step_size epochs
model2 = SimpleNet().to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer2,
    step_size=2,   # reduce every 2 epochs
    gamma=0.5      # multiply LR by 0.5 each time
)

print("\nWith StepLR scheduler:")
for epoch in range(6):
    pred = model2(X)
    loss = loss_fn(pred, y)
    optimizer2.zero_grad()
    loss.backward()
    optimizer2.step()
    scheduler.step()          # update the learning rate
    lr = optimizer2.param_groups[0]['lr']
    print(f"  Epoch {epoch+1} | Loss: {loss.item():.4f} | LR: {lr:.6f}")

# ReduceLROnPlateau — reduce LR when loss stops improving
model3 = SimpleNet().to(device)
optimizer3 = torch.optim.Adam(model3.parameters(), lr=0.1)
scheduler3 = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer3,
    patience=2,    # wait 2 epochs before reducing
    factor=0.5    # multiply LR by 0.5
)

print("\nWith ReduceLROnPlateau:")
for epoch in range(20):
    pred = model3(X)
    loss = loss_fn(pred, y)
    optimizer3.zero_grad()
    loss.backward()
    optimizer3.step()
    scheduler3.step(loss)     # pass loss to scheduler
    lr = optimizer3.param_groups[0]['lr']
    if epoch % 4 == 0:
        print(f"  Epoch {epoch+1:2d} | Loss: {loss.item():.4f} | LR: {lr:.6f}")    
