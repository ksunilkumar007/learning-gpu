import torch
import torch.nn as nn

device = torch.device("cuda")
torch.manual_seed(42)

# Small dataset — only 20 samples (easy to overfit)
X_train = torch.randn(20, 10).to(device)   # 20 samples, 10 features
y_train = torch.randint(0, 2, (20,)).to(device).float()

X_test = torch.randn(100, 10).to(device)   # 100 test samples
y_test = torch.randint(0, 2, (100,)).to(device).float()

print("Train samples:", X_train.shape)
print("Test samples: ", X_test.shape)

# Model WITHOUT dropout — will overfit
class OverfitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.sigmoid(self.fc3(x)).squeeze()

# Model WITH dropout — will generalize better
class DropoutNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)  # randomly zero 50% of neurons
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)               # drop neurons during training
        x = self.relu(self.fc2(x))
        x = self.dropout(x)               # drop neurons during training
        return self.sigmoid(self.fc3(x)).squeeze()

model_overfit  = OverfitNet().to(device)
model_dropout  = DropoutNet().to(device)

print("OverfitNet params:", sum(p.numel() for p in model_overfit.parameters()))
print("DropoutNet params:", sum(p.numel() for p in model_dropout.parameters()))

def train_and_evaluate(model, name, epochs=500):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        # Training
        model.train()
        pred = model(X_train)
        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Final evaluation
    model.eval()
    with torch.no_grad():
        # Training accuracy
        train_pred = (model(X_train) > 0.5).float()
        train_acc = (train_pred == y_train).float().mean()

        # Test accuracy
        test_pred = (model(X_test) > 0.5).float()
        test_acc = (test_pred == y_test).float().mean()

    print(f"\n{name}:")
    print(f"  Train accuracy: {train_acc.item():.2%}")
    print(f"  Test accuracy:  {test_acc.item():.2%}")
    print(f"  Gap:            {(train_acc - test_acc).item():.2%}")

train_and_evaluate(model_overfit, "Without Dropout")
train_and_evaluate(model_dropout, "With Dropout")

# Real patterned data — y = 1 if sum of features > 0 else 0
torch.manual_seed(42)

X_train2 = torch.randn(20, 10).to(device)
y_train2 = (X_train2.sum(dim=1) > 0).float()

X_test2 = torch.randn(500, 10).to(device)
y_test2 = (X_test2.sum(dim=1) > 0).float()

model_overfit2 = OverfitNet().to(device)
model_dropout2 = DropoutNet().to(device)

print("\n--- With real patterned data ---")
train_and_evaluate(model_overfit2, "Without Dropout", epochs=1000)
train_and_evaluate(model_dropout2, "With Dropout", epochs=1000)

# More training data — 200 samples instead of 20
torch.manual_seed(42)

X_train3 = torch.randn(200, 10).to(device)
y_train3 = (X_train3.sum(dim=1) > 0).float()

X_test3 = torch.randn(500, 10).to(device)
y_test3 = (X_test3.sum(dim=1) > 0).float()

model_overfit3 = OverfitNet().to(device)
model_dropout3 = DropoutNet().to(device)

def train_and_evaluate2(model, name, X_tr, y_tr, X_te, y_te, epochs=1000):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.BCELoss()

    for epoch in range(epochs):
        model.train()
        pred = model(X_tr)
        loss = loss_fn(pred, y_tr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    with torch.no_grad():
        train_acc = ((model(X_tr) > 0.5).float() == y_tr).float().mean()
        test_acc  = ((model(X_te) > 0.5).float() == y_te).float().mean()

    print(f"\n{name}:")
    print(f"  Train accuracy: {train_acc.item():.2%}")
    print(f"  Test accuracy:  {test_acc.item():.2%}")
    print(f"  Gap:            {(train_acc - test_acc).item():.2%}")

print("\n--- 200 training samples ---")
train_and_evaluate2(model_overfit3, "Without Dropout", X_train3, y_train3, X_test3, y_test3)
train_and_evaluate2(model_dropout3, "With Dropout",    X_train3, y_train3, X_test3, y_test3)
