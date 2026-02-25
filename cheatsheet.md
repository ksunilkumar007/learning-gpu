# GPU ML Cheatsheet
> RHEL 9 · NVIDIA L4 · PyTorch · CUDA 12.4

---

## The Universal ML Pattern

```
1. DATA     → load → tensor → DataLoader
2. MODEL    → nn.Module → layers → forward()
3. TRAIN    → predict → loss → backward → step
4. EVALUATE → model.eval() → no_grad → accuracy
5. DEPLOY   → state_dict → pipeline → optimize
```

---

## The Template

```python
import torch
import torch.nn as nn

device = torch.device("cuda")

# 1. DATA ─────────────────────────────────────────────────────────────────────
X = torch.tensor(...).to(device)
y = torch.tensor(...).to(device)

# 2. MODEL ────────────────────────────────────────────────────────────────────
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in, out)
        self.relu   = nn.ReLU()
        self.layer2 = nn.Linear(out, classes)

    def forward(self, x):
        x = self.relu(self.layer1(x))
        return self.layer2(x)

model = MyModel().to(device)

# 3. TRAIN ────────────────────────────────────────────────────────────────────
loss_fn   = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    pred = model(X)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()   # clear old gradients
    loss.backward()         # compute new gradients
    optimizer.step()        # update weights

# 4. EVALUATE ─────────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    predictions = model(X_test)
    predicted   = predictions.argmax(dim=1)
    accuracy    = (predicted == y_test).float().mean()

# 5. SAVE / LOAD ──────────────────────────────────────────────────────────────
torch.save(model.state_dict(), "models/mymodel.pth")

model.load_state_dict(torch.load("models/mymodel.pth"))
```

---

## Decision Tree

### What loss function?

| Output | Loss Function | Final Activation |
|--------|--------------|-----------------|
| A number (regression) | `nn.MSELoss()` | none |
| Yes or No (binary) | `nn.BCELoss()` | `nn.Sigmoid()` |
| One of N classes | `nn.CrossEntropyLoss()` | none |

### What layers?

| Data type | Layers to use |
|-----------|--------------|
| Flat numbers | `nn.Linear` |
| Images | `nn.Conv2d` + `nn.MaxPool2d` |
| Sequences | `nn.LSTM` / `nn.Transformer` |

### Overfitting?

```
1. More data          → always the best fix
2. nn.Dropout(p=0.5)  → randomly zero neurons during training
3. Simpler model      → fewer layers / neurons
```

### Training too slow or unstable?

```
1. Use DataLoader     → batch_size=32
2. Lower lr           → ReduceLROnPlateau
3. Transfer learning  → freeze layers, retrain last
```

---

## Activation Functions

| Function | Output Range | Use case |
|----------|-------------|----------|
| `nn.ReLU()` | 0 to ∞ | Hidden layers — most common |
| `nn.Sigmoid()` | 0 to 1 | Binary classification output |
| `nn.Tanh()` | -1 to 1 | Hidden layers, RNNs |

---

## CNN Pattern

```python
# conv → relu → pool  (repeat)  → flatten → linear
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.relu  = nn.ReLU()
        self.flat  = nn.Flatten()
        self.fc1   = nn.Linear(16 * 7 * 7, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # → [8, 14, 14]
        x = self.pool(self.relu(self.conv2(x)))  # → [16, 7, 7]
        x = self.relu(self.fc1(self.flat(x)))
        return self.fc2(x)
```

---

## Transfer Learning Pattern

```python
# 1. Load pretrained model
model.load_state_dict(torch.load("models/pretrained.pth"))

# 2. Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# 3. Replace last layer for new task
model.fc2 = nn.Linear(64, new_num_classes).to(device)

# 4. Train only last layer
optimizer = torch.optim.Adam(model.fc2.parameters(), lr=0.001)
```

---

## Pipeline Pattern

```python
class TrainingPipeline:
    def __init__(self, lr=0.001): ...   # build model
    def train(self, epochs):     ...   # training loop
    def save(self, path):        ...   # torch.save()

class InferencePipeline:
    def __init__(self, path):    ...   # load model + transform
    def predict(self, image):    ...   # single prediction
    def predict_batch(self, imgs): ... # batch prediction

if __name__ == "__main__":
    trainer = TrainingPipeline()
    trainer.train(epochs=5)
    trainer.save("models/model.pth")

    pipeline = InferencePipeline("models/model.pth")
    result   = pipeline.predict(image)
```

---

## Optimization Cheatsheet

```python
# TorchScript — faster inference
scripted = torch.jit.script(model)
torch.jit.save(scripted, "model_scripted.pt")

# Quantization — smaller model (float32 → int8)
quantized = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)

# Benchmarking — always warmup first
for _ in range(10): _ = model(x)          # warmup
torch.cuda.synchronize()
start = time.time()
for _ in range(1000): _ = model(x)
torch.cuda.synchronize()
ms = (time.time() - start) / 1000 * 1000
```

---

## GPU vs CPU Rules

```
Batch size = 1    → CPU may be faster (GPU overhead)
Batch size = 32+  → GPU starts winning
Batch size = 512  → GPU wins by 86x
GPU time          → stays flat (parallel)
CPU time          → grows linearly (sequential)
```

---

## Results Summary

| Block | Script | Result |
|-------|--------|--------|
| 09 | `09_mnist.py` | 97.59% — flat network |
| 11 | `02_cnn.py` | 98.79% — CNN, half the params |
| 14 | `01_transfer_learning.py` | 99.77% — 650 params only |
| 15 | `02_model_optimization.py` | 1.4x faster, 71% smaller |
| 17 | `04_gpu_vs_cpu_benchmark.py` | 86x speedup at batch=512 |

---
