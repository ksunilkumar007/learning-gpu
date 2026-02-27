# Learning GPU Programming

## Environment
- OS: RHEL 9
- GPU: NVIDIA L4 (23GB VRAM)
- Driver: 550.163.01
- CUDA: 12.4
- Python: 3.9.18
- Package Manager: uv 0.10.6

---

## Step 1 — Provision the RHEL 9 with NVIDIA GPU

Verify repos, GPU driver, and Python are ready:

```bash
yum repolist
```

```
repo id                                                                        repo name
codeready-builder-for-rhel-9-x86_64-eus-rpms                                   Red Hat CodeReady Linux Builder for RHEL 9 x86_64 - Extended Update Support (RPMs)
rhel-9-for-x86_64-appstream-eus-rpms                                           Red Hat Enterprise Linux 9 for x86_64 - AppStream - Extended Update Support (RPMs)
rhel-9-for-x86_64-baseos-eus-rpms                                              Red Hat Enterprise Linux 9 for x86_64 - BaseOS - Extended Update Support (RPMs)
```

```bash
nvidia-smi
```

```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|=========================================+========================+======================|
|   0  NVIDIA L4                      On  |   00000000:31:00.0 Off |                    0 |
| N/A   32C    P8             11W /   72W |       1MiB /  23034MiB |      0%      Default |
+-----------------------------------------+------------------------+----------------------+
```

```bash
python3 --version
# Python 3.9.18
```

---

## Step 2 — Install uv Package Manager

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv --version
# uv 0.10.6
```

---

## Step 3 — Install Dependencies

```bash
mkdir -p gpu-learning && cd gpu-learning

# Create virtual environment
uv venv
source .venv/bin/activate

# Install PyTorch with CUDA 12.4 support
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

> **NOTE:** By default, `uv` (and `pip`) downloads packages from **PyPI** — the standard Python
> package repository. The PyTorch on PyPI is the **CPU-only** version.
> NVIDIA GPU-enabled versions of PyTorch are hosted on PyTorch's own server at
> `download.pytorch.org/whl/cu124`, not PyPI. The `cu124` means "CUDA 12.4" — matching exactly
> what your GPU has. Without `--index-url` you'd install PyTorch but it would never use your GPU.
> With it, you get the GPU-accelerated build.

### Installed Packages

| Package | Version |
|---------|---------|
| torch | 2.6.0+cu124 |
| torchvision | 0.21.0+cu124 |
| torchaudio | 2.6.0+cu124 |
| numpy | 2.0.2 |
| triton | 3.2.0 |
| nvidia-cudnn-cu12 | 9.1.0.70 |
| nvidia-nccl-cu12 | 2.21.5 |
| pillow | 11.3.0 |
| sympy | 1.13.1 |
| + 18 other CUDA/utility packages | — |

## Step 4 Learning basics of pytorch

### Check the GPU details
```python
(gpu-learning) [cloud-user@bastion gpu-learning]$ touch verify_gpu.py
(gpu-learning) [cloud-user@bastion gpu-learning]$ cat verify_gpu.py 
import torch

print("Pytorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name())
(gpu-learning) [cloud-user@bastion gpu-learning]$ uv run verify_gpu.py 
Pytorch Version: 2.6.0+cu124
CUDA Available: True
CUDA Device Count: 1
GPU Name: NVIDIA L4
(gpu-learning) [cloud-user@bastion gpu-learning]$ 

```
### Create Tensors
> **NOTE:** A tensor is just a container for numbers. Think of it like this:
>
> ```
> Scalar (0D tensor) →  42
> Vector (1D tensor) →  [1, 2, 3]
> Matrix (2D tensor) →  [[1, 2], [3, 4]]
> 3D tensor          →  [[[1,2],[3,4]], [[5,6],[7,8]]]
> ```
>
> The magic is that tensors can live **on your GPU** — meaning all the math on them runs on the GPU automatically.

```python
(gpu-learning) [cloud-user@bastion gpu-learning]$ cat tensors.py 
import torch

# Create a simple tensor (lives on CPU by default)
t = torch.tensor([1.0, 2.0, 3.0])

print("Tensors:", t)
print("Shape:", t.shape)
print("Data Type:",t.dtype)
print("Device:", t.device)
(gpu-learning) [cloud-user@bastion gpu-learning]$ uv run tensors.py 
Tensors: tensor([1., 2., 3.])
Shape: torch.Size([3])
Data Type: torch.float32
Device: cpu
(gpu-learning) [cloud-user@bastion gpu-learning]$ 
```
> **NOTE:** Move tensor to GPU by default its in CPU
The :0 means GPU index 0 — if you had multiple GPUs it would be cuda:1, cuda:2, etc.
>
> ```
> One important concept — tensors on different devices can't interact:
> Both tensors must be on the same device before you can do any math between them. This is one of the most common beginner errors in PyTorch.
>
>```
>

```python
(gpu-learning) [cloud-user@bastion gpu-learning]$ cat tensors.py 
import torch

# Create a simple tensor (lives on CPU by default)
t = torch.tensor([1.0, 2.0, 3.0])

print("Tensors:", t)
print("Shape:", t.shape)
print("Data Type:",t.dtype)
print("Device:", t.device)


# Move tensor to GPU
t_gpu = t.to("cuda")

print("\nAfter moving to GPU:")
print("Device:", t_gpu.device)
(gpu-learning) [cloud-user@bastion gpu-learning]$ uv run tensors.py 
Tensors: tensor([1., 2., 3.])
Shape: torch.Size([3])
Data Type: torch.float32
Device: cpu

After moving to GPU:
Device: cuda:0
(gpu-learning) [cloud-user@bastion gpu-learning]$ 

```

> **NOTE:** Add tensors
>```
>
>A tensor is a container for numbers
>By default tensors live on CPU
>.to("cuda") moves a tensor to the GPU
>Math between two GPU tensors produces a GPU tensor
>Never mix CPU and GPU tensors in the same operation
>```
>
```python
(gpu-learning) [cloud-user@bastion gpu-learning]$ cat tensors.py 
import torch

# Create a simple tensor (lives on CPU by default)
t = torch.tensor([1.0, 2.0, 3.0])

print("Tensors:", t)
print("Shape:", t.shape)
print("Data Type:",t.dtype)
print("Device:", t.device)


# Move tensor to GPU
t_gpu = t.to("cuda")

print("\nAfter moving to GPU:")
print("Device:", t_gpu.device)

# Do math directly on the GPU both ensors needs to be in same device
a = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
b = torch.tensor([4.0, 5.0, 6.0]).to("cuda")

c = a + b

print("\nGPU Math:")
print("a + b =", c)
print("Device:", c.device)
(gpu-learning) [cloud-user@bastion gpu-learning]$ uv run tensors.py 
Tensors: tensor([1., 2., 3.])
Shape: torch.Size([3])
Data Type: torch.float32
Device: cpu

After moving to GPU:
Device: cuda:0

GPU Math:
a + b = tensor([5., 7., 9.], device='cuda:0')
Device: cuda:0
(gpu-learning) [cloud-user@bastion gpu-learning]$ 

```
## Step 5 — Shapes and Dimensions

> **NOTE:**
>
> ```
> A scalar is a tensor with 0 dimensions
> A vector is a tensor with 1 or more dimensions
> ```

### Scalar (0D)

```python
[cloud-user@bastion gpu-learning]$ cat shapes.py 
import torch

scalar = torch.tensor(43.0)
print("Scalar:", scalar)
print("Shapes:", scalar.shape)
print("Dimensions:",scalar.ndim)
[cloud-user@bastion gpu-learning]$
```

```
[cloud-user@bastion gpu-learning]$ uv run shapes.py 
Scalar: tensor(43.)
Shapes: torch.Size([])
Dimensions: 0
[cloud-user@bastion gpu-learning]$ 
```

> `torch.Size([])` — empty brackets mean no dimensions, just a single number.

### Vector (1D)
```python
[cloud-user@bastion gpu-learning]$ cat shapes.py 
import torch

scalar = torch.tensor(43.0)
print("Scalar:", scalar)
print("Shapes:", scalar.shape)
print("Dimensions:",scalar.ndim)

# 1D - Vector (a list of numbers)
vector = torch.tensor([1.0, 2.0, 3.0])
print("\nVector:", vector)
print("Shape:", vector.shape)
print("Dimensions:", vector.ndim)

[cloud-user@bastion gpu-learning]$ uv run shapes.py 
Scalar: tensor(43.)
Shapes: torch.Size([])
Dimensions: 0

Vector: tensor([1., 2., 3.])
Shape: torch.Size([3])
Dimensions: 1
[cloud-user@bastion gpu-learning]$ 

```
### Vector (2D)
```python
[cloud-user@bastion gpu-learning]$ cat shapes.py 
import torch

scalar = torch.tensor(43.0)
print("Scalar:", scalar)
print("Shapes:", scalar.shape)
print("Dimensions:",scalar.ndim)

# 1D - Vector (a list of numbers)
vector = torch.tensor([1.0, 2.0, 3.0])
print("\nVector:", vector)
print("Shape:", vector.shape)
print("Dimensions:", vector.ndim)

# 2D - Matrix (rows and columns)
matrix = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
print("\nMatrix:", matrix)
print("Shape:", matrix.shape)
print("Dimensions:", matrix.ndim)

[cloud-user@bastion gpu-learning]$ uv run shapes.py 
Scalar: tensor(43.)
Shapes: torch.Size([])
Dimensions: 0

Vector: tensor([1., 2., 3.])
Shape: torch.Size([3])
Dimensions: 1

Matrix: tensor([[1., 2., 3.],
        [4., 5., 6.]])
Shape: torch.Size([2, 3])
Dimensions: 2
[cloud-user@bastion gpu-learning]$ 
```
### Vector (3D)
```python
[cloud-user@bastion gpu-learning]$ cat shapes.py 
import torch

scalar = torch.tensor(43.0)
print("Scalar:", scalar)
print("Shapes:", scalar.shape)
print("Dimensions:",scalar.ndim)

# 1D - Vector (a list of numbers)
vector = torch.tensor([1.0, 2.0, 3.0])
print("\nVector:", vector)
print("Shape:", vector.shape)
print("Dimensions:", vector.ndim)

# 2D - Matrix (rows and columns)
matrix = torch.tensor([[1.0, 2.0, 3.0],
                        [4.0, 5.0, 6.0]])
print("\nMatrix:", matrix)
print("Shape:", matrix.shape)
print("Dimensions:", matrix.ndim)

# 3D - Stack of matrices (think: multiple images)
tensor_3d = torch.tensor([[[1.0, 2.0],
                            [3.0, 4.0]],
                           [[5.0, 6.0],
                            [7.0, 8.0]]])
print("\n3D Tensor:", tensor_3d)
print("Shape:", tensor_3d.shape)
print("Dimensions:", tensor_3d.ndim)

[cloud-user@bastion gpu-learning]$ uv run shapes.py 
Scalar: tensor(43.)
Shapes: torch.Size([])
Dimensions: 0

Vector: tensor([1., 2., 3.])
Shape: torch.Size([3])
Dimensions: 1

Matrix: tensor([[1., 2., 3.],
        [4., 5., 6.]])
Shape: torch.Size([2, 3])
Dimensions: 2

3D Tensor: tensor([[[1., 2.],
         [3., 4.]],

        [[5., 6.],
         [7., 8.]]])
Shape: torch.Size([2, 2, 2])
Dimensions: 3
[cloud-user@bastion gpu-learning]$ 

```

## Step 6 — Tensors Operations
```python
[cloud-user@bastion gpu-learning]$ cat operations.py 
import torch

a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])

# Basic Arithmatics

print("a + b =", a+b)
print("a - b =", a-b)
print("a * b =", a*b)
print("a / b =", a/b)
[cloud-user@bastion gpu-learning]$

[cloud-user@bastion gpu-learning]$ uv run operations.py 
a + b = tensor([5., 7., 9.])
a - b = tensor([-3., -3., -3.])
a * b = tensor([ 4., 10., 18.])
a / b = tensor([0.2500, 0.4000, 0.5000])
[cloud-user@bastion gpu-learning]$ 

[cloud-user@bastion gpu-learning]$ cat operations.py 
import torch

a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])

# Basic Arithmatics

print("a + b =", a+b)
print("a - b =", a-b)
print("a * b =", a*b)
print("a / b =", a/b)

m1 = torch.tensor([[1.0,2.0],
                    [3.0,4.0]])

m2 = torch.tensor([[5.0,6.0],
                    [7.0,8.0]])

result = torch.matmul(m1,m2)

print("\nMatrix Multiplication")
print("m1 @ m2 =", result)
[cloud-user@bastion gpu-learning]$ uv run operations.py 
a + b = tensor([5., 7., 9.])
a - b = tensor([-3., -3., -3.])
a * b = tensor([ 4., 10., 18.])
a / b = tensor([0.2500, 0.4000, 0.5000])

Matrix Multiplication
m1 @ m2 = tensor([[19., 22.],
        [43., 50.]])
[cloud-user@bastion gpu-learning]$ 


[cloud-user@bastion gpu-learning]$ cat operations.py 
import torch

a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])

# Basic Arithmatics

print("a + b =", a+b)
print("a - b =", a-b)
print("a * b =", a*b)
print("a / b =", a/b)

m1 = torch.tensor([[1.0,2.0],
                    [3.0,4.0]])

m2 = torch.tensor([[5.0,6.0],
                    [7.0,8.0]])

result = torch.matmul(m1,m2)

print("\nMatrix Multiplication")
print("m1 @ m2 =", result)

# Reduction operations
t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

print("\nReduction Operations:")
print("Sum:", t.sum())
print("Mean:", t.mean())
print("Max:", t.max())
print("Min:", t.min())

[cloud-user@bastion gpu-learning]$ uv run operations.py 
a + b = tensor([5., 7., 9.])
a - b = tensor([-3., -3., -3.])
a * b = tensor([ 4., 10., 18.])
a / b = tensor([0.2500, 0.4000, 0.5000])

Matrix Multiplication
m1 @ m2 = tensor([[19., 22.],
        [43., 50.]])

Reduction Operations:
Sum: tensor(15.)
Mean: tensor(3.)
Max: tensor(5.)
Min: tensor(1.)
[cloud-user@bastion gpu-learning]$ 

[cloud-user@bastion gpu-learning]$ cat operations.py 
import torch

a = torch.tensor([1.0,2.0,3.0])
b = torch.tensor([4.0,5.0,6.0])

# Basic Arithmatics

print("a + b =", a+b)
print("a - b =", a-b)
print("a * b =", a*b)
print("a / b =", a/b)

m1 = torch.tensor([[1.0,2.0],
                    [3.0,4.0]])

m2 = torch.tensor([[5.0,6.0],
                    [7.0,8.0]])

result = torch.matmul(m1,m2)

print("\nMatrix Multiplication")
print("m1 @ m2 =", result)

# Reduction operations
t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

print("\nReduction Operations:")
print("Sum:", t.sum())
print("Mean:", t.mean())
print("Max:", t.max())
print("Min:", t.min())


# Same operations but on GPU
a_gpu = torch.tensor([1.0, 2.0, 3.0]).to("cuda")
b_gpu = torch.tensor([4.0, 5.0, 6.0]).to("cuda")

result_gpu = torch.matmul(a_gpu, b_gpu)

print("\nOn GPU:")
print("Dot product:", result_gpu)
print("Device:", result_gpu.device)

[cloud-user@bastion gpu-learning]$ uv run operations.py 
a + b = tensor([5., 7., 9.])
a - b = tensor([-3., -3., -3.])
a * b = tensor([ 4., 10., 18.])
a / b = tensor([0.2500, 0.4000, 0.5000])

Matrix Multiplication
m1 @ m2 = tensor([[19., 22.],
        [43., 50.]])

Reduction Operations:
Sum: tensor(15.)
Mean: tensor(3.)
Max: tensor(5.)
Min: tensor(1.)

On GPU:
Dot product: tensor(32., device='cuda:0')
Device: cuda:0
[cloud-user@bastion gpu-learning]$ 
```
## Step 6 — Neural Networks

> **NOTE:**
> A neural network is just layers of matrix multiplications with some math in between.
>
> ```
> Input → [Layer 1] → [Layer 2] → [Output]
> ```
>
> Each layer takes numbers in, does math, passes numbers out.

### first_nn.py
```python
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 8)   # 3 inputs → 8 neurons
        self.layer2 = nn.Linear(8, 1)   # 8 neurons → 1 output

    def forward(self, x):
        x = self.layer1(x)              # pass through layer 1
        x = self.layer2(x)              # pass through layer 2
        return x

# Create the network and move to GPU
device = torch.device("cuda")
model = SimpleNeuralNetwork().to(device)
print(model)

# Create a sample input (3 numbers) and move to GPU
sample_input = torch.tensor([1.0, 2.0, 3.0]).to(device)

# Pass it through the network (forward pass)
output = model(sample_input)
print("Input:", sample_input)
print("Output:", output)
print("Device:", output.device)

# Count total learnable parameters
# layer1: 3 × 8 = 24 weights + 8 biases = 32
# layer2: 8 × 1 =  8 weights + 1 bias   =  9
# Total = 41
total_params = sum(p.numel() for p in model.parameters())
print("\nTotal trainable parameters:", total_params)
```
```
SimpleNeuralNetwork(
  (layer1): Linear(in_features=3, out_features=8, bias=True)
  (layer2): Linear(in_features=8, out_features=1, bias=True)
)
Input: tensor([1., 2., 3.], device='cuda:0')
Output: tensor([0.3785], device='cuda:0', grad_fn=<ViewBackward0>)
Device: cuda:0

Total trainable parameters: 41
```

> `grad_fn=<ViewBackward0>` — PyTorch is tracking how this output was calculated
> so it can work backwards and improve the weights. This is called **autograd**.

## Step 7 — Training the Network

> **NOTE:** Training is the process of adjusting the network's parameters
> until it makes good predictions. The cycle is:
>
> ```
> 1. Feed data in          → forward pass
> 2. Measure how wrong     → loss
> 3. Figure out blame      → backward pass
> 4. Adjust the weights    → optimizer step
> 5. Repeat
> ```

### training.py

```python
import torch
import torch.nn as nn

device = torch.device("cuda")

# We'll teach the network: y = 2x + 1
# Give it X, it should learn to predict y
X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]]).to(device)
y = torch.tensor([[3.0], [5.0], [7.0], [9.0], [11.0]]).to(device)

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
# lr = learning rate — how big each adjustment step is
# Too big = overshoots. Too small = too slow
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

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
```

```
Input X: torch.Size([5, 1])
Target y: torch.Size([5, 1])
Model ready on: cuda:0
Epoch    0 | Loss: 59.6270
Epoch  200 | Loss: 0.0068
Epoch  400 | Loss: 0.0002
Epoch  600 | Loss: 0.0000
Epoch  800 | Loss: 0.0000

Testing learned rule (y = 2x + 1):
  x=6  → predicted=13.00, expected=13
  x=7  → predicted=15.00, expected=15
  x=10 → predicted=21.00, expected=21
```

> The network was never shown x=6, 7, or 10 during training — it
> **generalized** the rule from just 5 examples. That is machine learning.

## Step 8 — Activation Functions

> **NOTE:** Without activation functions a network can only learn straight line
> patterns. Activation functions add non-linearity so the network can learn
> curves, clusters, and complex shapes.
>
> ```
> Without activation → can only draw straight lines
> With activation    → can draw any shape
> ```

### activations.py

```python
import torch
import torch.nn as nn

# Create a simple input
x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0, 3.0])

# ReLU — most common activation function
# Rule: if x < 0 → 0, if x >= 0 → keep x
relu = nn.ReLU()
print("Input: ", x)
print("ReLU:  ", relu(x))

# Sigmoid — squashes any number between 0 and 1
# Used for binary classification (yes/no, true/false)
sigmoid = nn.Sigmoid()
print("\nSigmoid:", sigmoid(x))

# Tanh — squashes any number between -1 and 1
# Similar to sigmoid but centered at 0
tanh = nn.Tanh()
print("Tanh:   ", tanh(x))

# Network WITHOUT activation (can only learn straight lines)
class LinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 8)
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

# Network WITH activation (can learn curves)
class NonLinearNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 8)
        self.relu = nn.ReLU()          # activation between layers
        self.layer2 = nn.Linear(8, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)               # apply activation
        x = self.layer2(x)
        return x

linear_model = LinearNet()
nonlinear_model = NonLinearNet()

print("\nLinear model params:    ", sum(p.numel() for p in linear_model.parameters()))
print("Non-linear model params:", sum(p.numel() for p in nonlinear_model.parameters()))
print("\nOnly difference is ReLU between layers — but now it can learn any shape!")
```

```
Input:  tensor([-2., -1.,  0.,  1.,  2.,  3.])
ReLU :  tensor([0., 0., 0., 1., 2., 3.])

Sigmoid: tensor([0.1192, 0.2689, 0.5000, 0.7311, 0.8808, 0.9526])
Tanh:    tensor([-0.9640, -0.7616,  0.0000,  0.7616,  0.9640,  0.9951])

Linear model params:     25
Non-linear model params: 25

Only difference is ReLU between layers — but now it can learn any shape!
```

> **Key insight:** Adding ReLU costs zero extra parameters but unlocks the
> network's ability to learn complex patterns. That's why activation functions
> are placed between every layer in real networks.
>
> | Activation | Output Range | Use Case |
> |------------|-------------|----------|
> | `ReLU`    | 0 to ∞      | Hidden layers — most common |
> | `Sigmoid` | 0 to 1      | Binary classification output |
> | `Tanh`    | -1 to 1     | Hidden layers, RNNs |

## Step 9 — Binary Classifier

> **NOTE:** Binary classification teaches the network to make a decision
> between two categories. The output is a probability between 0 and 1.
>
> ```
> Input → Network → Sigmoid → probability (0 to 1) → decision (0 or 1)
> ```

### classifier.py

```python
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

# Training loop
for epoch in range(1000):
    model.train()

    prediction = model(X).squeeze()     # squeeze removes extra dimension
    loss = loss_fn(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
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
```

```
Epoch    0 | Loss: 1.1624 | Accuracy: 50.00%
Epoch  200 | Loss: 0.0167 | Accuracy: 100.00%
Epoch  400 | Loss: 0.0042 | Accuracy: 100.00%
Epoch  600 | Loss: 0.0019 | Accuracy: 100.00%
Epoch  800 | Loss: 0.0011 | Accuracy: 100.00%

Predictions on new points:
  Point [2.0, 2.0] → Class 0 (confidence: 0.00%)
  Point [6.0, 6.0] → Class 1 (confidence: 100.00%)
  Point [1.5, 1.8] → Class 0 (confidence: 0.00%)
  Point [5.5, 6.2] → Class 1 (confidence: 100.00%)
```

> **Key insight:** `BCELoss` replaces `MSELoss` for classification problems.
> Sigmoid converts the output to a probability. Anything above 0.5 = class 1,
> below 0.5 = class 0.

## Step 10 — Real Dataset (MNIST)

> **NOTE:** MNIST is a collection of 70,000 handwritten digit images (0–9),
> each 28×28 pixels. It is the "hello world" of machine learning.
>
> ```
> Input: 28×28 image → Network → Output: digit 0-9
> ```

### mnist.py

```python
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

# DataLoader — feeds data in batches during training
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

# Training loop — 5 passes through all 60,000 images
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
```

```
Training samples: 60000
Test samples: 10000
Image shape: torch.Size([1, 28, 28])
Label example: 5
Batch image shape: torch.Size([32, 1, 28, 28])
Batch label shape: torch.Size([32])
Total parameters: 109386
Epoch 1/5 | Loss: 0.2939
Epoch 2/5 | Loss: 0.1200
Epoch 3/5 | Loss: 0.0813
Epoch 4/5 | Loss: 0.0604
Epoch 5/5 | Loss: 0.0485

Test Accuracy: 97.59%
Correct: 9759 / 10000
```

> **Key insight:** `DataLoader` feeds data in batches so you never load all
> 60,000 images into memory at once. `argmax(dim=1)` picks the digit with
> the highest score from the 10 output neurons.

## Step 11 — Save & Load Models

> **NOTE:** Training takes time. Save your model so you never have to
> retrain from scratch. The recommended way is saving weights only (state_dict).
>
> ```
> 1. Train model    → takes hours/days
> 2. Save weights   → torch.save()
> 3. Share/deploy   → copy .pth file anywhere
> 4. Load & predict → torch.load() + load_state_dict()
> ```

### src/intermediate/01_save_load.py

```python
import torch
import torch.nn as nn

device = torch.device("cuda")

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
```

```
Training done | Final loss: 0.0000
Model saved to models/simplenet.pth

Model loaded from models/simplenet.pth

Predictions from loaded model (y = 2x + 1):
  x=6  → predicted=13.00, expected=13
  x=7  → predicted=15.00, expected=15
  x=10 → predicted=21.00, expected=21
```

> **Key insight:** `state_dict()` is just a dictionary of all learned weights.
> Saving and loading it means you never lose your training work.
> The `.pth` file can be copied anywhere and loaded on any machine with PyTorch.

## Step 12 — Convolutional Neural Networks (CNNs)

> **NOTE:** Flat networks lose spatial information by flattening images.
> CNNs preserve spatial structure by sliding filters across the image,
> detecting patterns like edges, curves, and shapes.
>
> ```
> Flat Network: [784 numbers] → loses spatial structure
> CNN:          [28×28 grid]  → preserves spatial structure
> ```

### src/intermediate/02_cnn.py

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda")

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
```

```
Input shape:  torch.Size([1, 1, 28, 28])
Output shape: torch.Size([1, 8, 28, 28])
Filters learned: torch.Size([8, 1, 3, 3])
After pooling: torch.Size([1, 8, 14, 14])
Epoch 1/5 | Loss: 0.2417
Epoch 2/5 | Loss: 0.0788
Epoch 3/5 | Loss: 0.0590
Epoch 4/5 | Loss: 0.0455
Epoch 5/5 | Loss: 0.0373

Test Accuracy: 98.79%
Correct: 9879 / 10000
```

> **Key insight:** CNN beats flat network with half the parameters.
>
> ```
> Flat Network → 109,386 parameters → 97.59% accuracy
> CNN          →  52,138 parameters → 98.79% accuracy
> ```
>
> The flow through the CNN:
> ```
> [1, 1, 28, 28] → conv1+relu → [1,  8, 28, 28]
>                → pool       → [1,  8, 14, 14]
>                → conv2+relu → [1, 16, 14, 14]
>                → pool       → [1, 16,  7,  7]
>                → flatten    → [1, 784]
>                → fc1+relu   → [1,  64]
>                → fc2        → [1,  10]
> ```

## Step 13 — Overfitting & Dropout

> **NOTE:** Overfitting is when a model memorizes training data instead of
> learning general patterns. It performs great on training data but fails
> on new unseen data.
>
> ```
> Underfitting → model too simple, can't learn the pattern
> Good fit     → model learns the pattern, generalizes well
> Overfitting  → model memorizes training data, fails on new data
> ```

### src/intermediate/03_overfitting_dropout.py

```python
import torch
import torch.nn as nn

device = torch.device("cuda")
torch.manual_seed(42)

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

# Model WITH dropout — generalizes better
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
```

```
--- 20 samples (random noise) ---
Without Dropout → Train: 100% | Test: 58% | Gap: 42%
With Dropout    → Train: 100% | Test: 55% | Gap: 45%

--- 200 samples (real pattern) ---
Without Dropout → Train: 100% | Test: 97.80% | Gap: 2.20%
With Dropout    → Train: 100% | Test: 95.80% | Gap: 4.20%
```

> **Key insight:** The three rules of overfitting:
>
> ```
> 1. More data        → always the best fix
> 2. Simpler model    → fewer parameters = less memorization
> 3. Dropout          → forces network to not rely on any single neuron
> ```
>
> Dropout is disabled automatically during `model.eval()` — all neurons
> are active for prediction. It only fires during `model.train()`.

## Step 14 — Learning Rate Schedulers

> **NOTE:** A fixed learning rate is not optimal. Early in training you want
> big steps, late in training you want small steps.
>
> ```
> Early training → big steps are good (far from the answer)
> Late training  → big steps are bad  (overshoot the answer)
> ```

### src/intermediate/04_lr_scheduler.py

```python
import torch
import torch.nn as nn

device = torch.device("cuda")
torch.manual_seed(42)

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

loss_fn = nn.BCELoss()

# StepLR — reduce LR by gamma every step_size epochs
model2 = SimpleNet().to(device)
optimizer2 = torch.optim.Adam(model2.parameters(), lr=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer2,
    step_size=2,   # reduce every 2 epochs
    gamma=0.5      # multiply LR by 0.5 each time
)

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
    factor=0.5,    # multiply LR by 0.5
)

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
```

```
With StepLR scheduler:
  Epoch 1 | Loss: 0.6971 | LR: 0.100000
  Epoch 2 | Loss: 0.5834 | LR: 0.050000
  Epoch 4 | Loss: 0.4129 | LR: 0.025000
  Epoch 6 | Loss: 0.3203 | LR: 0.012500

With ReduceLROnPlateau:
  Epoch  1 | Loss: 0.6816 | LR: 0.100000
  Epoch  9 | Loss: 0.0224 | LR: 0.100000
  Epoch 17 | Loss: 0.0019 | LR: 0.100000
```

> **Key insight:**
>
> | Scheduler | When to use |
> |-----------|------------|
> | `StepLR` | Reduce at fixed intervals — simple and predictable |
> | `ReduceLROnPlateau` | Reduce when loss stops improving — most practical |
> | `CosineAnnealingLR` | Smooth decay — popular in modern research |


# Step 15 — Transfer Learning

> **NOTE:** Transfer learning reuses a pretrained model's knowledge for a
> new task. Freeze early layers, replace the last layer, retrain only that.
>
> ```
> 1. Train model on large dataset  → learns general features
> 2. Freeze early layers           → requires_grad = False
> 3. Replace last layer            → nn.Linear(64, new_classes)
> 4. Retrain only last layer       → fast, needs little data
> ```

### src/advanced/01_transfer_learning.py

```python
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

device = torch.device("cuda")

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

# Load pretrained CNN
model = CNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth"))

# Freeze all layers except last
for name, param in model.named_parameters():
    if "fc2" not in name:
        param.requires_grad = False

# Replace last layer for new task (5 classes instead of 10)
model.fc2 = nn.Linear(64, 5).to(device)

# Train only on digits 0-4
def filter_classes(dataset, classes):
    idx = [i for i, (_, label) in enumerate(dataset) if label in classes]
    return torch.utils.data.Subset(dataset, idx)

transform = transforms.ToTensor()
train_data = datasets.MNIST(root="data", train=True, download=True, transform=transform)
train_subset = filter_classes(train_data, classes=[0, 1, 2, 3, 4])
train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)

# Only optimize the new last layer
optimizer = torch.optim.Adam(model.fc2.parameters(), lr=0.001)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        pred = model(images)
        loss = loss_fn(pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"  Epoch {epoch+1}/3 | Loss: {total_loss/len(train_loader):.4f}")

# Test accuracy
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
```

```
Transfer learning — retraining last layer only:
  Epoch 1/3 | Loss: 0.1079
  Epoch 2/3 | Loss: 0.0179
  Epoch 3/3 | Loss: 0.0132

Test Accuracy (digits 0-4): 99.77%
Correct: 5127 / 5139
```

> **Key insight:**
>
> ```
> Full training   → 52,138 parameters → 98.79% accuracy
> Transfer learn  →    650 parameters → 99.77% accuracy
> ```
>
> Better accuracy, fewer parameters, faster training.

---

## Step 16 — Model Optimization

> **NOTE:** A trained model can be made faster and smaller without losing
> accuracy using two main techniques:
>
> ```
> TorchScript  → compile model = faster inference
> Quantization → float32 → int8 = smaller model
> ```

### src/advanced/02_model_optimization.py

```python
import torch
import torch.nn as nn
import time
import os

device = torch.device("cuda")

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

# Load pretrained model
model = CNN().to(device)
model.load_state_dict(torch.load("models/mnist_cnn.pth"))
model.eval()

dummy_input = torch.randn(1, 1, 28, 28).to(device)

# Warmup
for _ in range(10):
    _ = model(dummy_input)

# Baseline benchmark
torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    _ = model(dummy_input)
torch.cuda.synchronize()
baseline_ms = (time.time() - start) / 1000 * 1000
print(f"Baseline inference: {baseline_ms:.4f} ms per image")

# TorchScript — compile model into optimized representation
scripted_model = torch.jit.script(model)

for _ in range(10):
    _ = scripted_model(dummy_input)

torch.cuda.synchronize()
start = time.time()
for _ in range(1000):
    _ = scripted_model(dummy_input)
torch.cuda.synchronize()
scripted_ms = (time.time() - start) / 1000 * 1000
print(f"TorchScript inference: {scripted_ms:.4f} ms per image")
print(f"Speedup: {baseline_ms / scripted_ms:.2f}x")

torch.jit.save(scripted_model, "models/mnist_cnn_scripted.pt")

# Quantization — convert float32 weights to int8 (CPU only)
model_cpu = CNN()
model_cpu.load_state_dict(torch.load("models/mnist_cnn.pth"))
model_cpu.eval()

quantized_model = torch.quantization.quantize_dynamic(
    model_cpu,
    {nn.Linear},       # quantize Linear layers
    dtype=torch.qint8  # use 8-bit integers
)

torch.save(model_cpu.state_dict(), "models/mnist_cnn_fp32.pth")
torch.save(quantized_model.state_dict(), "models/mnist_cnn_int8.pth")

fp32_size = os.path.getsize("models/mnist_cnn_fp32.pth") / 1024
int8_size = os.path.getsize("models/mnist_cnn_int8.pth") / 1024
print(f"\nModel size comparison:")
print(f"  float32: {fp32_size:.1f} KB")
print(f"  int8:    {int8_size:.1f} KB")
print(f"  Reduction: {(1 - int8_size/fp32_size):.1%} smaller")

dummy_cpu = torch.randn(1, 1, 28, 28)
start = time.time()
for _ in range(1000):
    _ = quantized_model(dummy_cpu)
quantized_ms = (time.time() - start) / 1000 * 1000
print(f"\nQuantized CPU inference: {quantized_ms:.4f} ms per image")
```

```
Baseline inference:    0.2904 ms per image
TorchScript inference: 0.2079 ms per image  → 1.40x faster
Scripted model saved to models/mnist_cnn_scripted.pt

Model size comparison:
  float32: 206.8 KB
  int8:     59.3 KB
  Reduction: 71.3% smaller

Quantized CPU inference: 0.3166 ms per image
```

> **Key insight:**
>
> | Technique | Benefit | Use case |
> |-----------|---------|----------|
> | `TorchScript` | 1.4x faster | Production GPU serving |
> | `Quantization` | 71% smaller | Edge/mobile deployment |
> | Both together | Maximum optimization | Production deployment |

---


# Step 17 — Training & Inference Pipeline

> **NOTE:** In production, training and inference are always separate concerns.
> Wrapping them in classes makes them reusable and importable.
>
> ```
> MNISTTrainingPipeline  → train() → save()
>                                       ↓
> MNISTInferencePipeline → load() → predict() → predict_batch()
> ```

### src/advanced/03_inference_pipeline.py

```python
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from PIL import Image
import time

device = torch.device("cuda")

# ── Architecture ──────────────────────────────────────────────────────────────
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

# ── Training Pipeline ─────────────────────────────────────────────────────────
class MNISTTrainingPipeline:
    def __init__(self, num_classes=10, lr=0.001):
        self.model = CNN(num_classes=num_classes).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.CrossEntropyLoss()
        self.transform = transforms.Compose([transforms.ToTensor()])
        print(f"Training pipeline ready | device: {device}")

    def train(self, epochs=5):
        train_data = datasets.MNIST(root="data", train=True, download=True, transform=self.transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                pred = self.model(images)
                loss = self.loss_fn(pred, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            print(f"  Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f}")

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"Model saved to {path}")

# ── Inference Pipeline ────────────────────────────────────────────────────────
class MNISTInferencePipeline:
    def __init__(self, model_path):
        self.model = CNN().to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
        ])
        print(f"Inference pipeline ready | device: {device}")

    def predict(self, image):
        tensor = self.transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence, predicted = probabilities.max(dim=1)
        return {
            "digit": predicted.item(),
            "confidence": confidence.item(),
            "all_probs": probabilities.squeeze().tolist()
        }

    def predict_batch(self, images):
        tensors = torch.stack([self.transform(img) for img in images]).to(device)
        with torch.no_grad():
            outputs = self.model(tensors)
            probs = torch.softmax(outputs, dim=1)
            confidences, predicted = probs.max(dim=1)
        return [
            {"digit": d.item(), "confidence": c.item()}
            for d, c in zip(predicted, confidences)
        ]

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Training Pipeline ===")
    trainer = MNISTTrainingPipeline(lr=0.001)
    trainer.train(epochs=3)
    trainer.save("models/mnist_cnn.pth")

    print("\n=== Inference Pipeline ===")
    pipeline = MNISTInferencePipeline("models/mnist_cnn.pth")
    test_data = datasets.MNIST(root="data", train=False, download=True)

    print("\nSingle image predictions:")
    for i in range(5):
        image, true_label = test_data[i]
        result = pipeline.predict(image)
        status = "✓" if result["digit"] == true_label else "✗"
        print(f"  {status} True: {true_label} | Predicted: {result['digit']} | Confidence: {result['confidence']:.2%}")

    print("\nBatch prediction (10 images at once):")
    images = [test_data[i][0] for i in range(10)]
    labels = [test_data[i][1] for i in range(10)]
    start = time.time()
    results = pipeline.predict_batch(images)
    elapsed = (time.time() - start) * 1000
    correct = sum(r["digit"] == l for r, l in zip(results, labels))
    print(f"  Accuracy: {correct}/10")
    print(f"  Batch time: {elapsed:.2f} ms")
    print(f"  Per image: {elapsed/10:.2f} ms")
```

```
=== Training Pipeline ===
Training pipeline ready | device: cuda
  Epoch 1/3 | Loss: 0.2564
  Epoch 2/3 | Loss: 0.0774
  Epoch 3/3 | Loss: 0.0521
Model saved to models/mnist_cnn.pth

=== Inference Pipeline ===
Inference pipeline ready | device: cuda

Single image predictions:
  ✓ True: 7 | Predicted: 7 | Confidence: 99.99%
  ✓ True: 2 | Predicted: 2 | Confidence: 100.00%
  ✓ True: 1 | Predicted: 1 | Confidence: 99.84%
  ✓ True: 0 | Predicted: 0 | Confidence: 99.91%
  ✓ True: 4 | Predicted: 4 | Confidence: 99.97%

Batch prediction (10 images at once):
  Accuracy: 10/10
  Batch time: 5.47 ms
  Per image: 0.55 ms
```

> **Key insight:** `if __name__ == "__main__"` means these classes can be
> imported by other scripts without running training automatically.
> That is the proper Python module pattern.

---



## Step 19 — Embeddings

> **NOTE:** Text can't be fed into a neural network as strings.
> Embeddings convert words into float vectors that capture meaning.
>
> ```
> word → integer (vocabulary lookup)
> integer → float vector (embedding layer)
>
> "cat" → 3 → [0.2, 0.8, 0.1, 0.4]
> "dog" → 4 → [0.3, 0.7, 0.2, 0.3]  ← similar to cat
> "car" → 5 → [0.9, 0.1, 0.8, 0.2]  ← very different
> ```

### src/llm/01_embeddings.py

```python
import torch
import torch.nn as nn

device = torch.device("cuda")

# Vocabulary — maps words to integers
vocab = {
    "": 0, "": 1, "the": 2, "cat": 3,
    "dog": 4, "sat": 5, "on": 6, "mat": 7, "ran": 8, "fast": 9,
}

vocab_size = len(vocab)   # 10 words
embed_dim  = 4            # each word becomes a 4-number vector

# Embedding layer — lookup table of shape [vocab_size, embed_dim]
embedding = nn.Embedding(vocab_size, embed_dim).to(device)

# Convert sentence to vectors
sentence = ["the", "cat", "sat", "on", "the", "mat"]
indices  = torch.tensor([vocab[w] for w in sentence]).to(device)
vectors  = embedding(indices)

# Similarity model — train embeddings to understand relationships
class SimilarityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc        = nn.Linear(embed_dim * 2, 1)
        self.sigmoid   = nn.Sigmoid()

    def forward(self, w1, w2):
        e1 = self.embedding(w1)
        e2 = self.embedding(w2)
        x  = torch.cat([e1, e2], dim=-1)
        return self.sigmoid(self.fc(x)).squeeze()

# Training pairs — (word1, word2, related?)
pairs = [
    ("cat", "dog",  1),   # related — both animals
    ("cat", "mat",  0),   # not related
    ("dog", "fast", 0),   # not related
    ("sat", "ran",  1),   # related — both verbs
    ("the", "on",   1),   # related — both function words
    ("cat", "ran",  0),   # not related
]

model     = SimilarityModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn   = nn.BCELoss()

w1s = torch.tensor([vocab[p[0]] for p in pairs]).to(device)
w2s = torch.tensor([vocab[p[1]] for p in pairs]).to(device)
ys  = torch.tensor([p[2] for p in pairs], dtype=torch.float).to(device)

for epoch in range(500):
    pred = model(w1s, w2s)
    loss = loss_fn(pred, ys)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} | Loss: {loss.item():.4f}")
```

```
Epoch 0   | Loss: 0.7554
Epoch 100 | Loss: 0.0431
Epoch 200 | Loss: 0.0094
Epoch 300 | Loss: 0.0041
Epoch 400 | Loss: 0.0023

Similarity predictions:
  cat  + dog  → similarity: 0.9971  ← animals
  sat  + ran  → similarity: 0.9973  ← verbs
  cat  + mat  → similarity: 0.0004  ← unrelated
  dog  + fast → similarity: 0.0001  ← unrelated
```

> **Key insight:** The network was never told cat and dog are animals.
> It figured out relationships purely by adjusting 4 float numbers per word
> until similar words had similar vectors. This is exactly how GPT and LLaMA
> represent language internally — just at a much larger scale.
>
> ```
> Your model: 10 words,      4 dimensions
> GPT-4:      100,000 words, 12,288 dimensions

## Step 20 — Attention Mechanism

> **NOTE:** Embeddings give every word a fixed vector. Attention lets each
> word look at all other words and decide which ones are relevant.
>
> ```
> "bank" near "river" → attends to "river" → means riverbank
> "bank" near "loan"  → attends to "loan"  → means financial bank
>
> Attention(Q, K, V) = softmax(QK^T / √d) × V
>
> Q (Query) → what am I looking for?
> K (Key)   → what do I contain?
> V (Value) → what do I pass forward?
> ```

### src/llm/02_attention.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V  = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        # Step 1: similarity scores between every pair of words
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Step 2: convert to probabilities
        weights = F.softmax(scores, dim=-1)

        # Step 3: weighted mix of value vectors
        output = torch.matmul(weights, V)
        return output, weights

attention = SelfAttention(embed_dim=8).to(device)

# [batch, seq_len, embed_dim]
x_batch = torch.randn(1, 4, 8).to(device)
output, weights = attention(x_batch)
```

```
Input shape:  torch.Size([1, 4, 8])
Output shape: torch.Size([1, 4, 8])
Weights shape: torch.Size([1, 4, 4])

Attention weights per word:
  word0 attends: ['0.30', '0.47', '0.05', '0.18']
  word1 attends: ['0.23', '0.26', '0.31', '0.20']
  word2 attends: ['0.18', '0.18', '0.38', '0.26']
  word3 attends: ['0.41', '0.25', '0.17', '0.18']
```

> **Key insight:** Input and output are the same shape `[1, 4, 8]`.
> Attention does not change the shape — it enriches each word's vector
> with context from the whole sentence. This single formula is the core
> of every transformer — GPT, BERT, LLaMA all use exactly this.


#[cloud-user@bastion gpu-learning]$ uv run src/llm/03_transformer_block.py 
Input shape:  torch.Size([1, 4, 8])
Output shape: torch.Size([1, 4, 8])

Transformer block parameters:
  attention.W_Q.weight           | torch.Size([8, 8])
  attention.W_K.weight           | torch.Size([8, 8])
  attention.W_V.weight           | torch.Size([8, 8])
  ff.0.weight                    | torch.Size([32, 8])
  ff.0.bias                      | torch.Size([32])
  ff.2.weight                    | torch.Size([8, 32])
  ff.2.bias                      | torch.Size([8])
  norm1.weight                   | torch.Size([8])
  norm1.bias                     | torch.Size([8])
  norm2.weight                   | torch.Size([8])
  norm2.bias                     | torch.Size([8])
   1 blocks →    776 parameters
   2 blocks →   1552 parameters
   4 blocks →   3104 parameters
   8 blocks →   6208 parameters
  12 blocks →   9312 parameters

Real world:
  GPT-2 small  → 12 blocks, embed=768  → 117M params
  GPT-3        → 96 blocks, embed=12288 → 175B params
  LLaMA 7B     → 32 blocks, embed=4096  →   7B params
[cloud-user@bastion gpu-learning]$ 


# Step 21 — Transformer Block

> **NOTE:** A transformer block combines attention with feedforward layers
> and residual connections. Stacking these blocks is literally what GPT is.
>
> ```
> Input
>   ↓
> Self Attention      ← context-aware vectors
>   ↓
> Add & Norm          ← residual + layer normalization
>   ↓
> Feed Forward        ← two linear layers
>   ↓
> Add & Norm          ← residual + layer normalization
>   ↓
> Output (same shape as input)
> ```

### src/llm/03_transformer_block.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores  = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))  # attention + residual + norm
        x = self.norm2(x + self.ff(x))         # feedforward + residual + norm
        return x

class TransformerStack(nn.Module):
    def __init__(self, embed_dim, ff_dim, num_blocks):
        super().__init__()
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, ff_dim)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
```

```
Input shape:  torch.Size([1, 4, 8])
Output shape: torch.Size([1, 4, 8])

   1 blocks →    776 parameters
   2 blocks →  1,552 parameters
   4 blocks →  3,104 parameters
   8 blocks →  6,208 parameters
  12 blocks →  9,312 parameters

Real world:
  GPT-2 small  → 12 blocks, embed=768  → 117M params
  GPT-3        → 96 blocks, embed=12288 → 175B params
  LLaMA 7B     → 32 blocks, embed=4096  →   7B params
```

> **Key insight:** Same architecture — just wider and deeper.
>
> ```
> Your model:  embed=8,    12 blocks →       9,312 params
> GPT-2 small: embed=768,  12 blocks → 117,000,000 params
> LLaMA 7B:    embed=4096, 32 blocks →   7,000,000,000 params
> ```
>
> Residual connections (`x + attention(x)`) prevent the network from
> forgetting the original input as it passes through many layers.

---

## Step 22 — Mini GPT

> **NOTE:** A GPT generates text one token at a time by predicting the
> next token from all previous tokens. The causal mask prevents it from
> cheating by looking at future tokens.
>
> ```
> Text → tokenize → embed → positional encode
>      → transformer blocks → linear head
>      → softmax → sample next token → repeat
> ```

### src/llm/04_mini_gpt.py

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda")
torch.manual_seed(42)

# Training text
text = """the cat sat on the mat
the dog ran fast
the cat ran on the mat
the dog sat fast
the cat and dog sat on the mat
the dog and cat ran fast"""

# Character-level tokenizer
chars     = sorted(set(text))
vocab_size = len(chars)
ch2idx    = {ch: i for i, ch in enumerate(chars)}
idx2ch    = {i: ch for i, ch in enumerate(chars)}
encode    = lambda s: [ch2idx[c] for c in s]
decode    = lambda l: ''.join([idx2ch[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long).to(device)

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.W_Q   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V   = nn.Linear(embed_dim, embed_dim, bias=False)
        self.scale = embed_dim ** 0.5

    def forward(self, x):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        T      = x.size(1)
        mask   = torch.tril(torch.ones(T, T, device=x.device))
        scores = scores.masked_fill(mask == 0, float('-inf'))  # causal mask
        return torch.matmul(F.softmax(scores, dim=-1), V)

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.attention = SelfAttention(embed_dim)
        self.ff    = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.norm1(x + self.attention(x))
        x = self.norm2(x + self.ff(x))
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, num_blocks, max_seq_len):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding   = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, ff_dim) for _ in range(num_blocks)
        ])
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, idx):
        B, T     = idx.shape
        tok_emb  = self.token_embedding(idx)
        pos_emb  = self.pos_embedding(torch.arange(T, device=idx.device))
        x        = tok_emb + pos_emb
        for block in self.blocks:
            x = block(x)
        return self.head(x)

model = MiniGPT(vocab_size, embed_dim=32, ff_dim=128,
                num_blocks=2, max_seq_len=32).to(device)

# Training
block_size = 16
batch_size = 8

def get_batch():
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x  = torch.stack([data[i:i+block_size] for i in ix])
    y  = torch.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    x, y   = get_batch()
    logits = model(x)
    loss   = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Generation
def generate(prompt, max_new_tokens=50):
    model.eval()
    idx = torch.tensor(encode(prompt), dtype=torch.long).unsqueeze(0).to(device)
    for _ in range(max_new_tokens):
        logits   = model(idx[:, -block_size:])
        probs    = F.softmax(logits[:, -1, :], dim=-1)
        next_tok = torch.multinomial(probs, 1)
        idx      = torch.cat([idx, next_tok], dim=1)
    return decode(idx[0].tolist())

print(generate("the cat", max_new_tokens=60))
print(generate("the dog", max_new_tokens=60))
```

```
MiniGPT parameters: 25,103
  Epoch    0 | Loss: 2.9415
  Epoch  200 | Loss: 0.2790
  Epoch  400 | Loss: 0.2060
  Epoch  600 | Loss: 0.1580
  Epoch  800 | Loss: 0.1920

Generated text:
the cat and dog sat ran fast
the cat ran on the mat
the dog and cat
the dog sat on the mat
the dog and cat ran on the mat
```

> **Key insight:** This is exactly what GPT-4 does — just with billions
> more parameters and trillions of characters of training data.
>
> ```
> MiniGPT:  25K params,  135 chars  → generates cat/dog sentences
> GPT-4:    1.8T params, ~13T tokens → generates anything
> ```
>
> The causal mask (`torch.tril`) is what makes it generative —
> each token can only see past tokens, never future ones.

# Step 23 — Loading a Pretrained LLM (GPT-2)

> **NOTE:** HuggingFace Transformers lets you load pretrained models
> in two lines. Same architecture as your MiniGPT — just much bigger.
>
> ```
> Your MiniGPT → vocab=15,    embed=32,  blocks=2  →    25K params
> GPT-2        → vocab=50257, embed=768, blocks=12 → 124M params
> ```

### Install

```bash
uv pip install transformers accelerate
```

### src/llm/06_pretrained_llm.py

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

device = torch.device("cuda")

# Load pretrained GPT-2
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token   # GPT-2 has no pad token
model     = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")
print(f"Device:     {next(model.parameters()).device}")

def generate(prompt, max_new_tokens=50, temperature=0.8):
    inputs = tokenizer(
        prompt, return_tensors="pt", padding=True
    ).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(generate("The cat sat on the mat and"))
print(generate("Once upon a time in a land far away"))
print(generate("The best way to learn machine learning is"))
```

```
Parameters: 124,439,808
Device:     cuda:0

The cat sat on the mat and stared at it for a long time, then
his head snapped back up to look at his own.

Once upon a time in a land far away, the only one who knew how
to properly communicate was. He was no stranger to the black world...

The best way to learn machine learning is to learn how to do it
in a real way. I'm not going to say that you should learn machine
learning because you don't need it...
```

> **Key insight:**
>
> ```
> temperature=0.1 → very focused, repetitive, safe
> temperature=0.8 → balanced
> temperature=1.5 → creative, unpredictable
> ```
>
> Training data and scale is everything — same architecture,
> vastly different capability.

## Step 24 — Fine-tuning with LoRA

> **NOTE:** LoRA freezes the entire pretrained model and adds tiny
> trainable adapter matrices. Only 0.24% of parameters are updated.
>
> ```
> Full fine-tuning → update all 124M params → huge memory
> LoRA             → freeze all, add adapters → 294K params
>
> Original weight W (frozen)
> + Small matrices A × B (trainable, rank=8)
> = Effective weight W + AB
> ```

### Install

```bash
uv pip install peft
```

### src/llm/07_finetune_lora.py

```python
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from peft import LoraConfig, get_peft_model

device = torch.device("cuda")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

# Apply LoRA
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["c_attn"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

# Training data
training_data = [
    "Question: What is a neural network? Answer: A neural network is a machine learning model inspired by the brain, made of layers of connected neurons that learn patterns from data.",
    "Question: What is gradient descent? Answer: Gradient descent is an optimization algorithm that minimizes loss by updating weights in the direction that reduces the error.",
    "Question: What is overfitting? Answer: Overfitting is when a model memorizes training data and fails to generalize to new unseen data.",
    "Question: What is a GPU? Answer: A GPU is a graphics processing unit that accelerates machine learning by performing thousands of parallel computations simultaneously.",
    "Question: What is backpropagation? Answer: Backpropagation is the algorithm that computes gradients by propagating the error backwards through the network layers.",
]

def tokenize(text):
    tokens = tokenizer(
        text, return_tensors="pt",
        truncation=True, max_length=128, padding="max_length"
    )
    return tokens["input_ids"].to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Train
model.train()
for epoch in range(200):
    total_loss = 0
    for text in training_data:
        input_ids = tokenize(text)
        outputs   = model(input_ids, labels=input_ids)
        loss      = outputs.loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if epoch % 40 == 0:
        print(f"  Epoch {epoch:3d} | Loss: {total_loss/len(training_data):.4f}")

# Generate
def answer(question, max_new_tokens=60):
    model.eval()
    prompt = f"Question: {question} Answer:"
    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print(answer("What is a neural network?"))
print(answer("What is overfitting?"))
print(answer("What is a GPU?"))
```

```
Total parameters:     124,734,720
Trainable parameters:     294,912  → 0.24%

  Epoch   0 | Loss: 11.0423
  Epoch  40 | Loss: 0.7652
  Epoch  80 | Loss: 0.1327
  Epoch 120 | Loss: 0.0840
  Epoch 160 | Loss: 0.0511

Question: What is a neural network? Answer: A brain model of the human
body is inspired by an ancient Greek statue, made from layers of connected
neurons that learn patterns through experience.

Question: What is overfitting? Answer: Overfit refers to when a model
memorizes training data and fails in the general direction.

Question: What is a GPU? Answer: A GPUs are graphics processing units
that accelerate machine learning by performing thousands of parallel
computations simultaneously.
```

> **Key insight:** This is exactly how companies fine-tune LLaMA and
> Mistral on custom data — same LoRA pattern, just bigger models.
>
> ```
> r=8   → 294K trainable params  → fast, less accurate
> r=64  → 2.4M trainable params  → slower, more accurate
> ```


## Step 25 — Serving with Ollama

> **NOTE:** Ollama runs any LLM as a REST API server — same interface
> as OpenAI but fully local on your GPU.
>
> ```
> Without Ollama → python script.py → output → done
> With Ollama    → always running → any client can call it
>
> GET  /api/tags      → list models
> POST /api/generate  → single prompt
> POST /api/chat      → conversation with history
> ```

### Setup

```bash
# Run Ollama in Podman with GPU access
mkdir -p ~/.ollama
podman run -d \
  --name ollama \
  --device nvidia.com/gpu=all \
  -p 11434:11434 \
  -v ~/.ollama:/root/.ollama:Z \
  --security-opt label=disable \
  docker.io/ollama/ollama

# Pull a model
podman exec ollama ollama pull tinyllama
```

### src/llm/08_serve_ollama.py

```python
import requests

OLLAMA_URL = "http://localhost:11434"

def list_models():
    response = requests.get(f"{OLLAMA_URL}/api/tags")
    models = response.json()["models"]
    for m in models:
        print(f"  {m['name']:30s} | {m['size']/1e9:.2f} GB")

def generate(prompt, model="tinyllama", stream=False):
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": stream}
    )
    return response.json()["response"]

def chat(messages, model="tinyllama"):
    response = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={"model": model, "messages": messages, "stream": False}
    )
    return response.json()["message"]["content"]

if __name__ == "__main__":
    # 1. List models
    list_models()

    # 2. Generate
    print(generate("What is backpropagation? One sentence."))

    # 3. Chat
    messages = [{"role": "user", "content": "What is overfitting?"}]
    print(chat(messages))

    # 4. Multi-turn chat
    history = []
    for q in ["What is a neural network?", "How does it learn?", "What can go wrong?"]:
        history.append({"role": "user", "content": q})
        answer = chat(history)
        history.append({"role": "assistant", "content": answer})
        print(f"Q: {q}")
        print(f"A: {answer[:200]}")
        print()
```

```
=== Models ===
  tinyllama:latest               | 0.64 GB

=== Generate ===
Backpropagation is a process in which errors or adjustments are made
to the weights of an artificial neural network...

=== Multi-turn Chat ===
Q: What is a neural network?
A: A neural network is an algorithmic model inspired by biological
   nervous systems...

Q: How does it learn?
A: Neural networks learn by processing data through multiple layers...

Q: What can go wrong?
A: There are several potential issues including overfitting, vanishing
   gradients, and poor generalization...
```

> **Key insight:** Multi-turn chat passes the full conversation history
> with every request. Ollama has no memory — you maintain the state.
>
> ```
> tinyllama  → 0.64 GB  → fast, fits anywhere
> llama3     → 4.7 GB   → smarter, needs more VRAM
> llama3:70b → 40 GB    → very smart, needs your L4 + more
> ```