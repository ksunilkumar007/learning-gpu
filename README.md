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
