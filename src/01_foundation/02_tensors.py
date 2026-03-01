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
