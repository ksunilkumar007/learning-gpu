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
