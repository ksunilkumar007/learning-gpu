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
