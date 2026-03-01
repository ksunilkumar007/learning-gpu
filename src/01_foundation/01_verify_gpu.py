import torch

print("Pytorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Device Count:", torch.cuda.device_count())
print("GPU Name:", torch.cuda.get_device_name())
