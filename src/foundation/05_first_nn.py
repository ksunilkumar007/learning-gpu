import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3,8) #3 inouts , 8 neurons
        self.layer2 = nn.Linear(8,1) #8 inputs , 1 outputs

    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        return x

#Create the Network

model = SimpleNeuralNetwork()
print(model)

#Move model to GPU
device = torch.device("cuda")
model = model.to(device)

#Create a sample input (3 Numbers) and move to GPU

sample_input = torch.tensor([1.0,2.0,3.0]).to(device)

# pass it through network

output = model(sample_input)
print("Input:",sample_input)
print("Output:",output)
print("Device:",output.device)

#Count the total learnable parameters
total_params = sum(p.numel() for p in model.parameters())
print("\nTotal trainable paramaters:",total_params)
