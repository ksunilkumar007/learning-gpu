import torch
import torch.nn as nn

#Create a simple input
simple_input = torch.tensor([-2.0,-1.0,0.0,1.0,2.0,3.0])


# ReLU — most common activation function
# Rule: if x < 0 -> 0, if x >= 0 -> keep x

relu = nn.ReLU()
print("Input: ", simple_input)
print("ReLU : ",relu(simple_input))

# Sigmoid — squashes any number between 0 and 1
# Used for binary classification (yes/no, true/false)
sigmoid = nn.Sigmoid()
print("\nSigmoid:", sigmoid(simple_input))

# Tanh — squashes any number between -1 and 1
# Similar to sigmoid but centered at 0
tanh = nn.Tanh()
print("Tanh:   ", tanh(simple_input))


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
