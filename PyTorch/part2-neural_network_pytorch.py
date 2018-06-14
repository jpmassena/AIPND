import numpy as np
import matplotlib.pyplot as plt
import torch
import helper
from torchvision import datasets, transforms

from torch import nn
from torch import optim
import torch.nn.functional as F
from collections import OrderedDict


# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ])
# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True,
                          train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True)

# Download and load the test data
testset = datasets.MNIST('MNIST_data/', download=True,
                         train=False, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
# the squeeze removes dimensions of input of size 1.
# In this case the color channel
plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r')
# plt.show()
plt.clf()

# Hyperparameters for our network
input_size = 784
hidden_1_sizes = [400, 200]
hidden_2_sizes = [200, 100]
output_size = 10

model = nn.Sequential(OrderedDict([
                      ('fc1', nn.Linear(input_size, hidden_1_sizes[0])),
                      ('relu1', nn.ReLU()),
                      ('fc2', nn.Linear(hidden_1_sizes[0], hidden_1_sizes[1])),
                      ('relu2', nn.ReLU()),
                      ('fc3', nn.Linear(hidden_2_sizes[0], hidden_2_sizes[1])),
                      ('relu3', nn.ReLU()),
                      ('output', nn.Linear(hidden_2_sizes[1], output_size)),
                      ('softmax', nn.Softmax(dim=1))]))
print(model)

# Run this cell with your model to make sure it works ##
# Forward pass through the network and display output
images, labels = next(iter(trainloader))
images.resize_(images.shape[0], 1, 784)
ps = model.forward(images[0, :])
helper.view_classify(images[0].view(1, 28, 28), ps)
