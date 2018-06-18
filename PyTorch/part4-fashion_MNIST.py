import matplotlib.pyplot as plt
import numpy as np
import torch
import helper
from torchvision import datasets, transforms

from torch import nn
from torch import optim
import torch.nn.functional as F

# normalizes the pixels into tensors. this will make them follow a normal
# distribution from -1 to 1
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                ])

# load training set
trainset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=True,
                                 transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64,
                                          shuffle=True)

# load test set
testset = datasets.FashionMNIST('F_MNIST_data/', download=True, train=False,
                                transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=True)

# show one of the images to see if the dataset is correcly loaded
dataiter = iter(trainloader)
images, labels = dataiter.next()
# the squeeze removes dimensions of input of size 1.
# In this case the color channel
plt.imshow(images[1].numpy().squeeze())
# plt.show()
plt.clf()

# Hyperparameters for our network
INPUT_SIZE = 784
OUTPUT_SIZE = 10
HIDDEN_LAYERS = [512, 256, 128]
EPOCHS = 3


class Network(nn.Module):

    def __init__(self, input_size, output_size, hidden_layers):
        """Initialize the network
        Arguments:
            input_size {int} -- number of neurons of the input layer
            output_size {int} -- number of outputs
            hidden_layers {list(int)} -- number of neurons of each hidden layer
        """

        super().__init__()
        # create input-output sizes from the hidden_layers values
        hidden_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        # input layer
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        # hidden layers
        self.layers.extend(nn.Linear(h_1, h_2) for h_1, h_2 in hidden_sizes)
        # output
        self.output = nn.Linear(hidden_layers[-1], output_size)

    def forward(self, input):
        """Feeds the input forward through the network

        Arguments:
            input -- The network input data
        
        Returns:
            [type] -- [description]
        """

        output = input
        # feed through the hidden layers
        for linear_layer in self.layers:
            output = F.relu(linear_layer(output))
        # feed through the output layer
        output = self.output(output)

        # log-softmax is the log of the probability distributions.
        # we need the exp later to retrieve the classes (exp(log(x)) = x)
        return F.log_softmax(output, dim=1)


def validate(model, testloader, criterion):
    """Validate the model

    Arguments:
        model -- The network to validate
        testloader -- Test data
        criterion -- Loss function to use
    """
    test_loss = 0
    accuracy = 0
    for images, targets in testloader:
        images.resize_(images.size()[0], 784)

        output = model.forward(images)
        test_loss += criterion(output, targets).item()

        class_probabilities = torch.exp(output)
        equality = (targets.data == class_probabilities.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()

    return test_loss, accuracy


model = Network(INPUT_SIZE, OUTPUT_SIZE, HIDDEN_LAYERS)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

step = 0
for epoch in range(EPOCHS):
    running_loss = 0
    for images, targets in trainloader:
        step += 1

        # flatten image from 28*28 matrix to 784 vector
        images.resize_(images.size()[0], 784)

        optimizer.zero_grad()  # clean gradients

        output = model.forward(images)  # feed forward
        loss = criterion(output, targets)  # calculate loss
        loss.backward()  # back-propagation
        optimizer.step()  # gradient descent

        running_loss += loss.item()

        if step % 32 == 0:
            # Turn off gradients for validation, saves memory and computations
            with torch.no_grad():
                test_loss, accuracy = validate(model, testloader, criterion)

            print("Epoch: {}/{}.. ".format(epoch+1, EPOCHS),
                  "Training Loss: {:.3f}.. ".format(running_loss/32),
                  "Test Loss: {:.3f}.. ".format(test_loss/len(testloader)),
                  "Test Accuracy: {:.3f}".format(accuracy/len(testloader)))

            running_loss = 0

# test network
dataiter = iter(testloader)
images, _ = dataiter.next()
img = images[0]
# Convert 2D image to 1D vector
img = img.view(1, 784)

# Calculate the class probabilities (softmax) for img
with torch.no_grad():
    output = model.forward(img)

class_probabilities = torch.exp(output)

# Plot the image and probabilities
helper.view_classify(img.view(1, 28, 28), class_probabilities,
                     version='Fashion')
