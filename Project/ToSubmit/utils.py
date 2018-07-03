# -*- coding: utf-8 -*-

import torch.nn.functional as F
import numpy as np
import torch

from torchvision import models, datasets, transforms
from torch import nn


class ClassifierModel(nn.Module):  # Define the classifier network

    def __init__(self, input_size, output_size, hidden_layers, drop_p=0.5):
        """Initialize the network
        Arguments:
            input_size -- number of neurons of the input layer
            output_size -- number of outputs
            hidden_layers -- number of neurons of each hidden layer
            drop_p -- probability for a neuron to be deactivated
        """
        super(ClassifierModel, self).__init__()

        # Create input-output sizes from the hidden_layers values
        # trick from "Inference & Validation" video
        hidden_sizes = zip(hidden_layers[:-1], hidden_layers[1:])

        # Input layer takes 'input_size' as input and outputs
        # 'first hidden layer's size'
        self.layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])

        # Hidden layers. h_1 represents the input to the layer and h_2 its
        # output
        self.layers.extend([nn.Linear(h_1, h_2) for h_1, h_2 in hidden_sizes])

        # Last layer takes as input the 'last hidden layer's' output and
        # outputs 'output_size'
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Implement dropout
        self.dropout = nn.Dropout(p=drop_p)

    def forward(self, input):
        """Feeds the input forward through the network

        Arguments:
            input -- The network input batch data

        Returns:
            The log of the probability distribution of the categories for each
            input
        """
        output = input
        # feed through the hidden layers
        for linear_layer in self.layers:
            output = linear_layer(output)  # Linear calculation
            output = F.relu(output)  # Non-linearity of the layer
            output = self.dropout(output)  # Dropout

        output = self.output(output)  # last layer
        # returns the probabilities distributions of the dimension 1 (dim=0 is
        # the batch size)
        return F.log_softmax(output, dim=1)


def get_data_loaders(data_dir):
    """Loads the training and validation data into PyTorch DataLoaders.
    Transforms are applied to each dataset as a technique of Data Augmentation

    Arguments:
        data_dir -- The root folder of the data

    Returns:
        The DataLoaders for training and validation data and the class_to_idx
        dictionary
    """

    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    batch_size = 32

    train_transforms = transforms.Compose([  # Transforms for the training data
        transforms.RandomRotation(30),  # Rotates between -30 and 30 degrees
        transforms.RandomResizedCrop(224),  # Resize and random crop
        transforms.RandomHorizontalFlip(),  # Random horizontally flip
        transforms.ToTensor(),  # Turn image data into a tensor of pixels
        # Normalize the color channels
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    valid_transforms = transforms.Compose([  # Transforms for validation data
        # Resize the image so that the smallest edge is of the desired size
        transforms.Resize(256),
        transforms.CenterCrop(224),  # Crop a centered square of 224 pixels
        transforms.ToTensor(),  # Turn image data into a tensor of pixels
        # Normalize the color channels
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # Data to use in the training process
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    # Data to use tto validate the training
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Train data loader. The order of every batch is shuffled
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    # Validation data loader. The order of every batch is shuffled
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)
    return train_loader, valid_loader, train_dataset.class_to_idx


def get_arch_model(arch):
    """Loads a pretrained model, according with the passed value of arch

    Arguments:
        arch -- The model architecture to load

    Returns:
        The model and the size of the output vector of the feature part
    """

    # Loads the pretrained model according with arch
    feature_size = 0
    if arch == 'densenet':
        model = models.densenet161(pretrained=True)
        model_used = "Densenet-161"
        feature_size = model.classifier.in_features
    else:
        model = models.vgg16(pretrained=True)
        model_used = "VGG-16"
        feature_size = model.classifier[0].in_features

    print("Pretrained model that will be used is {}".format(model_used))

    # Freezes model parameters
    for param in model.parameters():
        param.requires_grad = False

    return model, feature_size


def get_hidden_layers(hidden_layers):
    """Get a list of sizes for hidden layers

    Arguments:
        hidden_layers -- Can be an int if we only want one layer or a list if
        we want more layers

    Returns:
        A list constructed from one passed int or the lsit that was passed
    """

    if type(hidden_layers) is list:
        return hidden_layers
    else:
        return [hidden_layers, hidden_layers]


def process_image(image):
    """Scales, crops, and normalizes a PIL image for a PyTorch model

    Arguments:
        image -- A Pillow image

    Returns:
        Numpy array of the image pixels
    """

    target_size = 256
    crop_size = 224

    width, height = image.size

    # Check which side is the shortest to resize it to target_size and scale
    # the other one
    if(width > height):
        width = int(width * target_size / height)
        height = int(target_size)
    else:
        height = int(height * target_size / width)
        width = int(target_size)

    image = image.resize((width, height))

    # Crop the center of the image
    left = (width - crop_size) / 2
    top = (height - crop_size) / 2
    right = left + crop_size
    bottom = top + crop_size

    image = image.crop((left, top, right, bottom))

    # Get a numpy array from the image
    np_image = np.array(image) / 255

    # Normalizes the image array
    image_mean = np.array([0.485, 0.456, 0.406])
    image_std = np.array([0.229, 0.224, 0.225])
    normalized_image = (np_image - image_mean) / image_std

    # Change the channels order
    normalized_image = normalized_image.transpose((2, 0, 1))

    return normalized_image
