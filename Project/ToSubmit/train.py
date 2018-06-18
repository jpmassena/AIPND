# -*- coding: utf-8 -*-

import argparse
import torch

from utils import get_data_loaders, get_arch_model, get_hidden_layers
from workspace_utils import active_session
from utils import ClassifierModel
from torch import optim
from torch import nn


def main():
    """Program main function"""

    args = read_args()  # Read command line arguments

    # Get the data loaders
    train_loader, valid_loader, class_to_idx = get_data_loaders(args.data_dir)

    # Get the pretrained model and the model features output size
    model, input_size = get_arch_model(args.arch)

    hidden_layers = get_hidden_layers(args.hidden_layers)

    # Replace the model classifier with our network
    model.classifier = ClassifierModel(input_size, 102, hidden_layers, 0.5)

    # Checks if user wants to use GPU and if the system is capable to use it
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    print("{} will be used to train the network".format(device))

    # Train the model
    with active_session():
        model = train_model(model, train_loader, valid_loader, args.learning_rate,
                            device, args.epochs)

    # Save the model to the filesystem
    save_model(model, input_size, 102, hidden_layers, class_to_idx,
               args.save_dir)


def read_args():
    """Reads command line arguments

    Returns:
        data structure with the values passed as arguments
    """

    parser = argparse.ArgumentParser()  # creates the arguments parser

    # Folder where the data is located
    parser.add_argument('data_dir', type=str, help='Path to the data folder')

    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save checkpoints')

    archs = ['vgg', 'densenet']

    parser.add_argument('--arch', type=str, default='vgg',
                        action='store', choices=archs,
                        help='Trained model architecture to use')

    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-4,
                        help='Learning rate hyperparameter')

    parser.add_argument('-hl', '--hidden_layers', nargs='+', type=int,
                        default=4096,
                        help='Hidden layers sizes hyperparameter')

    parser.add_argument('-ep', '--epochs', type=int, default=5,
                        help='Epochs hyperparameter')

    parser.add_argument('--gpu', action="store_true",
                        help='Use GPU for training if available')

    parsed_args = parser.parse_args()

    return parsed_args  # returns the args structure to the caller


def validate_model(model, valid_loader, criterion, device):
    """Validate the accuracy of the model during training

    Arguments:
        model -- The model to validate
        valid_loader -- Valdation data loader
        criterion -- Loss function used by the model
        device -- If it is to train on the cpu or gpu

    Returns:
        The actual loss and accuracy of the model W.R.T. the validation data
    """
    valid_loss = 0  # Running loss of the validation batch
    accuracy = 0  # Accuracy of the classifier

    for inputs, labels in valid_loader:  # for each batch of data / label

        inputs, labels = inputs.to(device), labels.to(device)
        output = model.forward(inputs)  # feed forward
        loss = criterion(output, labels)  # calculate the loss

        # Exponential because the output is the log of prob. distribution
        probabilities = torch.exp(output)
        # Compare the class label with the most probable classification class
        # from our classifier. dim=1 because dim=0 is the batch size
        # [1] because max returns the max probability value on [0] and the
        # index (our classes) on [1]
        equality = (labels.data == probabilities.max(dim=1)[1])
        # the average of right classifications
        accuracy += equality.type(torch.FloatTensor).mean()

        valid_loss += loss.item()

    return valid_loss, accuracy


def train_model(model, train_loader, valid_loader, learning_rate, device,
                epochs):
    """Trains a model with train_loader and validates it with valid_loader

    Arguments:
        model -- Model to train
        train_loader -- Data to train
        valid_loader -- Data to validate the training
        learning_rate -- Learning rate
        device -- Device where the computations will be executed
        epochs -- Number of epochs to train

    Returns:
        The trained model
    """

    # Our loss function will be 'negative log likelihood'
    criterion = nn.NLLLoss()
    # We only want to optimize our classifier parameters
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    # makes PyTorch use 'device' to compute
    model.to(device)
    criterion.to(device)

    print_every = 25
    step = 0
    for epoch in range(epochs):  # for each epoch
        running_loss = 0
        for inputs, labels in train_loader:  # for each batch of data / label
            step += 1
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()  # resets gradients to zero

            output = model.forward(inputs)  # feed forward
            loss = criterion(output, labels)  # calculate the loss
            loss.backward()  # back propagate the loss
            optimizer.step()  # do gradient descent

            running_loss += loss.item()

            if step % print_every == 0:

                model.eval()  # Turn off dropout to make the validation pass

                # Turn off gradients for the validation pass
                with torch.no_grad():
                    valid_loss, accuracy = validate_model(model, valid_loader,
                                                          criterion, device)

                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(
                          running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(
                          valid_loss/len(valid_loader)),
                      "Validation Accuracy: {:.3f}".format(
                          accuracy/len(valid_loader)))

                running_loss = 0

                model.train()  # enable dropout back

        model.eval()  # Turn off dropout to make the validation pass

        with torch.no_grad():  # Turn off gradients for the validation pass
            valid_loss, accuracy = validate_model(
                model, valid_loader, criterion, device)

        print("\nEpoch: {}/{}.. ".format(epoch+1, epochs),
              "Validation Loss: {:.3f}.. ".format(
                  valid_loss/len(valid_loader)),
              "Validation Accuracy: {:.3f}\n".format(
                  accuracy/len(valid_loader)))

        model.train()  # enable dropout back

    return model


def save_model(model, input_size, output_size, hidden_layers, class_to_idx,
               save_dir):
    """Saves the model state after training is complete

    Arguments:
        model -- Model to save
        input_size -- Expected size of the input vector
        output_size -- Size of the output
        hidden_layers -- Sizes of the hidden layers
        class_to_idx -- Mapping between classes and indexes
        save_dir -- Folder to save the file
    """

    model.class_to_idx = class_to_idx

    trained_checkpoint = {
        'input_size': input_size,
        'output_size': output_size,
        'hidden_layers': hidden_layers,
        'class_to_idx': model.class_to_idx,
        'state': model.state_dict()
    }

    torch.save(trained_checkpoint, save_dir+'/checkpoint.pth')
    print("Model was successfully saved!")


# Call to main function to run the program
if __name__ == "__main__":
    main()
