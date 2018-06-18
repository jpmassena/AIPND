# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch
import json

from utils import get_arch_model, ClassifierModel, process_image
from PIL import Image


def main():
    """Program main function"""

    args = read_args()  # Read command line arguments

    model = load_model(args.checkpoint)  # load the previously saved model

    # Checks if user wants to use GPU and if the system is capable to use it
    device = torch.device(
        "cuda:0" if (torch.cuda.is_available() and args.gpu) else "cpu")
    print("{} will be used to make the prediction".format(device))
    model.to(device)

    # Predict the top likely classes for the input image
    probabilities, classes = predict(args.input, model, device, args.top_k)

    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)

    if args.chart:
        show_graphic_results(args.input, probabilities, classes, cat_to_name)
    else:
        show_textual_results(probabilities, classes, cat_to_name)


def read_args():
    """Reads command line arguments

    Returns:
        data structure with the values passed as arguments
    """
    parser = argparse.ArgumentParser()  # creates the arguments parser

    # Path to the input image
    parser.add_argument('input', type=str, help='Path to the input image')
    # Path to the saved model file
    parser.add_argument('checkpoint', type=str, help='Path to the saved model')

    parser.add_argument('-t', '--top_k', type=int, default=1,
                        help='Top "K" most likely classes')

    parser.add_argument('-cat', '--category_names', type=str,
                        default='cat_to_name.json',
                        help='Mapping of categories to real names')
    parser.add_argument('--gpu', action="store_true",
                        help='Use GPU for inference if available')
    parser.add_argument('--chart', action="store_true",
                        help='Show results graphically instead of text')

    parsed_args = parser.parse_args()

    return parsed_args  # returns the args structure to the caller


def load_model(checkpoint):
    """Load a previously saved model state

    Returns:
        The recreated model with all the information that was saved
    """

    state = torch.load(checkpoint)

    model, _ = get_arch_model('vgg')  # Load pretrained model

    # Replace the classifier with a new one built with saved parameters
    model.classifier = ClassifierModel(state['input_size'],
                                       state['output_size'],
                                       state['hidden_layers'],
                                       0.5)

    model.load_state_dict(state['state'])
    model.class_to_idx = state['class_to_idx']
    model.eval()  # turn off dropout to make predictions

    print("Model was successfully loaded!")

    return model


def predict(image_path, model, device, topk=1):
    """Predict the topk classes of an image using a trained deep learning model

    Arguments:
        image_path -- Path to the image to do inference on
        model -- Model to use for the prediction
        device -- Device to make the computations
        topk -- Number of most likely classes to show (default: {5})

    Returns:
        The probabilities for each of the top k classes and the mapped name of
        the classes
    """

    image = Image.open(image_path)  # Load image from filesystem
    np_image = process_image(image)  # Preprocess it into a numpy array
    tensor_image = torch.from_numpy(np_image)  # Turn it into a tensor

    # Model expects 4 dimensions (first is batch size)
    tensor_image = tensor_image.unsqueeze(0)

    # Feed forward through the model
    output = model.forward(tensor_image.float().to(device))
    # Get the exp of the log_softmax probabilities
    probabilities = torch.exp(output).cpu()

    # Get the top K probabilities
    probabilities = probabilities.data.topk(topk)

    # Revert the mapping from index to classes
    idx_to_class = {model.class_to_idx[idx]: idx for idx in model.class_to_idx}

    # Get the top classes that were mapped for this prediction
    mapped_classes = [
        idx_to_class[label] for label in probabilities[1].numpy()[0]]

    return probabilities[0].numpy()[0], mapped_classes


def show_textual_results(probabilities, classes, cat_to_name):
    """Show the results textually

    Arguments:
        probabilities -- Probabilities of the top predicted classes
        classes -- Top predicted classes
        cat_to_name -- Mapping for the classes names
    """

    labels = [cat_to_name[idx] for idx in classes]
    if len(labels) == 1:  # Only one class prediction
        print("The most likely class for the input image is",
              "{}".format(labels[0]),
              "with {:.1%} probability".format(probabilities[0]))
    else:
        print("\nThe most likely classes for the input image are:\n")
        for i in range(len(labels)):
            print("{}".format(labels[i]),
                  "with {:.1%} probability".format(probabilities[i]))


def show_graphic_results(image_path, probabilities, classes, cat_to_name):
    """Show the results visually with a chart

    Arguments:
        image_path -- Path to the test image
        probabilities -- Probabilities of the top predicted classes
        classes -- Top predicted classes
        cat_to_name -- Mapping for the classes names
    """

    _, (ax1, ax2) = plt.subplots(figsize=(6, 9), ncols=1, nrows=2)

    # Display image
    ax1.axis('off')
    ax1.set_title(cat_to_name[classes[0]])
    ax1.imshow(Image.open(image_path))

    # Display Probabilities chart
    # Loads the mapped labels names
    labels = [cat_to_name[idx] for idx in classes]

    # Number of y ticks is the number of top classes
    y_ticks = np.arange(len(labels))
    ax2.set_yticks(y_ticks)
    ax2.set_yticklabels(labels)
    ax2.invert_yaxis()  # Invert the order
    ax2.barh(y_ticks, probabilities)  # horizontal bar chart

    plt.tight_layout()
    plt.show()


# Call to main function to run the program
if __name__ == "__main__":
    main()
