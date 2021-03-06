#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# */AIPND/intropylab-classifying-images/check_images.py
#
# DONE: 0. Fill in your information in the programming header below
# PROGRAMMER: Joao Massena
# DATE CREATED: 05/06/2018
# REVISED DATE:             <=(Date Revised - if any)
# REVISED DATE: 05/14/2018 - added import statement that imports the print
#                           functions that can be used to check the lab
# PURPOSE: Check images & report results: read them in, predict their
#          content (classifier), compare prediction to actual value labels
#          and output results
#
# Use argparse Expected Call with <> indicating expected user input:
#      python check_images.py --dir <directory with images> --arch <model>
#             --dogfile <file that contains dognames>
#   Example call:
#    python check_images.py --dir pet_images/ --arch vgg --dogfile dognames.txt
##

# Imports python modules
import argparse
from time import time
from os import listdir

# Imports classifier function for using CNN to classify images
from classifier import classifier

# Imports print functions that check the lab
from print_functions_for_lab_checks import *

# Main program function defined below


def main():
    # DONE: 1. Define start_time to measure total program runtime by
    # collecting start time
    start_time = time()

    # DONE: 2. Define get_input_args() function to create & retrieve command
    # line arguments
    in_arg = get_input_args()

    # prints the arguments values
    check_command_line_arguments(in_arg)

    # DONE: 3. Define get_pet_labels() function to create pet image labels by
    # creating a dictionary with key=filename and value=file label to be used
    # to check the accuracy of the classifier function
    answers_dic = get_pet_labels(in_arg.dir)

    # prints first 10 key-value pairs and the size of the dictionary
    check_creating_pet_image_labels(answers_dic)

    # DONE: 4. Define classify_images() function to create the classifier
    # labels with the classifier function uisng in_arg.arch, comparing the
    # labels, and creating a dictionary of results (result_dic)
    result_dic = classify_images(in_arg.dir, answers_dic, in_arg.arch)

    # prints matches and no matches of the classification result
    check_classifying_images(result_dic)

    # DONE: 5. Define adjust_results4_isadog() function to adjust the results
    # dictionary(result_dic) to determine if classifier correctly classified
    # images as 'a dog' or 'not a dog'. This demonstrates if the model can
    # correctly classify dog images as dogs (regardless of breed)
    adjust_results4_isadog(result_dic, in_arg.dogfile)

    # prints if the labels are of dogs or not
    check_classifying_labels_as_dogs(result_dic)

    # DONE: 6. Define calculates_results_stats() function to calculate
    # results of run and puts statistics in a results statistics
    # dictionary (results_stats_dic)
    results_stats_dic = calculates_results_stats(result_dic)

    # print the statistics
    check_calculating_results(result_dic, results_stats_dic)

    # DONE: 7. Define print_results() function to print summary results,
    # incorrect classifications of dogs and breeds if requested.
    print_results(result_dic, results_stats_dic, in_arg.arch, True, True)

    # DONE: 1. Define end_time to measure total program runtime
    # by collecting end time
    end_time = time()

    # DONE: 1. Define tot_time to computes overall runtime in
    # seconds & prints it in hh:mm:ss format
    tot_time = end_time - start_time
    print("\n** Total Elapsed Runtime:", print_pretty_time(tot_time))


# TODO: 2.-to-7. Define all the function below. Notice that the input
# paramaters and return values have been left in the function's docstrings.
# This is to provide guidance for acheiving a solution similar to the
# instructor provided solution. Feel free to ignore this guidance as long as
# you are able to acheive the desired outcomes with this lab.

def get_input_args():
    """
    Retrieves and parses the command line arguments created and defined using
    the argparse module. This function returns these arguments as an
    ArgumentParser object.
     3 command line arguments are created:
       dir - Path to the pet image files(default- 'pet_images/')
       arch - CNN model architecture to use for image classification(default-
              pick any of the following vgg, alexnet, resnet)
       dogfile - Text file that contains all labels associated to dogs(default-
                'dognames.txt'
    Parameters:
     None - simply using argparse module to create & store command line
     arguments
    Returns:
     parse_args() -data structure that stores the command line arguments object
    """

    # creates the arguments parser
    parser = argparse.ArgumentParser()

    # creates the 3 command line arguments. one for the images folder, one for
    # the CNN architecture and one for the file with dog names
    parser.add_argument('--dir', type=str, default='pet_images/',
                        help='Path to the folder pet_images')
    parser.add_argument('--arch', type=str, default='alexnet',
                        help='CNN model architecture to use')
    parser.add_argument('--dogfile', type=str, default='dognames.txt',
                        help='File with valid dognames')

    # assigns variable parsed_args to parse_args()
    parsed_args = parser.parse_args()

    # returns the args structure to the caller
    return parsed_args


def get_pet_labels(image_dir):
    """
    Creates a dictionary of pet labels based upon the filenames of the image
    files. Reads in pet filenames and extracts the pet image labels from the
    filenames and returns these label as petlabel_dic. This is used to check
    the accuracy of the image classifier model.
    Parameters:
     image_dir - The (full) path to the folder of images that are to be
                 classified by pretrained CNN models (string)
    Returns:
     petlabels_dic - Dictionary storing image filename (as key) and Pet Image
                     Labels (as value)
    """
    filename_list = listdir(image_dir)  # files in the pet folder

    # define the dictionary with filenamne as keys and labels as values
    petlabels_dic = {f: f[0:f.rfind('_')].lower().replace('_', ' ').strip()
                     for f in filename_list if f[0] != '.'}
    # or
    """
    petlabels_dic = {}
    for pet_file in filename_list:

        if(pet_file[0] != '.'):  # if it's not a hidden file (mac/linux)
            pet_name = pet_file.lower()  # change it to all lower case
            pet_name_words = pet_name.split('_')  # get all individual words
            pet_label = ""

            for pet_word in pet_name_words:
                if pet_word.isalpha():  # if word only has letters
                    pet_label += pet_word + ' '  # add word to the label

            pet_label = pet_label.strip()  # remove spaces from the start/end

            if pet_file not in petlabels_dic.keys():  # if key is a new one
                petlabels_dic[pet_file] = pet_label
            else:  # impossible to have duplicated filenames here...
                print("Duplicated file found in the directory!")
    """
    return petlabels_dic


def classify_images(images_dir, petlabel_dic, model):
    """
    Creates classifier labels with classifier function, compares labels, and
    creates a dictionary containing both labels and comparison of them to be
    returned.
     PLEASE NOTE: This function uses the classifier() function defined in
     classifier.py within this function. The proper use of this function is
     in test_classifier.py Please refer to this program prior to using the
     classifier() function to classify images in this function.
     Parameters:
      images_dir - The (full) path to the folder of images that are to be
                   classified by pretrained CNN models (string)
      petlabel_dic - Dictionary that contains the pet image(true) labels
                     that classify what's in the image, where its' key is the
                     pet image filename & it's value is pet image label where
                     label is lowercase with space between each word in label
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
     Returns:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)   where 1 = match between pet image and
                    classifer labels and 0 = no match between labels
    """
    results_dic = {}

    for filename in petlabel_dic:
        classifier_label = classifier(
            images_dir+filename, model).lower().strip()

        real_label = petlabel_dic[filename]

        found_idx = classifier_label.find(real_label)

        if (classifier_label == real_label) or (
            classifier_label.startswith(real_label) or (
                classifier_label[found_idx - 1] == ' ') and (
                    classifier_label.endswith(real_label) or (
                        classifier_label[found_idx+len(real_label):found_idx +
                                         len(real_label)+1] in (',', ' ')))):

            if filename not in results_dic.keys():
                results_dic[filename] = [real_label, classifier_label, 1]

        elif filename not in results_dic.keys():
            results_dic[filename] = [real_label, classifier_label, 0]

    return results_dic


def adjust_results4_isadog(results_dic, dogsfile):
    """
    Adjusts the results dictionary to determine if classifier correctly
    classified images 'as a dog' or 'not a dog' especially when not a match.
    Demonstrates if model architecture correctly classifies dog images even if
    it gets dog breed wrong (not a match).
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    --- where idx 3 & idx 4 are added by this function ---
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
     dogsfile - A text file that contains names of all dogs from ImageNet
                1000 labels (used by classifier model) and dog names from
                the pet image files. This file has one dog name per line
                dog names are all in lowercase with spaces separating the
                distinct words of the dogname. This file should have been
                passed in as a command line argument. (string - indicates
                text file's name)
    Returns:
           None - results_dic is mutable data type so no return needed.
    """
    dognames_dic = {}

    with open(dogsfile, 'r') as f:
        line = f.readline()  # read first line

        while line != '':  # while the line to process has text
            if line not in dognames_dic.keys():
                # add the new breed if it don't exists
                dognames_dic[line.rstrip()] = 1
            else:  # print warning
                print("Warning: a duplicated line was found!")

            line = f.readline()  # read next line

    for data in results_dic.values():  # for each classified result
        # add a 1 if the image label is of a dog or 0 if not
        data.append(1 if data[0] in dognames_dic.keys() else 0)

        # add a 1 if the classifier label is of a dog or 0 if not
        data.append(1 if data[1] in dognames_dic.keys() else 0)


def calculates_results_stats(results_dic):
    """
    Calculates statistics of the results of the run using classifier's model
    architecture on classifying images. Then puts the results statistics in a
    dictionary (results_stats) so that it's returned for printing as to help
    the user to determine the 'best' model for classifying images. Note that
    the statistics calculated as the results are either percentages or counts.
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
    Returns:
     results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
    """
    results_stats = {}

    # calculating required counts

    n_imgs = len(results_dic)  # number of images in the results
    n_dog_imgs = 0  # number of images of dogs in the results
    n_correct_dogs = 0  # number of correctly classified dogs
    n_correct_not_dogs = 0  # number of correctly classified not dogs
    n_correct_breed = 0  # number of correctly classified breeds of dogs

    for image in results_dic:
        # [idx_3] = 1 if control label is of a dog and 0 is not
        n_dog_imgs += results_dic[image][3]
        if results_dic[image][3] == 1 and results_dic[image][4] == 1:
            n_correct_dogs += 1
        elif results_dic[image][3] == 0 and results_dic[image][4] == 0:
            n_correct_not_dogs += 1

        if results_dic[image][2] == 1 and results_dic[image][3] == 1:
            n_correct_breed += 1

    # number of images of not-dogs in the results
    n_not_dog_imgs = n_imgs - n_dog_imgs

    results_stats['n_images'] = n_imgs
    results_stats['n_dogs_img'] = n_dog_imgs
    results_stats['n_notdogs_img'] = n_not_dog_imgs
    results_stats['n_correct_dogs'] = n_correct_dogs
    results_stats['n_correct_notdogs'] = n_correct_not_dogs
    results_stats['n_correct_breed'] = n_correct_breed

    # calculating required percentages

    if n_dog_imgs > 0:
        results_stats['pct_correct_dogs'] = n_correct_dogs/n_dog_imgs*100.0
        results_stats['pct_correct_breed'] = n_correct_breed/n_dog_imgs*100.0
    else:
        results_stats['pct_correct_dogs'] = 0
        results_stats['pct_correct_breed'] = 0

    if n_not_dog_imgs > 0:
        results_stats['pct_correct_notdogs'] = (
            n_correct_not_dogs/n_not_dog_imgs*100.0)
    else:
        results_stats['pct_correct_notdogs'] = 0

    return results_stats


def print_results(results_dic, results_stats, model,
                  print_incorrect_dogs=False, print_incorrect_breed=False):
    """
    Prints summary results on the classification and then prints incorrectly
    classified dogs and incorrectly classified dog breeds if user indicates
    they want those printouts (use non-default values)
    Parameters:
      results_dic - Dictionary with key as image filename and value as a List
             (index)idx 0 = pet image label (string)
                    idx 1 = classifier label (string)
                    idx 2 = 1/0 (int)  where 1 = match between pet image and
                            classifer labels and 0 = no match between labels
                    idx 3 = 1/0 (int)  where 1 = pet image 'is-a' dog and
                            0 = pet Image 'is-NOT-a' dog.
                    idx 4 = 1/0 (int)  where 1 = Classifier classifies image
                            'as-a' dog and 0 = Classifier classifies image
                            'as-NOT-a' dog.
      results_stats - Dictionary that contains the results statistics (either a
                     percentage or a count) where the key is the statistic's
                     name (starting with 'pct' for percentage or 'n' for count)
                     and the value is the statistic's value
      model - pretrained CNN whose architecture is indicated by this parameter,
              values must be: resnet alexnet vgg (string)
      print_incorrect_dogs - True prints incorrectly classified dog images and
                             False doesn't print anything(default) (bool)
      print_incorrect_breed - True prints incorrectly classified dog breeds and
                              False doesn't print anything(default) (bool)
    Returns:
           None - simply printing results.
    """

    print("*** Classification Results for the CNN Model ", model.upper(),
          "***")

    print("\nNumber of Images: {}".format(results_stats['n_images']))
    print("Number of Dog Images: {}".format(results_stats['n_dogs_img']))
    print('Number of "Not-a" Dog Images: {}'.format(
        results_stats['n_notdogs_img']))

    print("\n% Correct Dogs: {}%".format(results_stats['pct_correct_dogs']))
    print("% Correct Breed: {}%".format(results_stats['pct_correct_breed']))
    print('% Correct "Not-a" Dog: {}%'.format(
        results_stats['pct_correct_notdogs']))

    if print_incorrect_dogs and (
            (results_stats['n_correct_dogs'] +
             results_stats['n_correct_notdogs']) !=
            results_stats['n_images']):

        print("\nIncorrect Dog/Not Dog Classifications:")

        for result in results_dic:
            if sum(results_dic[result][3:]) == 1:
                print("Real: %-30s Classifier %-30s" %
                      (results_dic[result][0], results_dic[result][1]))

    if print_incorrect_breed and (results_stats['n_correct_dogs'] !=
                                  results_stats['pct_correct_breed']):

        print("\nIncorrect Breeds Classifications:")

        for result in results_dic:
            if sum(results_dic[result][3:]) == 2 and (
                    results_dic[result][2] == 0):
                print("Real: %-30s Classifier %-30s" %
                      (results_dic[result][0], results_dic[result][1]))


def print_pretty_time(total_seconds):
    """
    Receives an elapsed time in seconds and returns a string with the time
    expressed in hh:mm:ss

    Arguments:
            total_seconds {float} -- Total time expressed in seconds
    Returns:
            a string with the time expressed in the format hh:mm:ss
    """
    hours = int((total_seconds / 3600))
    minutes = int(((total_seconds % 3600) / 60))
    seconds = int(((total_seconds % 3600) % 60))

    return "{}:{}:{}".format(str(hours).zfill(2),
                             str(minutes).zfill(2),
                             str(seconds).zfill(2))


# Call to main function to run the program
if __name__ == "__main__":
    main()
