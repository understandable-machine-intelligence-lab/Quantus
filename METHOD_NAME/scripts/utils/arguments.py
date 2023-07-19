import os
import json

def save_arguments(arguments, savedir, filename):
    """
    Saves function arguments to specified file
    """

    # Check filename
    if not filename.endswith(".json"):
        raise ValueError("File extension {} is not supported. Please use '.json'.".format(filename.split(".")[-1]))

    # Create dir if it does not exist
    os.makedirs(savedir, exist_ok=True)

    # Create the checkpoint
    with open(os.path.join(savedir, filename), 'w') as file:
        json.dump(arguments, file)

def load_arguments(filename):
    """
    Loads function arguments from specified file
    """

    # Check filename
    if not filename.endswith(".json"):
        raise ValueError("File extension {} is not supported. Please use '.json'.".format(filename.split(".")[-1]))

    # Check if the path is correct
    if not os.path.exists(filename) or not os.path.isfile(filename):
        raise ValueError("{} is not a file or does not exist.".format(filename))

    # Load the arguments
    with open(filename, 'r') as file:
        arguments = json.load(file)

    # Return arguments
    return arguments