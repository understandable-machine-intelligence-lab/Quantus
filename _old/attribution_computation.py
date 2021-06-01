import argparse
import datetime
import time
import os
import numpy as np
import tracemalloc

from ..dataloading.dataloader import DataLoader
from ..dataloading.custom import get_dataset
from ..helpers.model_helper import init_model
from ..helpers.universal_helper import extract_filename


def combine_path(output_dir, attributes):
    """ Computes directory path to save computed relevance to. """

    # remove backslash at the end
    if output_dir[-1] == "/":
        output_dir = output_dir[:-1]

    for attr in attributes:

        if not os.path.exists(output_dir + "/" + attr):
            try:
                os.makedirs(output_dir + "/" + attr)
            except FileExistsError:
                pass
        output_dir = output_dir + "/" + attr

    return output_dir


def compute_attribution_wrapper(data_path, data_name, dataset_name, partition, batch_size, model_path, model_name, model_type, layer_names, xai_method, class_name, output_dir, startidx=0, endidx=0):
    """ Wrapper Function to compute the attributed relevances for the selected class. """

    print("start relevance map computation now")
    start = time.process_time()

    print("compute explanations for layer(s): {}".format(layer_names))

    # init model
    model = init_model(model_path, model_name, framework=model_type)

    # initialize dataset
    dataset = get_dataset(dataset_name)
    dataset = dataset(data_path, partition)

    # compute/create output dir
    output_dir = combine_path(output_dir, [data_name, model_name])

    compute_attributions_for_class(dataset, partition, batch_size, model, layer_names,
                                   xai_method, class_name, output_dir, startidx=startidx, endidx=endidx)

    print("Duration of relevance map computation:")
    print(time.process_time() - start)
    print("Job executed successfully.")


def compute_attributions_for_class(dataset, partition, batch_size, model, layer_names,
                                   xai_method, class_name, output_dir, startidx=0, endidx=0):
    """ Computes the explanations for the selected class and saves them to output dir. """
    dataset.set_mode("preprocessed")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, startidx=startidx, endidx=endidx)

    for batch in dataloader:

        # extract preprocessed data
        data = [sample.datum for sample in batch]

        # compute relevance
        R = model.compute_relevance(data, layer_names, class_name, xai_method, additional_parameter=None)       # TODO: add additional parameter to pipeline

        for layer_name in layer_names:
            layer_output_dir = combine_path(output_dir, [layer_name, xai_method, partition, str(class_name)])
            for r, relevance in enumerate(R[layer_name]):
                fname = extract_filename(batch[r].filename)
                filename = os.path.join(layer_output_dir, fname + ".npy")
                np.save(filename, relevance)


if __name__ == "__main__":
    current_datetime = datetime.datetime.now()
    print(current_datetime)

    print("relevance computation")


    def decode_layernames(string):
        """ Decodes the layer_names string to a list of strings. """
        return string.split(":")


    # Setting up an argument parser for command line calls
    parser = argparse.ArgumentParser(description="Test and evaluate multiple xai methods")

    parser.add_argument("-d", "--data_path", type=str, default=None, help="data path")
    parser.add_argument("-dn", "--data_name", type=str, default=None, help="The name of the dataset to be used")
    parser.add_argument("-dl", "--dataset_name", type=str, default=None, help="The name of the dataloader class to be used.")
    parser.add_argument("-rd", "--relevance_datapath", type=str, default=None, help="data folder of relevance maps")
    parser.add_argument("-o", "--output_dir", type=str, default="./output", help="Sets the output directory for the results")
    parser.add_argument("-m", "--model_path", type=str, default=None, help="path to the model")
    parser.add_argument("-mn", "--model_name", type=str, default=None, help="Name of the model to be used")
    parser.add_argument("-mt", "--model_type", type=str, default=None, help="AI Framework to use (tensorflow, pytorch")
    parser.add_argument("-si", "--start_index", type=int, default=0, help="Index of dataset to start with")
    parser.add_argument("-ei", "--end_index", type=int, default=50000, help="Index of dataset to end with")
    parser.add_argument("-p", "--partition", type=str, default="train", help="Either train or test for one of these partitions")
    parser.add_argument("-cl", "--class_label", type=int, default=0, help="Index of class to compute heatmaps for")
    parser.add_argument("-r", "--rule", type=str, default="LRPSequentialCompositeA", help="Rule to be used to compute relevance maps")
    parser.add_argument("-l", "--layer_names", type=decode_layernames, default=None, help="Layer to compute relevance maps for")
    parser.add_argument("-bs", "--batch_size", type=int, default=50, help="Batch size for relevance map computation")

    ARGS = parser.parse_args()

    #####################
    #       MAIN
    #####################

    print("start relevance map computation now")
    start = time.process_time()
    tracemalloc.start()

    compute_attribution_wrapper(ARGS.data_path,
                                ARGS.data_name,
                                ARGS.dataset_name,
                                ARGS.partition,
                                ARGS.batch_size,
                                ARGS.model_path,
                                ARGS.model_name,
                                ARGS.model_type,
                                ARGS.layer_names,
                                ARGS.rule,
                                ARGS.class_label,
                                ARGS.relevance_datapath,
                                startidx=ARGS.start_index,
                                endidx=ARGS.end_index
                                )

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print("Relevance maps for x_data computed")
    print("Duration of relevance map computation:")
    print(time.process_time() - start)
    print("Job executed successfully.")
