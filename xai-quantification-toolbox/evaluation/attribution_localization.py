import os
import argparse
import numpy as np
import time
import tracemalloc
import pandas as pd

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.universal_helper import compute_relevance_path, extract_filename


def get_explanation(relevance_path, data_name, model_name, layer, xai_method, filename, label):
    """ Load explanation for given filename and label. """
    filename = extract_filename(filename)
    explanation_dir = compute_relevance_path(relevance_path, data_name, model_name, layer, xai_method)
    fname = os.path.join(explanation_dir, "val", str(label), filename)

    explanation = np.load(fname + ".npy")
    return explanation


def attribution_localization(data_path, data_name, dataset_name, relevance_path, partition, batch_size, model_name, layer_names, xai_method, output_dir):
    """ Computes the attribution localization score. """

    # get input layer
    if isinstance(layer_names, list):
        input_layer = layer_names[0]
    else:
        input_layer = layer_names

    # initialize dataset and dataloader
    dataset = get_dataset(dataset_name)
    dataset = dataset(data_path, "val")
    dataset.set_mode("binary_mask")

    dataloader = DataLoader(dataset, batch_size=batch_size)

    total_scores = []
    weighted_scores = []

    for batch in dataloader:
        for sample in batch:
            sample_score = 0.0
            sample_weighted_score = 0.0
            for label in sample.label:
                # get attribution according to label
                explanation = get_explanation(relevance_path, data_name, model_name, input_layer, xai_method, sample.filename, dataset.classname_to_idx(label))
                binary_mask = sample.binary_mask[label]

                # check on any positive value in explanation
                if not np.all((explanation < 0.0)):

                    # preprocess explanation (different methods possible)
                    explanation = np.max(explanation, axis=2)

                    # filter positive explanation values
                    explanation[explanation < 0.0] = 0.0

                    # compute inside - total relevance ratios
                    binary_mask = binary_mask.astype(bool)[:, :, 0]
                    # binary_mask = np.repeat(binary_mask, 3, 2)

                    assert explanation.shape == binary_mask.shape

                    if not np.any(binary_mask):
                        print("no True values in binary mask discovered: {}, {}".format(sample.filename, dataset.classname_to_idxlabel))

                    inside_explanation = np.sum(explanation[binary_mask])
                    total_explanation = np.sum(explanation)

                    size_bbox = float(np.sum(binary_mask))
                    size_data = float(np.shape(binary_mask)[0] * np.shape(binary_mask)[1])

                    if inside_explanation / total_explanation > 1.0:
                        print("inside explanation {} greater than total explanation {}".format(inside_explanation, total_explanation))
                        # raise ValueError("inside explanation {} greater than total explanation {}".format(inside_explanation, total_explanation))
                    sample_score += (inside_explanation / total_explanation)
                    sample_weighted_score += ((inside_explanation / total_explanation) * (size_data / size_bbox))
                else:
                    print(sample.filename, dataset.classname_to_idx(label))

            total_scores.append(sample_score / len(sample.label))
            weighted_scores.append(sample_weighted_score / len(sample.label))

    total_score = float(np.sum(total_scores)) / len(total_scores)
    weighted_score = float(np.sum(weighted_scores) / len(weighted_scores))

    # save results
    results = [[data_name, model_name, xai_method, str(total_score), str(weighted_score)]]

    df = pd.DataFrame(results,
                      columns=['dataset', 'model', 'method', 'total_score', 'weighted score'])
    df.to_csv(os.path.join(output_dir, "{}_{}_{}.csv".format(data_name, model_name, xai_method)), index=False)


if __name__ == "__main__":
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
    parser.add_argument("-l", "--layer", type=str, default=None, help="Layer to compute relevance maps for")
    parser.add_argument("-bs", "--batch_size", type=int, default=50, help="Batch size for relevance map computation")

    ARGS = parser.parse_args()

    #####################
    #       MAIN
    #####################

    print("start explanation localization score estimation now")
    start = time.process_time()
    tracemalloc.start()

    attribution_localization(ARGS.data_path,
                             ARGS.data_name,
                             ARGS.dataset_name,
                             ARGS.relevance_datapath,
                             ARGS.partition,
                             ARGS.batch_size,
                             ARGS.model_name,
                             ARGS.layer,
                             ARGS.rule,
                             ARGS.output_dir)
    
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print("Duration of attribution localization score estimation:")
    print(time.process_time() - start)
