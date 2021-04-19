import argparse
import os
import time
import tracemalloc
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr, pearsonr
from skimage.feature import hog
from skimage.metrics import structural_similarity as ssim
import pandas as pd

from ..dataloading.custom import get_dataset
from ..dataloading.dataloader import DataLoader
from ..helpers.model_helper import init_model
from ..helpers.universal_helper import extract_filename, compute_relevance_path


#######################################
# implementation of distance measures #
#######################################
def mse_distance(explanation, original_explanation):
    return (np.square(original_explanation - explanation)).mean(axis=None)


def cosine_distance(explanation, original_explanation):
    return cosine(np.ravel(original_explanation), np.ravel(explanation))


def ssim_distance(explanation, original_explanation):
    if len(original_explanation.shape) == 3:
        return ssim(original_explanation, explanation, multichannel=True)
    else:
        return ssim(original_explanation, explanation)


def spearman_distance(explanation, original_explanation):
    return spearmanr(np.ravel(original_explanation), np.ravel(explanation))[0]


def hog_distance(explanation, original_explanation):
    if len(original_explanation.shape) == 3:
        hog_original = hog(original_explanation, multichannel=True)
        hog_new = hog(explanation, multichannel=True)
    else:
        hog_original = hog(original_explanation[:, :, np.newaxis])
        hog_new = hog(explanation[:, :, np.newaxis])
    return pearsonr(hog_original, hog_new)[0]


DISTANCES = {
    "mse": mse_distance,
    "cosine": cosine_distance,
    "ssim": ssim_distance,
    "spearman": spearman_distance,
    "hog": hog_distance
}


#################################
# implement layer randomization #
#################################
def layer_randomization(model, dataloader, classidx, xai_method, bottom_layer, explanationdir, output_dir,
                        top_down=True, independent=False, distances=None):

    if not distances:
        distances = ["cosine"]

    # configure save dir
    if not os.path.exists(os.path.join(output_dir, "explanations")):
        os.makedirs(os.path.join(output_dir, "explanations"))

    # output_dir = join_path(output_dir, "explanations")

    # get layers including weights and iterate them
    layer_names = model.get_layer_names(with_weights_only=True)

    if top_down:
        layer_names = layer_names[::-1]

    results = {}

    for layer_name in layer_names:

        # init result dict for this layer
        diff = {}
        for distance in distances:
            diff[distance] = []

        # randomize layer weights
        if independent:
            model = init_model(model.path, model.name, framework=model.type)

        model = model.randomize_layer_weights(layer_name)

        # iterate data and compute explanations
        for batch in dataloader:
            data = [sample.datum for sample in batch]
            # labels = [sample.one_hot_label for sample in batch]

            explanations = model.compute_relevance(data, [bottom_layer], neuron_selection=int(classidx),
                                                   xai_method=xai_method, additional_parameter=None)

            # save explanations and compute diff to original explanation

            for i, explanation in enumerate(explanations[bottom_layer]):

                # np.save(join_path(output_dir, extract_filename(batch[i].filename)) + ".npy", explanation) # ToDo: save some explanations for visual inspection

                # compute similarity
                original_explanation = np.load(os.path.join(explanationdir, "val", str(classidx), extract_filename(batch[i].filename)) + ".npy")

                # normalize explanations
                original_explanation = original_explanation / np.max(original_explanation)
                explanation = explanation / np.max(np.abs(explanation))

                for distance in distances:

                    try:
                        # get distance function from dictionary
                        distance_function = DISTANCES[distance]
                    except KeyError:
                        raise NotImplementedError("{} metric not implemented.".format(distance))

                    # compute distance value and append
                    score = distance_function(explanation, original_explanation)
                    # confirm that the distance score is non-negative
                    score = np.abs(score)
                    # append
                    diff[distance].append(score)

        # compute results and save to dict
        for distance in distances:
            diff[distance] = np.mean(diff[distance])

        results[layer_name] = diff

    return results


def save_model_param_randomization_results(data_name, model_name, xai_method, classidx, class_results, outputdir):
    """ Save the results. """
    # save results
    results = []
    for layer in class_results:
        results.append([data_name, model_name, xai_method, str(classidx), layer] + [str(score) for score in list(class_results[layer].values())])

    df = pd.DataFrame(results,
                      columns=['dataset', 'model', 'method', 'classindex', 'layer'] + list(class_results[layer].keys()))
    df.to_csv(os.path.join(outputdir, "{}_{}_{}_{}.csv".format(data_name, model_name, xai_method, str(classidx))),
              index=False)


def model_parameter_randomization(data_path, data_name, dataset_name, classidx, partition, batch_size,
                                  model_path, model_name, model_type,
                                  bottom_layer, xai_method, explanationdir, output_dir, maxidx=None, distances=None):
    """ Function to create explanations on randomized models. """

    # set distance measure
    if not distances:
        distances = ["cosine"]

    # init model
    model = init_model(model_path, model_name, framework=model_type)

    explanationdir = compute_relevance_path(explanationdir, data_name, model_name, bottom_layer, xai_method)

    # configure directories
    if not os.path.exists(os.path.join(output_dir, "cascading_top_down")):
        os.makedirs(os.path.join(output_dir, "cascading_top_down"))
        os.makedirs(os.path.join(output_dir, "cascading_bottom_up"))
        os.makedirs(os.path.join(output_dir, "independent"))

    print("iteration for class index {}".format(classidx))

    # initialize dataset
    datasetclass = get_dataset(dataset_name)
    class_data = datasetclass(data_path, partition, classidx=[classidx])
    class_data.set_mode("preprocessed")

    print(type(class_data))

    if maxidx:
        dataloader = DataLoader(class_data, batch_size=batch_size, shuffle=True, endidx=maxidx)
    else:
        dataloader = DataLoader(class_data, batch_size=batch_size)

    # CASE 1: cascading layer randomization top-down
    print("case 1: cascading layer randomization top-down")
    case_output_dir = os.path.join(output_dir, "cascading_top_down")
    class_results = layer_randomization(model, dataloader, classidx, xai_method, bottom_layer,
                                        explanationdir, case_output_dir, top_down=True, distances=distances)
    # save results
    save_model_param_randomization_results(data_name, model_name, xai_method, classidx, class_results,
                                           case_output_dir)

    # CASE 2: cascading layer randomization bottom-up
    print("case 2: cascading layer randomization bottom-up")
    case_output_dir = os.path.join(output_dir, "cascading_bottom_up")
    class_results = layer_randomization(model, dataloader, classidx, xai_method, bottom_layer,
                                        explanationdir, case_output_dir, top_down=False, distances=distances)
    # save results
    save_model_param_randomization_results(data_name, model_name, xai_method, classidx, class_results,
                                           case_output_dir)

    # CASE 3: independent layer randomization
    print("case 3: independent layer randomization")
    case_output_dir = os.path.join(output_dir, "independent")
    class_results = layer_randomization(model, dataloader, classidx, xai_method, bottom_layer, explanationdir,
                                        case_output_dir, top_down=False, independent=True, distances=distances)
    # save results
    save_model_param_randomization_results(data_name, model_name, xai_method, classidx, class_results,
                                           case_output_dir)


if __name__ == "__main__":

    def decode_list(string):
        """ Decodes the layer_names string to a list of strings. """
        return string.split(":")

    print("model parameter randomization")
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
    parser.add_argument("-l", "--layer", type=decode_list, default=None, help="Layer to compute relevance maps for")
    parser.add_argument("-bs", "--batch_size", type=int, default=50, help="Batch size for relevance map computation")
    parser.add_argument("-mi", "--max_idx", type=int, default=None, help="Max class index to compute values for")
    parser.add_argument("-dm", "--distance_measure", type=decode_list, default=None, help="Distance measure to compute between explanations.")

    ARGS = parser.parse_args()

    #####################
    #       MAIN
    #####################

    print("start relevance map computation now")
    start = time.process_time()
    tracemalloc.start()

    model_parameter_randomization(ARGS.data_path,
                                  ARGS.data_name,
                                  ARGS.dataset_name,
                                  ARGS.class_label,
                                  ARGS.partition,
                                  ARGS.batch_size,
                                  ARGS.model_path,
                                  ARGS.model_name,
                                  ARGS.model_type,
                                  ARGS.layer[0],
                                  ARGS.rule,
                                  ARGS.relevance_datapath,
                                  ARGS.output_dir,
                                  maxidx=ARGS.max_idx,
                                  distances=ARGS.distance_measure
                                  )

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
    tracemalloc.stop()
    print("Model Parameter randomization done")
    print("Duration of score computation:")
    print(time.process_time() - start)
    print("Job executed successfully.")
