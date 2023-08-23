import click
import os
import logging
from typing import List, Union, Dict, Any
import json
import sys

import numpy as np
import torch
import wandb

import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PIL import Image

from zennit import attribution as zattr
from zennit import image as zimage
from zennit import composites as zcomp

import quantus
print(quantus.__file__)

from models import models
from data import dataloaders, datasets, transforms
from attribution import zennit_utils as zutils
from utils import arguments as argument_utils

# This is for handling a sporadic error on the gpu-cluster
if not torch.cuda.is_available():
    sys.exit(77)

XAI_METHODS = {
        "gradient": {
            "xai_lib": "zennit",
            "attributor": zattr.SmoothGrad,
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
        },
        "lrp-epsilon": {
            "xai_lib": "zennit",
            "attributor": zattr.SmoothGrad,
            "composite": zutils.Epsilon,
            "canonizer_kwargs": {},
            "composite_kwargs": {
                "stabilizer": 1e-6,
                "epsilon": 1e-6
            },
        },
        "lrp-zplus": {
            "xai_lib": "zennit",
            "attributor": zattr.SmoothGrad,
            "composite": zutils.ZPlus,
            "canonizer_kwargs": {},
            "composite_kwargs": {
                "stabilizer": 1e-6,
            },
        },
        "guided-backprop": {
            "xai_lib": "zennit",
            "attributor": zattr.SmoothGrad,
            "composite": zcomp.GuidedBackprop,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
        },
        "excitation-backprop": {
            "xai_lib": "zennit",
            "attributor": zattr.SmoothGrad,
            "composite": zcomp.ExcitationBackprop,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
        },
    }

QUALITY_FUNCTIONS = {
    "entropy": quantus.functions.complexity_func.entropy
}

def plot_model_parameter_randomisation_experiment(
    results,
    *args,
    **kwargs,
) -> None:

    import matplotlib as mpl
    import matplotlib.font_manager as font_manager
    mpl.rcParams['font.family']='serif'
    cmfont = font_manager.FontProperties(fname=mpl.get_data_path() + '/fonts/ttf/cmr10.ttf')
    mpl.rcParams['font.serif']=cmfont.get_name()
    mpl.rcParams['mathtext.fontset']='cm'
    mpl.rcParams['axes.unicode_minus']=False
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 15})
    fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    ax.set_ylabel(kwargs.get("similarity_metric", "Score"))
    ax.set_xlabel("Layers")
    ax.set_ylim([0, 1])

    for method in results.keys():
        layers = list(results[method].keys())
        scores: Dict[Any, Any] = {k: [] for k in layers}
        for layer in layers:
            for rand in results[method][layer]:
                for sample in rand:
                    scores[layer].append(sample)

        ax.plot(layers, [np.mean(v) for k, v in scores.items()], label=method)

    ax.set_xticklabels(layers, rotation=90)
    ax.legend()

    fig.subplots_adjust(left=0.2, right=0.95, bottom=0.2, top=0.95, wspace=0, hspace=0)
    
    if kwargs.get("file_name", None) is not None:
        fig.savefig(kwargs.get("file_name", None))
    #plt.show()

def plot_heatmap(attr, dst, cmap="seismic", level=2.0):
    """
    Plots a single heatmaps from src .npy file, and
    """

    # Preprocess attributions
    attr = np.sum(attr, axis=0)
    amax = attr.max((0, 1), keepdims=True)
    attr = (attr + amax) / 2 / amax

    # Render and save image
    zimage.imsave(
        dst,
        attr,
        vmin=0.,
        vmax=1.,
        level=level,
        cmap=cmap
    )

def plot_input(img, dst, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    Plots a single input image from src .npy file, and saves it to dst as a .png
    """

    # TODO: Preprocess?
    img = img.transpose(1, 2, 0)
    img *= std
    img += mean
    img = img*255
    img = np.uint8(img)

    # Render and save image
    img = Image.fromarray(img, mode="RGB")
    img.save(dst)

@click.group()
def main():
    pass

@main.command(name="evaluate-randomisation")
@click.argument("save-path", type=click.Path(), required=True)
@click.argument("dataset-name", type=str, required=True)
@click.argument("data-path", type=click.Path(), required=True)
@click.argument("labelmap-path", type=click.Path(), required=True)
@click.argument("model-name", type=str, required=True)
@click.argument("xai-methodname", type=str, required=True)
@click.argument("eval-metricname", type=str, required=True)
@click.option("--xai-n-noisedraws", type=int, default=1, required=False)
@click.option("--xai-noiselevel", type=float, default=0.0, required=False)
@click.option("--eval-layerorder", type=str, default="bottom_up", required=False)
@click.option("--smpr-n-perturbations", type=int, default=1, required=False)
@click.option("--smpr-perturbation-noiselevel", type=float, default=0.1, required=False)
@click.option("--smpr-n-randomizations", type=int, default=1, required=False)
@click.option("--mptc-quality-func", type=str, default="entropy", required=False)
@click.option("--mptc-nr-samples", type=int, default=10, required=False)
@click.option("--use-cpu", type=bool, default=False, required=False)
@click.option("--batch-size", type=int, default=32, required=False)
@click.option("--shuffle", type=bool, default=False, required=False)
@click.option("--seed", type=int, default=None, required=False)
@click.option("--wandb-key", type=str, default="", required=False)
@click.option("--use-wandb", type=bool, default=True, required=False)
def randomization(
        save_path,
        dataset_name,
        data_path,
        labelmap_path,
        model_name,
        xai_methodname,
        eval_metricname,
        xai_n_noisedraws,
        xai_noiselevel,
        eval_layerorder,
        smpr_n_perturbations,
        smpr_perturbation_noiselevel,
        smpr_n_randomizations,
        mptc_quality_func,
        mptc_nr_samples,
        use_cpu,
        batch_size,
        shuffle,
        seed,
        wandb_key,
        use_wandb
):
    """
    Evaluate
    """

    # Set save paths
    save_path = os.path.join(save_path, dataset_name, model_name)
    os.makedirs(save_path, exist_ok=True)
    wandbpath = os.path.join(save_path, "wandb")
    os.makedirs(wandbpath, exist_ok=True)

    # Use Wandb?
    if not use_wandb:
        os.environ["WANDB_MODE"] = "disabled"
    else:
        os.environ["WANDB_MODE"] = "enabled"

    # Set wandb key
    if wandb_key is not None:
        os.environ["WANDB_API_KEY"] = wandb_key

    # Set up wandb
    id = wandb.util.generate_id()
    wandb.init(
        id=id,
        resume="allow",
        project="denoise-sanity-checks",
        dir=wandbpath,
        config={
            "dataset_name": dataset_name,
            "model_name": model_name,
            "xai_methodname": xai_methodname,
            "xai_n_noisedraws": xai_n_noisedraws,
            "xai_noiselevel": xai_noiselevel,
            "eval_metricname": eval_metricname,
            "eval_layer_order": eval_layerorder,
            "smpr_n_perturbations": smpr_n_perturbations,
            "smpr_perturbation_noiselevel": smpr_perturbation_noiselevel,
            "smpr_n_randomizations": smpr_n_randomizations,
            "mptc_quality_func": mptc_quality_func,
            "mptc_nr_samples": mptc_nr_samples,
            "use_cpu": use_cpu,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "seed": seed,
        }
    )

    # Save arguments
    print("Saving arguments...")
    args = locals()
    argument_utils.save_arguments(args, save_path, "arguments.json")

    # Find correct device
    print("Preparing device and transforms...")
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = None
        raise ValueError("CUDA IS NOT AVAILABLE")

    # Prepare transforms
    transform = transforms.get_transforms(dataset_name, mode="test")

    # Prepare datasets
    logging.info("Preparing datasets...")
    print("Preparing datasets...")
    dataset = datasets.get_dataset(
        dataset_name,
        data_path,
        transform,
        mode="test",
        labelmap_path=labelmap_path
    )

    print(f"Number of Samples in Dataset: {len(dataset.samples)}")
    dataset.samples = dataset.samples[:1000]
    print(f"Reduced of Samples in Dataset: {len(dataset.samples)}")

    # Prepare dataloaders
    logging.info("Preparing dataloaders...")
    print("Preparing dataloaders...")
    loader = dataloaders.get_dataloader(
        dataset_name = dataset_name,
        dataset = dataset,
        batch_size = batch_size,
        shuffle = shuffle,
    )

    # Prepare model
    model = models.get_model(model_name, device)
    model.eval()

    # Prepare Quantus Eval
    if eval_metricname == "smpr":
        metrics = {
            f"SMPR": quantus.ModelParameterRandomisationSampling(
                n_perturbations = smpr_n_perturbations,
                perturbation_noise_level = smpr_perturbation_noiselevel,
                n_randomisations = smpr_n_randomizations,
                similarity_func = quantus.similarity_func.ssim,
                layer_order = eval_layerorder,
                seed = seed,
                return_sample_correlation = False,
                abs = False,
                normalise = True,
                normalise_func = quantus.normalise_func.normalise_by_average_second_moment_estimate,
                return_aggregate = False,
                aggregate_func = None,
                default_plot_func = None,
                disable_warnings = False,
                display_progressbar = False,
            ),
        }
    elif eval_metricname == "mptc":
        metrics = {
            f"MPTC": quantus.MPT_Complexity(
                quality_func = QUALITY_FUNCTIONS[mptc_quality_func],
                layer_order = eval_layerorder,
                nr_samples = mptc_nr_samples,
                seed = seed,
                return_sample_entropy = False,
                abs = False,
                normalise = True,
                normalise_func = quantus.normalise_func.normalise_by_average_second_moment_estimate,
                return_aggregate = False,
                aggregate_func = None,
                default_plot_func = None,
                disable_warnings = False,
                display_progressbar = False,
            ),
        }
    else:
        metrics = {}
        raise ValueError("Please provide a valid eval_metricname")

    xai_attributor_kwargs = {
        "n_iter": xai_n_noisedraws,
        "noise_level": xai_noiselevel
    }
    xai_canonizer = zutils.get_zennit_canonizer(model)
    xai_methods = {xai_methodname: {
        **XAI_METHODS[xai_methodname], 
        "canonizer": xai_canonizer,
        "attributor_kwargs": xai_attributor_kwargs,
        "device": device,
    }}

    # Iterate over Batches and Evaluate
    results =  {m: {x: {} for x in xai_methods.keys()} for m in metrics.keys()} 
    for i, (batch, labels) in enumerate(loader):

        print("Evaluating Batch {}/{}".format(i+1, len(loader)))

        batch_results = quantus.evaluate(
            metrics = metrics,
            xai_methods = xai_methods,
            model = model,
            x_batch = batch.numpy(),
            y_batch = labels.numpy(),
            s_batch = None,
            agg_func = lambda x: x,
            progress = True,
            explain_func_kwargs = None,
            attributions_path = os.path.join(save_path, "attributions"),
            call_kwargs = {"set1": {"device": device}},
        )

        # Append correctly to results
        for m in metrics.keys():
            for x in xai_methods.keys():
                layers = list(batch_results[x][m]["set1"].keys())
                for l in layers:
                    if l not in results[m][x].keys():
                        results[m][x][l] = []
                    results[m][x][l] += batch_results[x][m]["set1"][l]

    resultsfile = os.path.join(save_path, "results.json")
    with open(resultsfile, "w") as jsonfile:
        json.dump(results, jsonfile)

    for metric, m_results in results.items():
        if eval_metricname == "smpr":
            plot_model_parameter_randomisation_experiment(
                m_results,
                similarity_score = "SSIM",
                file_name = os.path.join(save_path, f"{metric}.svg")
            )
        elif eval_metricname == "mptc":
            plot_model_parameter_randomisation_experiment(
                m_results,
                similarity_score = "Entropy",
                file_name = os.path.join(save_path, f"{metric}.svg")
            )
        for method in m_results.keys():
            layers = list(m_results[method].keys())
            for l, layer in enumerate(layers):
                scores = []
                for rand in m_results[method][layer]:
                    for sample in rand:
                        scores.append(sample)
                wandb.log({
                    "layer": layer,
                    "layer_id": l,
                    "mean-score": np.mean(scores),
                    "std-score": np.std(scores)
                })



@main.command(name="plot-attribution-results")
@click.argument("path", type=str, required=True)
def randomization(
        path,
):
    for layer in os.listdir(path):
        for file in os.listdir(os.path.join(path, layer)):
            if file.endswith(".npy"):
                fpath = os.path.join(path, layer, file)
                if "input" in file:
                    plot_input(np.load(fpath), fpath.split(".npy")[0]+".png", mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
                elif "original_attribution" in file:
                    plot_heatmap(np.load(fpath), fpath.split(".npy")[0]+".png", cmap="seismic", level=2.0)
                elif "perturbed_attribution" in file:
                    plot_heatmap(np.load(fpath), fpath.split(".npy")[0]+".png", cmap="seismic", level=2.0)

if __name__ == "__main__":
    main()