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

def smoothexplain(model, inputs, targets, **kwargs):

    xai_noiselevel = kwargs.pop("xai_noiselevel", 0.1)
    xai_n_noisedraws = kwargs.pop("xai_n_noisedraws", 1)
    
    device = kwargs.get("device", None)
    if not isinstance(inputs, torch.Tensor):
        inputs = torch.Tensor(inputs).to(device)

    if not isinstance(targets, torch.Tensor):
        targets = torch.as_tensor(targets).to(device)

    dims = tuple(range(1, inputs.ndim))
    std = xai_noiselevel * (inputs.amax(dims, keepdim=True) - inputs.amin(dims, keepdim=True))

    result = np.zeros(inputs.shape)
    for n in range(xai_n_noisedraws):
        # the last epsilon is defined as zero to compute the true output,
        # and have SmoothGrad w/ n_iter = 1 === gradient
        if n == xai_n_noisedraws - 1:
            epsilon = torch.zeros_like(inputs)
        else:
            epsilon = torch.randn_like(inputs) * std
            
        expl = quantus.explain(model, inputs + epsilon, targets, **kwargs)
        result += expl / xai_n_noisedraws

    return result

def plot_smpr_experiment(
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

def plot_emprt_experiment(
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
    ax.set_ylabel(kwargs.get("quality_func", "Score"))
    ax.set_xlabel("Layers")
    #ax.set_ylim([0, 1])

    for method in results.keys():
        layers = list(results[method].keys())
        scores: Dict[Any, Any] = {k: [] for k in layers}
        for layer in layers:
            for sample in results[method][layer]:
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
@click.option("--nr-test-samples", type=int, default=1000, required=False)
@click.option("--xai-n-noisedraws", type=int, default=1, required=False)
@click.option("--xai-noiselevel", type=float, default=0.0, required=False)
@click.option("--eval-layerorder", type=str, default="bottom_up", required=False)
@click.option("--eval-normalise", type=bool, default=True, required=False)
@click.option("--use-cpu", type=bool, default=False, required=False)
@click.option("--batch-size", type=int, default=32, required=False)
@click.option("--shuffle", type=bool, default=False, required=False)
@click.option("--wandb-key", type=str, default="", required=False)
@click.option("--wandb-projectname", type=str, default="", required=False)
@click.option("--use-wandb", type=bool, default=True, required=False)
def evaluate_randomization(
        save_path,
        dataset_name,
        data_path,
        labelmap_path,
        model_name,
        xai_methodname,
        eval_metricname,
        nr_test_samples,
        xai_n_noisedraws,
        xai_noiselevel,
        eval_layerorder,
        eval_normalise,
        use_cpu,
        batch_size,
        shuffle,
        wandb_key,
        wandb_projectname,
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

    # Set wandb key
    if wandb_key is not None:
        os.environ["WANDB_API_KEY"] = wandb_key

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
    dataset.samples = dataset.samples[:nr_test_samples]
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
    for i, (batch, labels) in enumerate(loader):
        x_batch = batch.numpy()
        img_size = x_batch.shape[2]
        nr_channels = x_batch.shape[1]
        break

    # Prepare model
    model = models.get_model(model_name, device)
    model.eval()

    # xai-specific params:
    eval_abs = True if xai_methodname in ["gradient"] else False
    eval_normalise_func = quantus.normalise_func.normalise_by_relu if xai_methodname in ["grad-cam"] else quantus.normalise_func.normalise_by_average_second_moment_estimate

    # Prepare Quantus Eval
    if eval_metricname == "smprt":
        metric_kwargs = {
            "n_noisy_models": 1,
            "ng_std_level": 0.1,
            "n_random_models": 1,
            "similarity_func": quantus.similarity_func.ssim,
            "layer_order": eval_layerorder,
            "return_sample_correlation": False,
            "abs": eval_abs,
            "normalise": eval_normalise,
            "normalise_func": eval_normalise_func,
            "return_aggregate": False,
            "aggregate_func": None,
            "default_plot_func": None,
            "disable_warnings": False,
            "display_progressbar": False,
        }
        metric = quantus.sMPRT(
                **metric_kwargs
        )
    elif eval_metricname == "emprt":
        metric_kwargs = {
            "layer_order":  eval_layerorder,
            "return_average_sample_score": False,
            "complexity_func": quantus.complexity_func.discrete_entropy,
            "complexity_func_kwargs": {"n_bins": 100}, #scotts
            "return_delta_explanation_vs_model":  True,
            "similarity_func": quantus.correlation_spearman,
            "skip_layers": False,
            "abs": eval_abs,
            "normalise": eval_normalise,
            "normalise_func": eval_normalise_func,
            "return_aggregate": False,
            "aggregate_func": None,
            "default_plot_func": None,
            "disable_warnings": False,
            "display_progressbar": False,
        }
        metric = quantus.eMPRT(
                **metric_kwargs
        )
    else:
        raise ValueError("Please provide a valid eval_metricname")

    XAI_METHOD_KWARGS = {
        "lrp-zplus": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "composite": zutils.ZPlus,
            "canonizer": zutils.get_zennit_canonizer(model),
            "canonizer_kwargs": {},
            "composite_kwargs": {
                "stabilizer": 1e-6,
            },
            "device": device,
        },
        "gradient": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "canonizer": None,
            "composite": None,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
        "lrp-epsilon": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "composite": zutils.Epsilon,
            "canonizer": zutils.get_zennit_canonizer(model),
            "canonizer_kwargs": {},
            "composite_kwargs": {
                "stabilizer": 1e-6,
                "epsilon": 1e-6
            },
            "device": device,
        },
        "guided-backprop": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "canonizer": None,
            "composite": zcomp.GuidedBackprop,
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device,
        },
        "grad-cam" : {
            "xai_lib": "captum",
            "device": device,
            "gc_layer": eval("model.features[-2]") if model_name == "vgg16" else eval("list(model.named_modules())[61][1]"),
            "interpolate": (img_size, img_size),
            "interpolate_mode": "bilinear",
            "method": "LayerGradCam",
        }
    }

    # Set up wandb
    id = wandb.util.generate_id()
    wandb.init(
        id=id,
        resume="allow",
        project=wandb_projectname,
        dir=wandbpath,
        config={
            "dataset_name": dataset_name,
            "model_name": model_name,
            "xai_methodname": xai_methodname,
            "xai_n_noisedraws": xai_n_noisedraws,
            "xai_noiselevel": xai_noiselevel,
            "eval_metricname": eval_metricname,
            "eval_layer_order": eval_layerorder,
            "use_cpu": use_cpu,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "nr_test_samples": nr_test_samples,
            "xai_kwargs": XAI_METHOD_KWARGS[xai_methodname],
            "metric_kwargs": metric_kwargs
        }
    )
    scores = {}
    scores["explanation_scores"] = {}
    if eval_metricname == "emprt":
        scores["model_scores"] = {}
        scores["delta_explanation_scores"] = []
        scores["delta_model_scores"] = []
        scores["fraction_explanation_scores"] = []
        scores["fraction_model_scores"] = []
        scores["delta_explanation_vs_models"] = []
    # Iterate over Batches and Evaluate
    for i, (batch, labels) in enumerate(loader):

        print("Evaluating Batch {}/{}".format(i+1, len(loader)))

        batch_results = metric(
            model=model,
            x_batch=batch.numpy(),
            y_batch=labels.numpy(),
            a_batch=None,
            device=device,
            explain_func=smoothexplain,
            explain_func_kwargs={**XAI_METHOD_KWARGS[xai_methodname], "xai_n_noisedraws": xai_n_noisedraws, "xai_noiselevel": xai_noiselevel}
        )

        if eval_metricname == "smprt":
            for k, v in batch_results.items():
                if k not in scores["explanation_scores"].keys():
                    scores["explanation_scores"][k] = []
                scores["explanation_scores"][k] += v[0]

        elif eval_metricname == "emprt":
            for k, v in metric.explanation_scores.items():
                if k not in scores["explanation_scores"].keys():
                    scores["explanation_scores"][k] = []
                scores["explanation_scores"][k] += v
            for k, v in metric.model_scores.items():
                if k not in scores["model_scores"].keys():
                    scores["model_scores"][k] = []
                scores["model_scores"][k] += v
            scores["delta_explanation_scores"] += metric.delta_explanation_scores
            scores["delta_model_scores"] += metric.delta_model_scores
            scores["fraction_explanation_scores"] += metric.fraction_explanation_scores
            scores["fraction_model_scores"] += metric.fraction_model_scores
            scores["delta_explanation_vs_models"] += metric.delta_explanation_vs_models

    resultsfile = os.path.join(save_path, "scores.json")
    with open(resultsfile, "w") as jsonfile:
        json.dump(scores, jsonfile)

    for k, v in scores["explanation_scores"].items():
        wandb.log({
            "mean-score": np.mean(v),
        })
    wandb.log({"scores": scores})

@main.command(name="plot-attribution-results")
@click.argument("path", type=str, required=True)
def plot_attribution_results(
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