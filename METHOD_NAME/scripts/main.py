import click
import os
import logging

import numpy as np
import torch

import quantus
from zennit import attribution as zattr
from zennit import composites as zcomp

from models import models
from data import dataloaders, datasets, transforms
from attribution import zennit_utils as zutils
from utils import arguments as argument_utils

@click.group()
def main():
    pass

@main.command(name="evaluate-randomisation")
@click.argument("dataset-name", type=str, required=True)
@click.argument("data-path", type=click.Path(), required=True)
@click.argument("labelmap-path", type=click.Path(), required=True)
@click.argument("model-name", type=str, required=True)
@click.argument("save-path", type=click.Path(), required=True)
@click.option("--use-cpu", type=bool, default=False, required=False)
@click.option("--batch-size", type=int, default=32, required=False)
@click.option("--shuffle", type=bool, default=False, required=False)
def randomization(
        dataset_name,
        data_path,
        labelmap_path,
        model_name,
        save_path,
        use_cpu,
        batch_size,
        shuffle
):
    """
    Evaluate
    """
    # Set experiment path using current timestamp
    save_path = os.path.join(save_path, dataset_name, model_name)
    os.makedirs(save_path, exist_ok=True)

    # Save arguments
    print("Saving arguments...")
    args = locals()
    argument_utils.save_arguments(args, save_path, "arguments.json")

    # Find correct device
    print("Preparing device and transforms...")
    device = torch.device('cuda' if torch.cuda.is_available() and not use_cpu else 'cpu')

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

    # TODO: Remove
    dataset.samples = dataset.samples[:50]

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
    metrics = {
        "sampling-based-model-parameter-randomisation": quantus.ModelParameterRandomisationSampling(
            num_draws = 1,
            similarity_func = quantus.similarity_func.ssim,
            layer_order = "bottom_up",
            seeds = [42],
            return_sample_correlation = False,
            abs = False,
            normalise = True,
            normalise_func = quantus.normalise_func.normalise_by_average_second_moment_estimate,
            return_aggregate = False,
            aggregate_func = None,
            default_plot_func = None,
            disable_warnings = False,
            display_progressbar = True
        )
    }
    xai_methods = {
        "gradient": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "canonizer": None,
            "composite": None,
            "attributor_kwargs": {},
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device
        },
        "lrp-epsilon": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "canonizer": zutils.get_zennit_canonizer(model),
            "composite": zutils.Epsilon,
            "attributor_kwargs": {},
            "canonizer_kwargs": {},
            "composite_kwargs": {
                "stabilizer": 1e-6,
                "epsilon": 1e-6
            },
            "device": device
        },
        "lrp-zplus": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "canonizer": zutils.get_zennit_canonizer(model),
            "composite": zutils.ZPlus,
            "attributor_kwargs": {},
            "canonizer_kwargs": {},
            "composite_kwargs": {
                "stabilizer": 1e-6,
            },
            "device": device
        },
        "guided-backprop": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "canonizer": zutils.get_zennit_canonizer(model),
            "composite": zcomp.GuidedBackprop,
            "attributor_kwargs": {},
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device
        },
        "excitation-backprop": {
            "xai_lib": "zennit",
            "attributor": zattr.Gradient,
            "canonizer": zutils.get_zennit_canonizer(model),
            "composite": zcomp.ExcitationBackprop,
            "attributor_kwargs": {},
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device
        },
        "smoothgrad": {
            "xai_lib": "zennit",
            "attributor": zattr.SmoothGrad,
            "canonizer": None,
            "composite": None,
            "attributor_kwargs": {
                "noise_level": 0.1,
                "n_iter": 10,
            },
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device
        },
        "intgrad": {
            "xai_lib": "zennit",
            "attributor": zattr.IntegratedGradients,
            "canonizer": None,
            "composite": None,
            "attributor_kwargs": {
                "baseline_fn": None,
                "n_iter": 10,
            },
            "canonizer_kwargs": {},
            "composite_kwargs": {},
            "device": device
        },
    }


    # Iterate over Batches and Evaluate
    results = []
    for batch, labels in loader:

        results.append(quantus.evaluate(
            metrics = metrics,
            xai_methods = xai_methods,
            model = model,
            x_batch = batch.numpy(),
            y_batch = labels.numpy(),
            s_batch = None,
            agg_func = lambda x: x,
            progress = True,
            explain_func_kwargs = None,
            call_kwargs = {"set1": {"device": device}},
        ))

        print(results)
        exit()



if __name__ == "__main__":
    main()