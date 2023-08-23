#!/bin/bash

seed="$1"

data="imagenet"
model="resnet18"
xai_methodname="gradient"
eval_metricname="mptc"
xai_n_noisedraws=1
xai_noiselevel=0.0
eval_layer_order="bottom_up"
smpr_n_perturbations=1
smpr_perturbation_noiselevel=0.1
smpr_n_randomizations=1
mptc_quality_func="entropy"
mptc_nr_samples=10

python ../scripts/main.py evaluate-randomisation \
                          /media/lweber/f3ed2aae-a7bf-4a55-b50d-ea8fb534f1f51/mptc/ \
                          ${data} \
                          /media/lweber/f3ed2aae-a7bf-4a55-b50d-ea8fb534f1f51/Datasets/imagenet \
                          ../scripts/label_map_imagenet.json \
                          ${model} \
                          ${xai_methodname} \
                          ${eval_metricname} \
                          --xai-n-noisedraws ${xai_n_noisedraws} \
                          --xai-noiselevel ${xai_noiselevel} \
                          --eval-layerorder ${eval_layer_order} \
                          --smpr-n-perturbations ${smpr_n_perturbations} \
                          --smpr-perturbation-noiselevel ${smpr_perturbation_noiselevel} \
                          --smpr-n-randomizations ${smpr_n_randomizations} \
                          --mptc-quality-func ${mptc_quality_func} \
                          --mptc-nr-samples ${mptc_nr_samples} \
                          --use-cpu False \
                          --batch-size 32 \
                          --shuffle True \
                          --seed ${seed} \
                          --use-wandb False