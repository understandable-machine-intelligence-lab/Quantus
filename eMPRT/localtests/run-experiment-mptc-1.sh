#!/bin/bash

seed="$1"
wandb_key="$2"

data="imagenet"
models=(vgg16 resnet18)
xai_methodnames=(gradient smoothgrad lrp-epsilon lrp-zplus guided-backprop excitation-backprop)
eval_metricname="emprt"
xai_n_noisedraws=1
xai_noiselevel=0.0
eval_layer_orders=(top_down bottom_up)
smpr_n_perturbations=1
smpr_perturbation_noiselevel=0.1
smpr_n_randomizations=1
emprt_quality_func="entropy"
emprt_nr_samples=10
wandb_projectname="testing-emprt"

for model in "${models[@]}"; do
    for xai_methodname in "${xai_methodnames[@]}"; do
        if [ ${xai_methodname} == "smoothgrad" ]; then
            xai_n_noisedraws=50
            xai_noiselevel=0.1
        fi;
        for eval_layer_order in "${eval_layer_orders[@]}"; do

            python ../scripts/main.py evaluate-randomisation \
                                    /media/lweber/f3ed2aae-a7bf-4a55-b50d-ea8fb534f1f51/emprt/ \
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
                                    --emprt-quality-func ${emprt_quality_func} \
                                    --emprt-nr-samples ${emprt_nr_samples} \
                                    --use-cpu False \
                                    --batch-size 32 \
                                    --shuffle True \
                                    --seed ${seed} \
                                    --wandb-key ${wandb_key} \
                                    --wandb-projectname ${wandb_projectname} \
                                    --use-wandb False

        done;
    done;
done;