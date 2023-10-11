#!/bin/bash

partition="$1" # Cluster Partition
wandb_key="$2"
wandb_projectname="$3"
nr_test_samples="$4"

data=imagenet
eval_metricname=emprt
xai_noiselevel=0.1
model=vgg16

# Runs the experiments
counter=0

xai_methodname=lrp-epsilon
xai_n_noisedraws=1
eval_layer_order=top_down
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=1
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=1
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=1
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=5
eval_layer_order=top_down
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=5
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=5
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=5
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=10
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-epsilon
xai_n_noisedraws=30
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=1
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=5
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=10
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=25
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=30
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=35
eval_layer_order=top_down
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=35
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=40
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=50
eval_layer_order=top_down
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=100
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=guided-backprop
xai_n_noisedraws=150
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=1
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=5
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=20
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=35
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=35
eval_layer_order=top_down
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=40
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=45
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=50
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=100
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=150
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=200
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=lrp-zplus
xai_n_noisedraws=250
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=1
eval_layer_order=top_down
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=1
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=1
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=1
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=5
eval_layer_order=top_down
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=5
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=5
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=5
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=100
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=150
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=150
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=200
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=250
eval_layer_order=bottom_up
eval_normalise=true
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=250
eval_layer_order=bottom_up
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

xai_methodname=gradient
xai_n_noisedraws=300
eval_layer_order=top_down
eval_normalise=false
jid=$(sbatch -p ${partition} --parsable --output=%x-${data}-${model}-${xai_methodname}-${eval_metricname}-${nr_test_samples}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_normalise}-${wandb_projectname}-${partition}-%j.out smpr.sh ${data} ${model} ${xai_methodname} ${eval_metricname} ${nr_test_samples} ${xai_n_noisedraws} ${xai_noiselevel} ${eval_layer_order} ${eval_normalise} ${wandb_key} ${wandb_projectname})
counter=$((counter+1))

echo "Started " ${counter} " JOBS"
