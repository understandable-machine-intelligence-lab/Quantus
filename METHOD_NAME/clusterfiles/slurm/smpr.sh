#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --job-name=smpr
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1

data="$1"
model="$2"
xai_methodname="$3"
xai_n_noisedraws="$4"
xai_noiselevel="$5"
eval_layer_order="$6"
eval_n_perturbations="$7"
eval_perturbation_noiselevel="$8"
eval_n_randomizations="${9}"
seed="${10}"
wandb_key="${11}"
partition="${12}"

SUBMIT_DIR=/data/cluster/users/lweber/smpr/gpu-cluster-results-2
mkdir -p ${SUBMIT_DIR}

source "/etc/slurm/local_job_dir.sh"
mkdir -p ${LOCAL_JOB_DIR}/results
mkdir -p ${LOCAL_JOB_DIR}/datasets
mkdir -p ${LOCAL_JOB_DIR}/labelmaps

echo "COPYING LABELMAPS..."
cp -r ${HOME}/Quantus/METHOD_NAME/scripts/label_map_imagenet.json ${LOCAL_JOB_DIR}/labelmaps

dir ${LOCAL_JOB_DIR}
dir ${LOCAL_JOB_DIR}/results

/usr/bin/time -v singularity \
  run \
        --nv \
        --bind /data/datapool/datasets/ImageNet-complete/:/mnt/datasets/${data} \
        --bind ${LOCAL_JOB_DIR}/labelmaps:/mnt/labelmaps/ \
        --bind ${LOCAL_JOB_DIR}/results:/mnt/output/ \
        ../singularity/smpr.sif evaluate-randomisation \
          ${data} \
          /mnt/datasets/${data} \
          /mnt/labelmaps/label_map_imagenet.json \
          ${model} \
          ${xai_methodname} \
          ${xai_n_noisedraws} \
          ${xai_noiselevel} \
          ${eval_layer_order} \
          ${eval_n_perturbations} \
          ${eval_perturbation_noiselevel} \
          ${eval_n_randomizations} \
          /mnt/output/ \
          --use-cpu True \
          --batch-size 32 \
          --shuffle True \
          --seed ${seed} \
          --wandb-key ${wandb_key} \

base=${data}-${model}-${xai_methodname}-${xai_n_noisedraws}-${xai_noiselevel}-${eval_layer_order}-${eval_n_perturbations}-${eval_perturbation_noiselevel}-${eval_n_randomizations}-${seed} \
cd ${LOCAL_JOB_DIR}
tar -czf ${base}.tgz results
cp ${base}.tgz ${SLURM_SUBMIT_DIR}
rm -rf ${LOCAL_JOB_DIR}/results

