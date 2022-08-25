import os
n_threads = "1"

#os.environ["NVIDIA_VISIBLE_DEVICES"] = "3"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["OMP_NUM_THREADS"] = n_threads
os.environ["OPENBLAS_NUM_THREADS"] = n_threads
os.environ["MKL_NUM_THREADS"] = n_threads
os.environ["VECLIB_MAXIMUM_THREADS"] = n_threads
os.environ["NUMEXPR_NUM_THREADS"] = n_threads

import argparse
import joblib
import os
import shutil

from timeit import default_timer as timer
from datetime import timedelta

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, TensorDataset

from config import Config
from loading.load_data import load_xy_data, load_event_data
from preprocessing import get_monocular_statistics

# currently in source directory
from quantus import quantus


benchmark_metrics = {
    'batched': {
        'class': quantus.PixelFlipping,
        'init_kwargs': {
            "perturb_baseline": "mean",
            "normalise": False,
            "abs": False,
            "features_in_step": 2,
            "max_steps_per_input": 2,
            "disable_warnings": True,
            "display_progressbar": True,
        },
        'call_kwargs':{
        },
    },
    'multiprocess_pass_in_return_out': {
        'class': quantus.PixelFlippingMultiProcessPassInReturnOut,
        'init_kwargs': {
            "perturb_baseline": "mean",
            "normalise": False,
            "abs": False,
            "features_in_step": 2,
            "max_steps_per_input": 2,
            "disable_warnings": True,
            "display_progressbar": True,
        },
        'call_kwargs':{
            "batch_size": 64,
            "buffer_size": 640,
            "n_workers": 8,
        },
    },
    'multiprocess_shared_in_return_out': {
        'class': quantus.PixelFlippingMultiProcessSharedInReturnOut,
        'init_kwargs': {
            "perturb_baseline": "mean",
            "normalise": False,
            "abs": False,
            "features_in_step": 2,
            "max_steps_per_input": 2,
            "disable_warnings": True,
            "display_progressbar": True,
        },
        'call_kwargs':{
            "batch_size": 64,
            "buffer_size": 640,
            "n_workers": 8,
        },
    },
    'multiprocess_shared_in_out': {
        'class': quantus.PixelFlippingMultiProcessSharedInOut,
        'init_kwargs': {
            "perturb_baseline": "mean",
            "normalise": False,
            "abs": False,
            "features_in_step": 2,
            "max_steps_per_input": 2,
            "disable_warnings": True,
            "display_progressbar": True,
        },
        'call_kwargs':{
            "batch_size": 64,
            "buffer_size": 640,
            "n_workers": 8,
        },
    },
    'multiprocess_shared_in': {
        'class': quantus.PixelFlippingMultiProcessSharedIn,
        'init_kwargs': {
            "perturb_baseline": "mean",
            "normalise": False,
            "abs": False,
            "features_in_step": 2,
            "max_steps_per_input": 2,
            "disable_warnings": True,
            "display_progressbar": True,
        },
        'call_kwargs':{
        },
    },
    'multiprocess': {
        'class': quantus.PixelFlippingMultiProcess,
        'init_kwargs': {
            "perturb_baseline": "mean",
            "normalise": False,
            "abs": False,
            "features_in_step": 2,
            "max_steps_per_input": 2,
            "disable_warnings": True,
            "display_progressbar": True,
        },
        'call_kwargs':{
        },
    },
    'multiprocess_multithreading_pass_in_return_out': {
        'class': quantus.PixelFlippingMultiThreadingPassInReturnOut,
        'init_kwargs': {
            "perturb_baseline": "mean",
            "normalise": False,
            "abs": False,
            "features_in_step": 2,
            "max_steps_per_input": 2,
            "disable_warnings": True,
            "display_progressbar": True,
        },
        'call_kwargs':{
            "batch_size": 64,
            "buffer_size": 640,
            "n_workers": 8,
        },
    },
    'multiprocess_multithreading_queue': {
        'class': quantus.PixelFlippingMultiThreadingQueue,
        'init_kwargs': {
            "perturb_baseline": "mean",
            "normalise": False,
            "abs": False,
            "features_in_step": 2,
            "max_steps_per_input": 2,
            "disable_warnings": True,
            "display_progressbar": True,
        },
        'call_kwargs':{
            "batch_size": 64,
            "buffer_size": 640,
            "n_workers": 8,
        },
    },
}




def one_hot_max(output):
    '''Get the one-hot encoded max at the original indices in dim=1'''
    values, indices = output.max(1)
    return values[:, None] * torch.eye(output.shape[1]).to(output.device)[indices]


def benchmark_quantus_metrics(
        model,
        X, Y, A,
        S=None,
) -> np.ndarray:

    for metric_name, metric_config in benchmark_metrics.items():
        print('benchmark', metric_name)
        
        start_time = timer()

        metric_class = metric_config['class']
        metric_init_kwargs = metric_config['init_kwargs']
        metric_call_kwargs = metric_config['call_kwargs']

        metric = metric_class(**metric_init_kwargs)
        score = metric(
            model=model,
            x_batch=X,
            y_batch=Y,
            a_batch=A,
            s_batch=S,
            device=model.device,
            model_predict_kwargs={
                'device': model.device,
            },
            **metric_call_kwargs,
        )
        
        end_time = timer()
        computation_time = end_time - start_time
        print(f'{metric_name} computation time:', timedelta(seconds=computation_time))

    return None
    

def evaluate_attributions_for_folds_quantus(
        model_config,
        X, Y, Y_pred, A, fold_indices,
        model_dirpath,
        gpu_id,
):
    # import and set model class
    model_class = model_config['class']
    batch_size= model_config['batch_size']

    if model_config['framework'] != 'pytorch':
        raise TypeError('model framework needs to be pytorch but is: {model_config["framework"]}')

    # get model setup parameters
    seq_len = X.shape[1]
    n_channels = X.shape[2]
    n_classes = Y.shape[1]

    # swap axes for pytorch
    # tensorflow: channel axis is last axis
    # pytorch: channel axis is second-last axis
    X = np.swapaxes(X, 1, 2)
    A = np.swapaxes(A, 1, 2)

    '''
    df_events = load_event_data(event_filepaths=config.paths['events'])
    print(df_events)
    print("event types:", df_events.event_type.unique())


    S = np.zeros(X.shape)
    event_id = 0
    for _, event in tqdm(df_events.iterrows(), total=len(df_events)):
        start = max(0, event.start)
        end = min(A.shape[1], event.end)
        instance_id = int(event.instance_id)

        S[instance_id, event_id, start:end] = True
    '''

    # wait to initialize scores before width of scores is known
    scores = None
    
    for fold_id, idxs in tqdm(fold_indices.items()):
        # partition date into test and validation sets
        # partition data into test, train and validation sets
        idx_test, idx_train, idx_val = idxs['test'], idxs['train'], idxs['val']
        X_test, X_train, X_val = X[idx_test], X[idx_train], X[idx_val]
        Y_test, Y_train, Y_val = Y[idx_test], Y[idx_train], Y[idx_val]
        A_test, A_train, A_val = A[idx_test], A[idx_train], A[idx_val]
        #S_test, S_train, S_val = S[idx_test], S[idx_train], S[idx_val]

        # preprocess data if requested
        preprocessing = model_config.get('preprocessing')
        # default pass needed for simple error-catch in else-clause
        if preprocessing is None:
            pass
        elif preprocessing.startswith('zstd'):
            if preprocessing == 'zstd':
                # z-score normalization with train-set statistics
                mean = X_train.mean(axis=(0, 2), keepdims=True)
                std = X_train.std(axis=(0, 2), keepdims=True)
            elif preprocessing == 'zstd-mono':
                # z-score normalization on first two channels only
                mean = get_monocular_statistics(X_train, 'mean')
                std = get_monocular_statistics(X_train, 'std')
            else:
                raise ValueError('preprocessing setting "{preprocessing}" not valid')

            X_test = (X_test - mean) / std
            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std
        else:
            raise ValueError('preprocessing setting "{preprocessing}" not valid')

        # load keyword arguments for model initialization
        model_init_kwargs = model_config.get('init_kwargs', {})

        # replace placeholders with values
        for key, value in model_init_kwargs.items():
            if type(value) == str and value == '$n_classes':
                model_init_kwargs[key] = n_classes
            elif type(value) == str and value == '$n_channels':
                model_init_kwargs[key] = n_channels
            elif type(value) == str and value == '$mean':
                model_init_kwargs[key] = X_train.mean(axis=(0, 2), keepdims=True)
            elif type(value) == str and value == '$std':
                model_init_kwargs[key] = X_train.std(axis=(0, 2), keepdims=True)
            elif type(value) == str and value == '$mean-mono':
                model_init_kwargs[key] = get_monocular_statistics(X_train, 'mean')
            elif type(value) == str and value == '$std-mono':
                model_init_kwargs[key] = get_monocular_statistics(X_train, 'std')
        print('model_init_kwargs:', model_init_kwargs)

        # initialize model
        model = model_class(**model_init_kwargs)

        # load fitted model state dict
        model_filepath = os.path.join(model_dirpath,
                                      f'model_fold_{fold_id}.pth')
        model.load_state_dict(torch.load(model_filepath))
        model.eval()

        # move the model to the correct gpu device
        #model.to(f'cuda:{gpu_id}')
        model.to(f'cuda')

        scores_fold = benchmark_quantus_metrics(
            model=model,
            X=X,
            Y=Y,
            A=A,
            #S=S_test,
        )

        break



def main(input_key: str, model_key: str, explainer_key: str,
         gpu_id: str):
    print("Selected GPU ID:", gpu_id)
    print("Selected input:", input_key)
    print("Selected model:", model_key)
    print("Selected explainer:", explainer_key)

    # load config
    config = Config(
        input_key=input_key,
        model_key=model_key,
        explainer_key=explainer_key,
    )

    # load input data
    print("Loading input data...")
    selected_input_channels = config.inputs['X_channels']
    X, Y = load_xy_data(
        X_filepath=config.paths['X'],
        Y_filepath=config.paths['Y'],
        X_format_filepath=config.paths['X_format'],
        selected_input_channels=selected_input_channels)
    Y = np.load(config.paths['Y_labels']).astype(int)

    # load fold indices
    fold_indices = joblib.load(config.paths['fold_indices'])

    print("Selected input channels:", selected_input_channels)
    print("X.shape:", X.shape)
    print("Y.shape:", Y.shape)

    # check if explainer class fits model class
    if config.model['framework'] not in config.explainer['supported_model_frameworks']:
        raise NotImplementedError(
            f'Attribution method "{explainer_key}" does not support'
            f' model {model_key} of framework {config.model["framework"]}.'
            f'\r\nSupported frameworks are: {config.explainer["supported_model_frameworks"]}'
        )

    # load model predictions
    #Y_pred_filepath = os.path.join(config.paths['predictions'], 'Y_pred.npy')
    Y_pred_filepath = f'/home/krakowczyk/workspace/xai-timeseries/predictions/{input_key}/{model_key}/Y_pred.npy'
    Y_pred = np.load(Y_pred_filepath)

    # define attributions filepath
    A = np.random.uniform(low=-1.0, high=1.0, size=X.shape)

    # run attribution compuations
    evaluate_attributions_for_folds_quantus(
        model_config=config.model,
        X=X, Y=Y, Y_pred=Y_pred, A=A,
        fold_indices=fold_indices,
        gpu_id=gpu_id,
        model_dirpath=config.paths['models'],
    )



def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--gpu', type=int, required=True,
                        help='set gpu id for training')

    args = parser.parse_args()
    return {
        'gpu_id': args.gpu,
    }

    
if __name__ == '__main__' :
    arguments = parse_arguments()

    input_key = 'mnist1d_sl1000'
    model_key = 'cnn3_dense2_notrafo_pytorch'
    explainer_key = 'deeplift_zero'
    gpu_id = arguments['gpu_id']

    start_time = timer()
    main(
        input_key=input_key,
        model_key=model_key,
        explainer_key=explainer_key,
        gpu_id=gpu_id,
    )
    end_time = timer()
    computation_time = end_time - start_time
    print('Total computation time:', timedelta(seconds=computation_time))

