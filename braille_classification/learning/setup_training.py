import os
import shutil
import argparse
import numpy as np

from tactile_data.utils_data import save_json_obj

ARROW_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
ALPHA_NAMES = ['Q', 'W', 'E', 'R', 'T', 'Y', 'U', 'I', 'O', 'P',
               'A', 'S', 'D', 'F', 'G', 'H', 'J', 'K', 'L',
               'Z', 'X', 'C', 'V', 'B', 'N', 'M',
               'SPACE', 'NONE']


def csv_row_to_label(row):
    return {
        'id': np.array(row['obj_id'] - 1),
        'label': row['obj_lbl'],
    }


def setup_parse_args(
        tasks=['arrows'],
        models=['simple_cnn'],
        device='cuda'
):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['arrows', 'alphabet'].",
        default=tasks
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose model from ['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit'].",
        default=models
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default=device
    )
    # parse arguments
    args = parser.parse_args()
    return args.tasks, args.models, args.device


def setup_learning(save_dir=None):

    learning_params = {
        'seed': 42,
        'batch_size': 64,
        'epochs': 10,
        'lr': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 1e-6,
        'adam_b1': 0.9,
        'adam_b2': 0.999,
        'shuffle': True,
        'n_cpu': 1,
    }

    image_processing_params = {
        'dims': (128, 128),
        'bbox': None,
        'thresh': None,
        'stdiz': False,
        'normlz': True,
    }

    augmentation_params = {
        'rshift': (0.025, 0.025),
        'rzoom': None,
        'brightlims': None,
        'noise_var': None,
    }

    preproc_params = {
        'image_processing': image_processing_params,
        'augmentation': augmentation_params
    }

    if save_dir:
        save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
        save_json_obj(preproc_params, os.path.join(save_dir, 'preproc_params'))

    return learning_params, preproc_params


def setup_model(model_type, save_dir):

    model_params = {
        'model_type': model_type
    }

    if model_type == 'simple_cnn':
        model_params['model_kwargs'] = {
                'conv_layers': [32, 32, 32, 32],
                'conv_kernel_sizes': [11, 9, 7, 5],
                'fc_layers': [512, 512],
                'activation': 'relu',
                'dropout': 0.0,
                'apply_batchnorm': True,
        }

    elif model_type == 'posenet_cnn':
        model_params['model_kwargs'] = {
                'conv_layers': [256, 256, 256, 256, 256],
                'conv_kernel_sizes': [3, 3, 3, 3, 3],
                'fc_layers': [64],
                'activation': 'elu',
                'dropout': 0.0,
                'apply_batchnorm': True,
        }

    elif model_type == 'nature_cnn':
        model_params['model_kwargs'] = {
            'fc_layers': [512, 512],
            'dropout': 0.0,
        }

    elif model_type == 'resnet':
        model_params['model_kwargs'] = {
            'layers': [2, 2, 2, 2],
        }

    elif model_type == 'vit':
        model_params['model_kwargs'] = {
            'patch_size': 32,
            'dim': 128,
            'depth': 6,
            'heads': 8,
            'mlp_dim': 512,
            'pool': 'cls',  # for classification
        }

    # save parameters
    save_json_obj(model_params, os.path.join(save_dir, 'model_params'))

    return model_params


def setup_task(task_name):
    """
    Returns task specific details.
    """

    if task_name == 'arrows':
        label_names = ARROW_NAMES

    elif task_name == 'alphabet':
        label_names = ALPHA_NAMES

    else:
        raise ValueError('Incorrect task_name specified: {}'.format(task_name))

    return label_names


def setup_training(model_type, task, data_dirs, save_dir=None):
    learning_params, preproc_params = setup_learning(save_dir)
    model_params = setup_model(model_type, save_dir)
    task_params = setup_task(task)

    # retain data parameters
    if save_dir:
        shutil.copy(os.path.join(data_dirs[0], 'sensor_params.json'), save_dir)

    return learning_params, model_params, preproc_params, task_params
