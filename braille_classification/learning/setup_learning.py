import os

from tactile_learning.utils.utils_learning import save_json_obj
from braille_classification.learning.utils_learning import ARROW_NAMES
from braille_classification.learning.utils_learning import ALPHA_NAMES


def setup_learning(save_dir=None):

    # Parameters
    learning_params = {
        'seed': 42,
        'batch_size': 64,
        'epochs': 100,
        'lr': 1e-4,
        'lr_factor': 0.5,
        'lr_patience': 10,
        'adam_decay': 1e-6,
        'adam_b1': 0.9,
        'adam_b2': 0.999,
        'shuffle': True,
        'n_cpu': 8,
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

    if save_dir:
        save_json_obj(learning_params, os.path.join(save_dir, 'learning_params'))
        save_json_obj(image_processing_params, os.path.join(save_dir, 'image_processing_params'))
        save_json_obj(augmentation_params, os.path.join(save_dir, 'augmentation_params'))

    return learning_params, image_processing_params, augmentation_params


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
        out_dim = len(ARROW_NAMES)
        label_names = ARROW_NAMES

    elif task_name == 'alphabet':
        out_dim = len(ALPHA_NAMES)
        label_names = ALPHA_NAMES

    else:
        raise ValueError('Incorrect task_name specified: {}'.format(task_name))

    return out_dim, label_names
