"""
python demo_image_generation.py -t arrows
python demo_image_generation.py -t alphabet
python demo_image_generation.py -t arrows alphabet
"""

import os
import argparse

from braille_classification import BASE_DATA_PATH
from braille_classification.learning.utils_learning import csv_row_to_label

from tactile_learning.supervised.image_generator import demo_image_generation


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['arrows', 'alphabet'].",
        default=['arrows']
    )
    args = parser.parse_args()
    tasks = args.tasks

    learning_params = {
        'batch_size':  8,
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
        'rshift': None,  # (0.015, 0.015),
        'rzoom':   None,
        'brightlims': None,
        'noise_var': None,
    }

    data_dirs = [
        os.path.join(BASE_DATA_PATH, task, 'train') for task in tasks
        # os.path.join(BASE_DATA_PATH, task, 'val') for task in tasks
    ]

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        image_processing_params,
        augmentation_params
    )
