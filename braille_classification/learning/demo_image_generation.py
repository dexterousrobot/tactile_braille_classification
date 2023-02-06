"""
python demo_image_generation.py -t arrows
python demo_image_generation.py -t alphabet
python demo_image_generation.py -t arrows alphabet
"""

import os
import argparse

from braille_classification import BASE_DATA_PATH
from braille_classification.learning.utils_learning import csv_row_to_label
from braille_classification.learning.setup_learning import setup_learning

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

    learning_params, image_processing_params, augmentation_params = setup_learning()

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
