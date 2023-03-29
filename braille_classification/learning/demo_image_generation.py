import os

from tactile_data.braille_classification import BASE_DATA_PATH
from braille_classification.learning.setup_training import setup_parse_args, setup_learning
from braille_classification.learning.utils_learning import csv_row_to_label
from tactile_learning.supervised.image_generator import demo_image_generation


if __name__ == '__main__':

    tasks, models, device = setup_parse_args(
        tasks=['arrows'],
        models=['simple_cnn'],
        device='cuda'
    )

    learning_params, preproc_params = setup_learning()

    data_dirs = [
        *[os.path.join(BASE_DATA_PATH, task, 'train') for task in tasks],
        *[os.path.join(BASE_DATA_PATH, task, 'val') for task in tasks]
    ]

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        preproc_params['image_processing_params'],
        preproc_params['augmentation_params']
    )
