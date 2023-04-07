import os

from braille_classification.utils.setup_parse_args import setup_parse_args
from braille_classification.learning.setup_training import csv_row_to_label
from braille_classification.learning.setup_training import setup_learning

from tactile_learning.supervised.image_generator import demo_image_generation

from tactile_data.braille_classification import BASE_DATA_PATH


if __name__ == '__main__':

    robot_str, sensor_str, tasks, _, _ = setup_parse_args()

    learning_params, preproc_params = setup_learning()

    data_dirs = [
        *[os.path.join(BASE_DATA_PATH, robot_str + '_' + sensor_str, task, 'train') for task in tasks],
        *[os.path.join(BASE_DATA_PATH, robot_str + '_' + sensor_str, task, 'val') for task in tasks]
    ]

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        preproc_params['image_processing_params'],
        preproc_params['augmentation_params']
    )
