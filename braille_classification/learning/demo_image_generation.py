import os

from braille_classification.utils.parse_args import parse_args
from braille_classification.learning.setup_training import csv_row_to_label
from braille_classification.learning.setup_training import setup_learning

from tactile_learning.supervised.image_generator import demo_image_generation

from tactile_data.braille_classification import BASE_DATA_PATH


if __name__ == '__main__':

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        version=['']
    )

    output_dir = '_'.join([args.robot, args.sensor])
    train_dir_name = '_'.join(filter(None, ["train", *args.version]))
    val_dir_name = '_'.join(filter(None, ["val", *args.version]))

    learning_params, preproc_params = setup_learning()

    data_dirs = [
        *[os.path.join(BASE_DATA_PATH, output_dir, task, train_dir_name) for task in args.tasks],
        *[os.path.join(BASE_DATA_PATH, output_dir, task, val_dir_name) for task in args.tasks]
    ]

    demo_image_generation(
        data_dirs,
        csv_row_to_label,
        learning_params,
        preproc_params['image_processing'],
        preproc_params['augmentation']
    )
