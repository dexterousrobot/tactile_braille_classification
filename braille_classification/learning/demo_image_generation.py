import os

from tactile_data.braille_classification import BASE_DATA_PATH
from tactile_learning.supervised.image_generator import demo_image_generation

from braille_classification.learning.setup_training import setup_learning, csv_row_to_label
from braille_classification.utils.parse_args import parse_args


if __name__ == '__main__':

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        data_version=['temp']
    )

    output_dir = '_'.join([args.robot, args.sensor])
    train_dir_name = '_'.join(filter(None, ["train", *args.data_version]))
    val_dir_name = '_'.join(filter(None, ["val", *args.data_version]))

    learning_params, preproc_params = setup_learning()

    for args.task in args.tasks:

        data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, train_dir_name),
            os.path.join(BASE_DATA_PATH, output_dir, args.task, val_dir_name)
        ]

        demo_image_generation(
            data_dirs,
            csv_row_to_label[args.task],
            learning_params,
            preproc_params['image_processing'],
            preproc_params['augmentation']
        )
