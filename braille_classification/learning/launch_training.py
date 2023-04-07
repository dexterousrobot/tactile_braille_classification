import os

from braille_classification.learning.setup_training import setup_training
from braille_classification.learning.setup_training import csv_row_to_label
from braille_classification.utils.setup_parse_args import setup_parse_args

from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.simple_train_model import simple_train_model
from tactile_learning.utils.utils_learning import seed_everything, make_dir
from tactile_learning.supervised.image_generator import ImageDataGenerator

from braille_classification.learning.evaluate_model import evaluate_model
from braille_classification.learning.utils_learning import LabelEncoder

from tactile_data.braille_classification import BASE_DATA_PATH
from tactile_data.braille_classification import BASE_MODEL_PATH


def launch(
    robot='sim',
    sensor='tactip',
    tasks=['arrows'],
    models=['simple_cnn'],
    device='cuda'
):

    model_version = ''

    # parse arguments
    robot_str, sensor_str, tasks, models, device = setup_parse_args(robot, sensor, tasks, models, device)

    # for setting network and learning params
    prediction_mode = 'classification'
    robot_sensor_str = robot_str + '_' + sensor_str

    for task, model_str in zip(tasks, models):

        # data dirs - list of directories combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, robot_sensor_str, task, 'train'),
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, robot_sensor_str, task, 'val'),
        ]

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, robot_sensor_str, task, model_str + model_version)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, preproc_params, task_params = setup_training(
            model_str,
            task,
            train_data_dirs,
            save_dir
        )

        # create the encoder/decoder for labels
        label_encoder = LabelEncoder(task_params['out_dim'], task_params['label_names'], device)

        # create the model
        seed_everything(learning_params['seed'])
        model = create_model(
            in_dim=preproc_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            device=device
        )

        # set generators and loaders
        train_generator = ImageDataGenerator(
            train_data_dirs,
            csv_row_to_label,
            **{**preproc_params['image_processing'], **preproc_params['augmentation']}
        )
        val_generator = ImageDataGenerator(
            val_data_dirs,
            csv_row_to_label,
            **preproc_params['image_processing']
        )

        # run training
        simple_train_model(
            prediction_mode,
            model,
            label_encoder,
            train_generator,
            val_generator,
            learning_params,
            save_dir,
            device=device
        )

        # run evaluation
        evaluate_model(
            task,
            model,
            label_encoder,
            val_generator,
            learning_params,
            save_dir,
            device=device
        )


if __name__ == "__main__":
    launch()
