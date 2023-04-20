"""
python launch_training.py -m simple_cnn -t arrows
"""
import os
import itertools as it

from tactile_data.braille_classification import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_data.utils import make_dir
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.train_model import train_model
from tactile_learning.utils.utils_learning import seed_everything
from tactile_learning.utils.utils_plots import ClassificationPlotter

from braille_classification.learning.setup_training import setup_training, csv_row_to_label
from braille_classification.prediction.evaluate_model import evaluate_model
from braille_classification.utils.label_encoder import LabelEncoder
from braille_classification.utils.parse_args import parse_args


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])
    train_dir_name = '_'.join(filter(None, ["train", *args.data_version]))
    val_dir_name = '_'.join(filter(None, ["val", *args.data_version]))

    for args.task, args.model in it.product(args.tasks, args.models):

        # data dirs - list of directories combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, train_dir_name),
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, val_dir_name),
        ]

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, args.model)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, preproc_params, task_params = setup_training(
            args.model,
            args.task,
            train_data_dirs,
            save_dir
        )

        # set generators and loaders
        train_generator = ImageDataGenerator(
            train_data_dirs,
            csv_row_to_label[args.task],
            **{**preproc_params['image_processing'], **preproc_params['augmentation']}
        )
        val_generator = ImageDataGenerator(
            val_data_dirs,
            csv_row_to_label[args.task],
            **preproc_params['image_processing']
        )

        # create the label encoder/decoder and plotter
        label_encoder = LabelEncoder(task_params['label_names'], args.device)
        error_plotter = ClassificationPlotter(task_params['label_names'], save_dir, final_only=False)

        # create the model
        seed_everything(learning_params['seed'])
        model = create_model(
            in_dim=preproc_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            device=args.device
        )

        train_model(
            prediction_mode='classification',
            model=model,
            label_encoder=label_encoder,
            train_generator=train_generator,
            val_generator=val_generator,
            learning_params=learning_params,
            save_dir=save_dir,
            error_plotter=error_plotter,
            device=args.device
        )

        # run evaluation
        evaluate_model(
            model,
            label_encoder,
            val_generator,
            learning_params,
            error_plotter,
            device=args.device
        )


if __name__ == "__main__":

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['alphabet'],
        models=['simple_cnn_temp'],
        data_version=['temp'],
        device='cuda'
    )

    launch(args)
