"""
python launch_training.py -m simple_cnn -t arrows
"""
import os
import itertools as it

from tactile_data.braille_classification import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_data.utils_data import make_dir
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.supervised.models import create_model
# from tactile_learning.supervised.train_model_w_metrics import train_model_w_metrics
from tactile_learning.supervised.simple_train_model import simple_train_model
from tactile_learning.utils.utils_learning import seed_everything

from braille_classification.learning.evaluate_model import evaluate_model
from braille_classification.learning.setup_training import setup_training, csv_row_to_label
from braille_classification.learning.utils_learning import LabelEncoder
from braille_classification.learning.utils_plots import ClassErrorPlotter
from braille_classification.utils.parse_args import parse_args


def launch():

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        models=['simple_cnn'],
        version=[''],
        device='cuda'
    )

    output_dir = '_'.join([args.robot, args.sensor])
    train_dir_name = '_'.join(filter(None, ["train", *args.version]))
    val_dir_name = '_'.join(filter(None, ["val", *args.version]))

    for args.task, args.model in it.product(args.tasks, args.models):

        model_dir_name = '_'.join([args.model, *args.version])

        # data dirs - list of directories combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, train_dir_name),
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, val_dir_name),
        ]

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, preproc_params, task_params = setup_training(
            args.model,
            args.task,
            train_data_dirs,
            save_dir
        )

        # create the encoder/decoder for labels
        label_encoder = LabelEncoder(task_params['label_names'], args.device)

        # create the model
        seed_everything(learning_params['seed'])
        model = create_model(
            in_dim=preproc_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            device=args.device
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

        simple_train_model(
            'classification',
            model,
            label_encoder,
            train_generator,
            val_generator,
            learning_params,
            save_dir,
            device=args.device
        )

        # create plotter of classification errors
        error_plotter = ClassErrorPlotter(
            task_params['label_names'],
            save_dir,
            name='error_plot.png',
            # plot_during_training=True
        )

        # train_model_w_metrics(
        #     model,
        #     label_encoder,
        #     train_generator,
        #     val_generator,
        #     learning_params,
        #     save_dir,
        #     error_plotter=error_plotter,
        #     calculate_train_metrics=False,
        #     device=args.device
        # )

        # perform a final evaluation using the last model
        error_plotter.name = 'error_plot_final.png'

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
    launch()
