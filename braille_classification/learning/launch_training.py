"""
python launch_training.py -m simple_cnn -t arrows
"""
import os

from tactile_data.braille_classification import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_data.utils_data import make_dir
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.simple_train_model import simple_train_model
from tactile_learning.utils.utils_learning import seed_everything
from tactile_learning.supervised.image_generator import ImageDataGenerator

from evaluate_model import evaluate_model
from setup_training import setup_training, setup_parse_args, csv_row_to_label
from utils_learning import LabelEncoder
from utils_plots import ClassErrorPlotter


def launch(
    tasks=['alphabet'],
    models=['simple_cnn'],
    device='cuda'
):
    model_version = ''
    sensor_str = 'tactip_331_25mm'

    tasks, models, device = setup_parse_args(tasks, models, device)

    for task, model_str in zip(tasks, models):

        # data dir - can specify list of directories as these are combined in generator
        train_data_dirs = [
            os.path.join(BASE_DATA_PATH, sensor_str, task, 'train')
        ]
        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, sensor_str, task, 'val')
        ]

        # setup save dir
        save_dir = os.path.join(BASE_MODEL_PATH, sensor_str, task, model_str + model_version)
        make_dir(save_dir)

        # setup parameters
        learning_params, model_params, preproc_params, class_names = setup_training(
            model_str, 
            task, 
            train_data_dirs, 
            save_dir
        )

        # create the encoder/decoder for labels
        label_encoder = LabelEncoder(class_names, device)

        # create plotter of classificaiton
        error_plotter = ClassErrorPlotter(class_names, save_dir, name='error_plot_best.png')

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
            'classification',
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
            model,
            label_encoder,
            val_generator,
            learning_params,
            error_plotter,
            device=device
        )


if __name__ == "__main__":

    # for profiling and debugging slow functions
    # import cProfile
    # import pstats
    # pstats.Stats(
    #     cProfile.Profile().run("launch()")
    # ).sort_stats(
    #     pstats.SortKey.TIME
    # ).print_stats(20)

    launch()
