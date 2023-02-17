import os
import shutil

from braille_classification.learning.setup_learning import parse_args
from braille_classification.learning.setup_learning import setup_model
from braille_classification.learning.setup_learning import setup_learning
from braille_classification.learning.setup_learning import setup_task

from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.simple_train_model import simple_train_model
from tactile_learning.utils.utils_learning import seed_everything, make_dir
from tactile_learning.supervised.image_generator import ImageDataGenerator

from braille_classification.learning.evaluate_model import evaluate_model
from braille_classification.learning.utils_learning import LabelEncoder
from braille_classification.learning.utils_learning import csv_row_to_label

from braille_classification import BASE_DATA_PATH
from braille_classification import BASE_MODEL_PATH


def launch():

    # parse arguments
    args = parse_args()
    tasks = args.tasks
    models = args.models
    device = args.device

    # for setting network and learning params
    prediction_mode = 'classification'

    for task in tasks:
        for model_type in models:

            # task specific parameters
            out_dim, label_names = setup_task(task)

            # setup save dir
            save_dir = os.path.join(BASE_MODEL_PATH, task, model_type)
            make_dir(save_dir)

            # setup parameters
            model_params = setup_model(model_type, save_dir)
            learning_params, image_processing_params, augmentation_params = setup_learning(save_dir)

            # keep record of sensor params
            shutil.copy(os.path.join(BASE_DATA_PATH, task, 'train', 'sensor_params.json'), save_dir)

            # create the model
            seed_everything(learning_params['seed'])
            model = create_model(
                in_dim=image_processing_params['dims'],
                in_channels=1,
                out_dim=out_dim,
                model_params=model_params,
                device=device
            )

            # data dir - can specify list of directories as these are combined in generator
            train_data_dirs = [
                os.path.join(BASE_DATA_PATH, task, 'train')
            ]
            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, task, 'val')
            ]

            # set generators and loaders
            train_generator = ImageDataGenerator(
                data_dirs=train_data_dirs,
                csv_row_to_label=csv_row_to_label,
                **{**image_processing_params, **augmentation_params}
            )
            val_generator = ImageDataGenerator(
                data_dirs=val_data_dirs,
                csv_row_to_label=csv_row_to_label,
                **image_processing_params
            )

            # create the encoder/decoder for labels
            label_encoder = LabelEncoder(out_dim, label_names, device)

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

    # for profiling and debugging slow functions
    # import cProfile
    # import pstats
    # pstats.Stats(
    #     cProfile.Profile().run("launch()")
    # ).sort_stats(
    #     pstats.SortKey.TIME
    # ).print_stats(20)

    launch()
