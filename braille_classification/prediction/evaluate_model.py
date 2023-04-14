"""
python evaluate_model.py -m simple_cnn -t arrows
"""
import os
import itertools as it
import pandas as pd
from torch.autograd import Variable
import torch

from tactile_data.braille_classification import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_data.utils import load_json_obj
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.utils.utils_plots import ClassificationPlotter

from braille_classification.learning.setup_training import setup_task, csv_row_to_label
from braille_classification.utils.label_encoder import LabelEncoder
from braille_classification.utils.parse_args import parse_args


def evaluate_model(
    model,
    label_encoder,
    generator,
    learning_params,
    error_plotter,
    device='cpu'
):

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    pred_df = pd.DataFrame()
    targ_df = pd.DataFrame()

    for batch in loader:

        # get inputs
        inputs, labels_dict = batch['images'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)

        # decode predictions into label
        predictions_dict = label_encoder.decode_label(outputs)

        # append predictions and labels to dataframes
        batch_pred_df = pd.DataFrame.from_dict(predictions_dict)
        batch_targ_df = pd.DataFrame.from_dict(labels_dict)
        pred_df = pd.concat([pred_df, batch_pred_df])
        targ_df = pd.concat([targ_df, batch_targ_df])

    # reset indices to be 0 -> test set size
    pred_df = pred_df.reset_index(drop=True).fillna(0.0)
    targ_df = targ_df.reset_index(drop=True).fillna(0.0)
    metrics = label_encoder.calc_metrics(pred_df, targ_df)

    # plot full error graph
    error_plotter.final_plot(
        pred_df, targ_df, metrics
    )


if __name__ == "__main__":

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        models=['simple_cnn'],
        version=[''],
        device='cuda'
    )

    model_version = ''

    # test the trained networks
    for args.task, args.model in it.product(args.tasks, args.models):

        output_dir = '_'.join([args.robot, args.sensor])
        val_dir_name = '_'.join(filter(None, ["val", *args.version]))
        model_dir_name = '_'.join([args.model, *args.version])

        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, output_dir, args.task, val_dir_name)
        ]

        # set save dir
        model_dir = os.path.join(BASE_MODEL_PATH, output_dir, args.task, model_dir_name)

        # setup parameters
        learning_params = load_json_obj(os.path.join(model_dir, 'learning_params'))
        model_params = load_json_obj(os.path.join(model_dir, 'model_params'))
        preproc_params = load_json_obj(os.path.join(model_dir, 'preproc_params'))

        # create the encoder/decoder for labels
        task_params = setup_task(args.task)
        label_encoder = LabelEncoder(task_params['label_names'], args.device)

        # create plotter of classificaiton
        error_plotter = ClassificationPlotter(task_params['label_names'], model_dir, name='error_plot_best.png')

        # create the model
        model = create_model(
            in_dim=preproc_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            saved_model_dir=model_dir,
            device=args.device
        )
        model.eval()

        val_generator = ImageDataGenerator(
            val_data_dirs,
            csv_row_to_label,
            **preproc_params['image_processing']
        )

        evaluate_model(
            model,
            label_encoder,
            val_generator,
            learning_params,
            error_plotter,
            device=args.device
        )
