"""
python evaluate_model.py -m simple_cnn -t arrows
"""
import os
import numpy as np
from torch.autograd import Variable
import torch

from tactile_data.braille_classification import BASE_DATA_PATH, BASE_MODEL_PATH
from tactile_data.utils_data import load_json_obj
from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.image_generator import ImageDataGenerator

from setup_training import setup_parse_args, setup_task, csv_row_to_label
from utils_learning import LabelEncoder
from utils_plots import ClassErrorPlotter

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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
    len_gen = generator.__len__()
    pred_arr = np.zeros([len_gen*learning_params['batch_size']])
    targ_arr = np.zeros([len_gen*learning_params['batch_size']])
    index = 0

    for batch in loader:

        # get inputs
        inputs, targ_dict = batch['images'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)
        pred_dict = label_encoder.decode_label(outputs)

        # count correct for accuracy metric
        for i in range(len(pred_dict['id'])):
            pred_arr[index] = pred_dict['id'][i]
            targ_arr[index] = targ_dict['id'][i]
            index += 1

    # plot full error graph
    error_plotter.final_plot(
        pred_arr, targ_arr
    )


if __name__ == "__main__":

    tasks, models, device = setup_parse_args(
        tasks=['alphabet'],
        models=['simple_cnn'],
        device='cuda'
    )

    model_version = ''
    sensor = 'tactip_331_25mm'

    # test the trained networks
    for task, model_type in zip(tasks, models):

        val_data_dirs = [
            os.path.join(BASE_DATA_PATH, sensor, task, 'val')
        ]

        # set save dir
        save_dir = os.path.join(BASE_MODEL_PATH, sensor, task, model_type + model_version)

        # setup parameters
        learning_params = load_json_obj(os.path.join(save_dir, 'learning_params'))
        model_params = load_json_obj(os.path.join(save_dir, 'model_params'))
        preproc_params = load_json_obj(os.path.join(save_dir, 'preproc_params'))

        # create the encoder/decoder for labels
        class_names = setup_task(task)
        label_encoder = LabelEncoder(class_names, device)

        # create plotter of classificaiton
        error_plotter = ClassErrorPlotter(class_names, save_dir, name='error_plot_best.png')

        # create the model
        model = create_model(
            in_dim=preproc_params['image_processing']['dims'],
            in_channels=1,
            out_dim=label_encoder.out_dim,
            model_params=model_params,
            saved_model_dir=save_dir,
            device=device
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
            device=device
        )
