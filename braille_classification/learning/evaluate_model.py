"""
python evaluate_model.py -t alphabet
python evaluate_model.py -t arrows
python evaluate_model.py -t alphabet arrows
"""
import os
import argparse
import numpy as np
from torch.autograd import Variable
import torch

from sklearn.metrics import confusion_matrix

from tactile_learning.supervised.models import create_model
from tactile_learning.supervised.image_generator import ImageDataGenerator
from tactile_learning.utils.utils_learning import load_json_obj

from braille_classification.learning.setup_learning import setup_task
from braille_classification.learning.utils_plots import plot_confusion_matrix
from braille_classification.learning.utils_learning import LabelEncoder
from braille_classification.learning.utils_learning import csv_row_to_label

from braille_classification import BASE_DATA_PATH
from braille_classification import BASE_MODEL_PATH

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def evaluate_model(
    task,
    model,
    label_encoder,
    data_dirs,
    learning_params,
    image_processing_params,
    save_dir,
    device='cpu'
):
    # set generators and loaders
    generator = ImageDataGenerator(
        data_dirs=data_dirs,
        csv_row_to_label=csv_row_to_label,
        **image_processing_params
    )

    loader = torch.utils.data.DataLoader(
        generator,
        batch_size=learning_params['batch_size'],
        shuffle=learning_params['shuffle'],
        num_workers=learning_params['n_cpu']
    )

    # complete dateframe of predictions and targets
    len_gen = generator.__len__()
    predictions_array = np.zeros([len_gen*learning_params['batch_size']])
    ground_truth_array = np.zeros([len_gen*learning_params['batch_size']])
    counter = 0

    for batch in loader:

        # get inputs
        inputs, labels_dict = batch['images'], batch['labels']

        # wrap them in a Variable object
        inputs = Variable(inputs).float().to(device)

        # forward pass
        outputs = model(inputs)
        predictions_dict = label_encoder.decode_label(outputs)

        # count correct for accuracy metric
        for i in range(len(predictions_dict['id'])):
            prediction = predictions_dict['id'][i]
            ground_truth = labels_dict['id'][i]
            predictions_array[counter] = prediction
            ground_truth_array[counter] = ground_truth
            counter += 1

    # reset indices to be 0 -> test set size
    cnf_matrix = confusion_matrix(ground_truth_array, predictions_array)

    # plot full error graph
    plot_confusion_matrix(
        cnf_matrix,
        classes=label_encoder.target_label_names,
        normalize=True,
        title='Normalized Confusion matrix',
        save_dir=save_dir,
        name='cnf_mtrx.png',
        show_plot=True
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose task from ['alphabet', 'arrows'].",
        default=['arrows']
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose model from ['simple_cnn', 'nature_cnn', 'resnet', 'vit'].",
        default=['simple_cnn']
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda'].",
        default='cuda'
    )

    # parse arguments
    args = parser.parse_args()
    tasks = args.tasks
    models = args.models
    device = args.device

    # test the trained networks
    for model_type in models:
        for task in tasks:

            # task specific parameters
            out_dim, label_names = setup_task(task)

            # set save dir
            save_dir = os.path.join(BASE_MODEL_PATH, task, model_type)

            # setup parameters
            network_params = load_json_obj(os.path.join(save_dir, 'model_params'))
            learning_params = load_json_obj(os.path.join(save_dir, 'learning_params'))
            image_processing_params = load_json_obj(os.path.join(save_dir, 'image_processing_params'))

            # create the model
            model = create_model(
                image_processing_params['dims'],
                out_dim,
                network_params,
                saved_model_dir=save_dir,  # loads weights of best_model.pth
                device=device
            )
            model.eval()

            val_data_dirs = [
                os.path.join(BASE_DATA_PATH, task, 'val')
            ]

            # create the encoder/decoder for labels
            label_encoder = LabelEncoder(out_dim, label_names, device)

            evaluate_model(
                task,
                model,
                label_encoder,
                val_data_dirs,
                learning_params,
                image_processing_params,
                save_dir,
                device=device
            )
