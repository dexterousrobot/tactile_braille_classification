"""
python launch_collect_data.py -r sim -s tactip -t arrows
"""
import os

from tactile_data.braille_classification import BASE_DATA_PATH
from tactile_data.collect_data.collect_data import collect_data
from tactile_data.collect_data.process_data import process_data, split_data
from tactile_data.collect_data.setup_embodiment import setup_embodiment
from tactile_data.collect_data.setup_targets import setup_targets
from tactile_data.utils import make_dir

from braille_classification.collect_data.setup_collect_data import setup_collect_data
from braille_classification.utils.parse_args import parse_args


def launch(args):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
        for args.data_dir, args.data_sample_num in zip(args.data_dirs, args.data_sample_nums):

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.task, args.data_dir)
            image_dir = os.path.join(save_dir, "images")
            make_dir(save_dir)
            make_dir(image_dir)

            # setup parameters
            collect_params, env_params, sensor_params = setup_collect_data(
                args.robot,
                args.sensor,
                args.task,
                save_dir
            )

            # setup embodiment
            robot, sensor = setup_embodiment(
                env_params,
                sensor_params
            )

            # setup targets to collect
            target_df = setup_targets(
                collect_params,
                args.data_sample_num,
                save_dir
            )

            # collect
            collect_data(
                robot,
                sensor,
                target_df,
                image_dir,
                collect_params
            )


def process(args, process_params, split=None):

    output_dir = '_'.join([args.robot, args.sensor])

    for args.task in args.tasks:
            path = os.path.join(BASE_DATA_PATH, output_dir, args.task)

            data_dirs = split_data(path, args.data_dirs, split)
            process_data(path, data_dirs, process_params)


if __name__ == "__main__":

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        data_dirs = ['train_temp', 'val_temp'],
        data_sample_nums = [80, 20] # per key
    )

    process_params = {
        "bbox": (12, 12, 240, 240)  # sim (12, 12, 240, 240)
    }

    launch(args)
    process(args, process_params)#, split=0.8)
