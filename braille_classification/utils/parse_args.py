import argparse


def parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        models=['simple_cnn'],
        data_version=[],
        device='cuda'
):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-r', '--robot',
        type=str,
        help="Choose robot from ['sim', 'ur', 'mg400', 'cr']",
        default=robot
    )
    parser.add_argument(
        '-s', '--sensor',
        type=str,
        help="Choose sensor from ['tactip', 'tactip_331_25mm']",
        default=sensor
    )
    parser.add_argument(
        '-t', '--tasks',
        nargs='+',
        help="Choose tasks from ['alphabet', 'arrows']",
        default=tasks
    )
    parser.add_argument(
        '-m', '--models',
        nargs='+',
        help="Choose models from ['simple_cnn', 'posenet_cnn', 'nature_cnn', 'resnet', 'vit']",
        default=models
    )
    parser.add_argument(
        '-dv', '--data_version',
        type=str,
        help="Choose version.",
        default=data_version
    )
    parser.add_argument(
        '-d', '--device',
        type=str,
        help="Choose device from ['cpu', 'cuda']",
        default=device
    )

    return parser.parse_args()
