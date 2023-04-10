import os

from tactile_data.utils_data import save_json_obj

POSE_LABEL_NAMES = ["x", "y", "z", "Rx", "Ry", "Rz"]
SHEAR_LABEL_NAMES = ["dx", "dy", "dz", "dRx", "dRy", "dRz"]
ARROW_LABEL_NAMES = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE']
ALPHABET_LABEL_NAMES = [
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'NONE',
]


def setup_sensor_params(robot, sensor, save_dir=None):

    bbox_dict = {
        'mini': (320-160,    240-160+25, 320+160,    240+160+25),
        'midi': (320-220+10, 240-220-20, 320+220+10, 240+220-20)
    }
    sensor_type = 'midi'  # TODO: Fix hardcoded sensor type

    if robot == 'sim':
        sensor_params = {
            "type": "standard_tactip",
            "image_size": (256, 256),
            "show_tactile": True
        }
    else:
        sensor_params = {
            'type': sensor_type,
            'source': 0,
            'exposure': -7,
            'gray': True,
            'bbox': bbox_dict[sensor_type]
        }

    if save_dir:
        save_json_obj(sensor_params, os.path.join(save_dir, 'sensor_params'))

    return sensor_params


def setup_collect_params(robot, task, save_dir=None):

    pose_lims_dict = {
        'arrows': [(-2.5, -2.5, 1, -5, -5, -10), (2.5, 2.5, 5, 5, 5, 10)],
        'alphabet': [(-1.5, -1.5, 1, -5, -5, -5), (1.5, 1.5, 5, 5, 5, 5)],
    }

    shear_lims_dict = {
        'sim':     [(0, 0, 0, 0, 0, 0),   (0, 0, 0, 0, 0, 0)],
    }

    if task == 'arrows':
        obj_label_names = ARROW_LABEL_NAMES
    elif task == 'alphabet':
        obj_label_names = ALPHABET_LABEL_NAMES

    collect_params = {
        'obj_label_names': obj_label_names,
        'pose_label_names': POSE_LABEL_NAMES,
        'pose_llims': pose_lims_dict[task][0],
        'pose_ulims': pose_lims_dict[task][1],
        'shear_label_names': SHEAR_LABEL_NAMES,
        'shear_llims': shear_lims_dict[robot][0],
        'shear_ulims': shear_lims_dict[robot][1],
        'sample_disk': True,
        'sort': False,
        'seed': 0,
    }

    if robot == 'sim':
        collect_params['sort'] = 'Rz'

    if save_dir:
        save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))

    return collect_params


def setup_env_params(robot, task, save_dir=None):

    # TODO: Combining TCP and workframe is confusing...
    work_frame_dict = {
        'sim_arrows':   [(593, -7, 25, -180, 0, 0),     (0, 0, -85, 0, 0, 90)],
        'sim_alphabet': [(593, -7, 25, -180, 0, 0),     (0, 0, -85, 0, 0, 90)],
    }

    env_params = {
        'robot': robot,
        'stim_name': 'static_keyboard',
        'speed': float("inf"),
        'work_frame': work_frame_dict[robot + '_' + task][0],
        'tcp_pose': work_frame_dict[robot + '_' + task][1],
        'show_gui': True
    }

    if robot == 'sim':
        env_params['stim_pose'] = (600, 0, 0, 0, 0, 0)

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_collect_data(robot, sensor, task, save_dir=None):
    collect_params = setup_collect_params(robot, task, save_dir)
    sensor_params = setup_sensor_params(robot, sensor, save_dir)
    env_params = setup_env_params(robot, task, save_dir)
    return collect_params, env_params, sensor_params


if __name__ == '__main__':
    pass
