import os

from tactile_data.utils import save_json_obj

KEY_LABEL_NAMES = [
    'UP', 'DOWN', 'LEFT', 'RIGHT', 'NONE',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'SPACE', 'NONE'
]


def setup_sensor_params(robot, sensor, save_dir=None):

    bbox_dict = {
        'mini': (320-160,    240-160+25, 320+160,    240+160+25),
        'midi': (320-220+10, 240-220-20, 320+220+10, 240+220-20)
    }
    sensor_type = 'midi'  # TODO: Fix hardcoded sensor type

    if 'sim' in robot:
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
        'arrows': [(-2.5, -2.5, 3, 0, 0, -10), (2.5, 2.5, 5, 0, 0, 10)],
        'alphabet': [(-2.5, -2.5, 3, 0, 0, -10), (2.5, 2.5, 5, 0, 0, 10)],
    }

    # WARNING: urdf does not follow this pattern exactly due to auto placement of STLs.
    # This can introduce some bias in the data due to a slight offset in the placement of the key.

    if task == 'arrows':
        object_poses_dict = {
            label: (-17.5*2, 17.5*(6+i%10), 0, 0, 0, 0) 
                for i, label in enumerate(KEY_LABEL_NAMES[:5])
        }

    if task == 'alphabet':
        object_poses_dict = {
            label: (-17.5*(i//10), 17.5*(i%10), 0, 0, 0, 0) 
                for i, label in enumerate(KEY_LABEL_NAMES[5:])
        }
        object_poses_dict['SPACE'] = (-17.5*3, 17.5*3, 0, 0, 0, 0)
        
    object_poses_dict['NONE'] = (-17.5*3, 17.5*8, -10, 0, 0, 0)

    collect_params = {
        'object_poses': object_poses_dict,
        'pose_llims': pose_lims_dict[task][0],
        'pose_ulims': pose_lims_dict[task][1],
        'sample_disk': True,
        'sort': True,
        'seed': 0
    }

    if save_dir:
        save_json_obj(collect_params, os.path.join(save_dir, 'collect_params'))

    return collect_params


def setup_env_params(robot, save_dir=None):

    work_frame_dict = {
        'sim':   (593, -7, 25, -180, 0, 0),
    }

    tcp_pose_dict = {
        'cr':    (0, 0, -70, 0, 0, 0),
        'mg400': (0, 0, -50, 0, 0, 0),
        'sim':   (0, 0, -85, 0, 0, 90),
    } ## SHOULD BE ROBOT + SENSOR 

    env_params = {
        'robot': robot,
        'stim_name': 'static_keyboard',
        'work_frame': work_frame_dict[robot],
        'tcp_pose': tcp_pose_dict[robot],
        'show_gui': True
    }

    if 'sim' in robot:
        env_params['speed'] = float('inf')
        env_params['stim_pose'] = (600, 0, 0, 0, 0, 0)

    if save_dir:
        save_json_obj(env_params, os.path.join(save_dir, 'env_params'))

    return env_params


def setup_collect_data(robot, sensor, task, save_dir=None):
    collect_params = setup_collect_params(robot, task, save_dir)
    sensor_params = setup_sensor_params(robot, sensor, save_dir)
    env_params = setup_env_params(robot, save_dir)
    return collect_params, env_params, sensor_params


if __name__ == '__main__':
    pass
