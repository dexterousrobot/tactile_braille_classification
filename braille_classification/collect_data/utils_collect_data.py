import os
import numpy as np
import pandas as pd

# WARNING: urdf does not follow this pattern exactly due to auto placement of STLs.
# This can introduce some bias in the data due to a slight offset in the placement of the key.
x_increment = -17.5
y_increment = 17.5
obj_poses_dict = {
    'A': [x_increment*0, y_increment*0, 0.0, 0.0, 0.0, 0.0],
    'B': [x_increment*0, y_increment*1, 0.0, 0.0, 0.0, 0.0],
    'C': [x_increment*0, y_increment*2, 0.0, 0.0, 0.0, 0.0],
    'D': [x_increment*0, y_increment*3, 0.0, 0.0, 0.0, 0.0],
    'E': [x_increment*0, y_increment*4, 0.0, 0.0, 0.0, 0.0],
    'F': [x_increment*0, y_increment*5, 0.0, 0.0, 0.0, 0.0],
    'G': [x_increment*0, y_increment*6, 0.0, 0.0, 0.0, 0.0],
    'H': [x_increment*0, y_increment*7, 0.0, 0.0, 0.0, 0.0],
    'I': [x_increment*0, y_increment*8, 0.0, 0.0, 0.0, 0.0],
    'J': [x_increment*0, y_increment*9, 0.0, 0.0, 0.0, 0.0],
    'K': [x_increment*1, y_increment*0, 0.0, 0.0, 0.0, 0.0],
    'L': [x_increment*1, y_increment*1, 0.0, 0.0, 0.0, 0.0],
    'M': [x_increment*1, y_increment*2, 0.0, 0.0, 0.0, 0.0],
    'N': [x_increment*1, y_increment*3, 0.0, 0.0, 0.0, 0.0],
    'O': [x_increment*1, y_increment*4, 0.0, 0.0, 0.0, 0.0],
    'P': [x_increment*1, y_increment*5, 0.0, 0.0, 0.0, 0.0],
    'Q': [x_increment*1, y_increment*6, 0.0, 0.0, 0.0, 0.0],
    'R': [x_increment*1, y_increment*7, 0.0, 0.0, 0.0, 0.0],
    'S': [x_increment*1, y_increment*8, 0.0, 0.0, 0.0, 0.0],
    'T': [x_increment*1, y_increment*9, 0.0, 0.0, 0.0, 0.0],
    'U': [x_increment*2, y_increment*0, 0.0, 0.0, 0.0, 0.0],
    'V': [x_increment*2, y_increment*1, 0.0, 0.0, 0.0, 0.0],
    'W': [x_increment*2, y_increment*2, 0.0, 0.0, 0.0, 0.0],
    'X': [x_increment*2, y_increment*3, 0.0, 0.0, 0.0, 0.0],
    'Y': [x_increment*2, y_increment*4, 0.0, 0.0, 0.0, 0.0],
    'Z': [x_increment*2, y_increment*5, 0.0, 0.0, 0.0, 0.0],
    'UP': [x_increment*2, y_increment*6, 0.0, 0.0, 0.0, 0.0],
    'DOWN': [x_increment*2, y_increment*7, 0.0, 0.0, 0.0, 0.0],
    'LEFT': [x_increment*2, y_increment*8, 0.0, 0.0, 0.0, 0.0],
    'RIGHT': [x_increment*2, y_increment*9, 0.0, 0.0, 0.0, 0.0],
    'SPACE': [x_increment*3, y_increment*3, 0.0, 0.0, 0.0, 0.0],
    'NONE': [x_increment*3, y_increment*8, -10.0, 0.0, 0.0, 0.0],
}


def setup_target_df(
    collect_params,
    num_poses=100,
    save_dir=None,
):

    pose_lims = [collect_params['pose_llims'], collect_params['pose_ulims']]
    shear_lims = [collect_params['shear_llims'], collect_params['shear_ulims']]
    sample_disk = collect_params.get('sample_disk', False)
    obj_label_names = collect_params['obj_label_names']
    sample_disk = collect_params.get('sample_disk', False)

    # generate random poses
    obj_poses = [obj_poses_dict[label_name] for label_name in obj_label_names]
    np.random.seed(collect_params['seed'])  # make deterministic
    poses = sample_poses(*pose_lims, num_poses, sample_disk)
    shears = sample_poses(*shear_lims, num_poses, sample_disk)

    # generate and save target data
    target_df = pd.DataFrame(
        columns=[
            "sensor_image",
            "obj_id",
            "obj_lbl",
            "obj_pose",
            "pose_id",
            *collect_params['pose_label_names'],
            *collect_params['shear_label_names']
        ]
    )

    # populate dataframe
    for i in range(num_poses * len(obj_poses)):
        image_name = f"image_{i+1}.png"
        obj_id = int(i / num_poses)
        pose_id = int(i % num_poses)

        pose = poses[pose_id, :]
        shear = shears[pose_id, :]
        target_df.loc[i] = np.hstack((
            (image_name, obj_id+1, obj_label_names[obj_id], obj_poses[obj_id], pose_id),
            pose, shear
        ))

    # save to file
    if save_dir:
        target_file = os.path.join(save_dir, "targets.csv")
        target_df.to_csv(target_file, index=False)

    return target_df


def random_linear(num_samples, x_max):
    """Return uniform random sample over a 1D segment [-x_max, x_max]."""
    return -x_max + 2 * x_max * np.random.rand(num_samples)


def sample_poses(llims, ulims, num_samples, sample_disk):

    poses_mid = (np.array(ulims) + llims) / 2
    poses_max = ulims - poses_mid

    # default linear sampling on all components
    samples = [random_linear(num_samples, x_max) for x_max in poses_max]
    poses = np.array(samples).T
    poses += poses_mid

    return poses
