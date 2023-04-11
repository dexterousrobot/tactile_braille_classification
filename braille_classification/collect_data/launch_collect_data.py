"""
python launch_collect_data.py -r sim -s tactip -t arrows
"""
import os
import numpy as np

from tactile_data.braille_classification import BASE_DATA_PATH
from tactile_data.utils_data import make_dir

from braille_classification.collect_data.setup_collect_data import setup_collect_data
from braille_classification.collect_data.utils_collect_data import setup_target_df
from braille_classification.utils.parse_args import parse_args
from braille_classification.utils.setup_embodiment import setup_embodiment


def collect_data(
    robot,
    sensor,
    targets_df,
    image_dir,
    collect_params,
):

    # start 50mm above workframe origin with zero joint 6
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints([*robot.joint_angles[:-1], 0])

    # collect reference image
    image_outfile = os.path.join(image_dir, 'image_0.png')
    sensor.process(image_outfile)

    # clear object by 10mm; use as reset pose
    clearance = (0, 0, 10, 0, 0, 0)
    robot.move_linear(np.zeros(6) - clearance)
    joint_angles = robot.joint_angles

    # ==== data collection loop ====
    for i, row in targets_df.iterrows():
        image_name = row.loc["sensor_image"]
        obj_pose = row.loc["obj_pose"]
        pose = row.loc[collect_params['pose_label_names']].values.astype(float)
        shear = row.loc[collect_params['shear_label_names']].values.astype(float)
        full_pose = obj_pose + pose

        # report
        with np.printoptions(precision=1, suppress=True):
            print(f"Collecting for pose {i+1}: pose{pose}, shear{shear}")

        # move to above new pose (avoid changing pose in contact with object)
        robot.move_linear(full_pose + shear - clearance)

        # move down to offset pose
        robot.move_linear(full_pose + shear)

        # move to target pose inducing shear
        robot.move_linear(full_pose)

        # collect and process tactile image
        image_outfile = os.path.join(image_dir, image_name)
        sensor.process(image_outfile)

        # move above the target pose
        robot.move_linear(full_pose - clearance)

        # if sorted, don't move to reset position
        if not collect_params['sort']:
            robot.move_joints(joint_angles)

    # finish 50mm above workframe origin then zero last joint
    robot.move_linear((0, 0, -50, 0, 0, 0))
    robot.move_joints((*robot.joint_angles[:-1], 0))
    robot.close()


def launch():

    args = parse_args(
        robot='sim',
        sensor='tactip',
        tasks=['arrows'],
        version=['']
    )

    data_params = {
        # 'data': 200,  # per key
        'train': 200,  # per key
        'val': 50,  # per key
    }

    for args.task in args.tasks:
        for data_dir_name, num_samples in data_params.items():

            data_dir_name = '_'.join(filter(None, [data_dir_name, *args.version]))
            output_dir = '_'.join([args.robot, args.sensor])

            # setup save dir
            save_dir = os.path.join(BASE_DATA_PATH, output_dir, args.task, data_dir_name)
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
            target_df = setup_target_df(
                collect_params,
                num_samples,
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


if __name__ == "__main__":
    launch()
