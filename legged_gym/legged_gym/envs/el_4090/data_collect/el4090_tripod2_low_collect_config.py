from legged_gym.envs.el_4090.spider_nomal.el4090_tripod2_low_config import (
    El4090Tripod2LowCfg,
    El4090Tripod2LowCfgPPO,
)


class El4090Tripod2LowCollectCfg(El4090Tripod2LowCfg):
    class collect:
        task_vec = [1.0, 0.0, 0.0]
    class commands(El4090Tripod2LowCfg.commands):
        curriculum = True
        max_curriculum = 3.0
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = False

        class ranges(El4090Tripod2LowCfg.commands.ranges):
            lin_vel_x = [-3.0, 3.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.2, 1.2]    # min max [rad/s]
            heading = [-3.14, 3.14]


class El4090Tripod2LowCollectCfgPPO(El4090Tripod2LowCfgPPO):
    class runner(El4090Tripod2LowCfgPPO.runner):
        experiment_name = "el4090_tripod2_low_collect"
