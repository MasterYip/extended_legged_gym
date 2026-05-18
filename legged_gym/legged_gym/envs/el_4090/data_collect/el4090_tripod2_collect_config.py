from legged_gym.envs.el_4090.spider_nomal.el4090_tripod2_config import (
    El4090Tripod2Cfg,
    El4090Tripod2CfgPPO,
)


class El4090Tripod2CollectCfg(El4090Tripod2Cfg):
    class collect:
        task_vec = [1.0, 0.0, 0.0]
    class commands(El4090Tripod2Cfg.commands):
        curriculum = True
        max_curriculum = 3.0
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = True

        class ranges(El4090Tripod2Cfg.commands.ranges):
            lin_vel_x = [-3, 3]  # min max [m/s]
            lin_vel_y = [-1.5, 1.5]   # min max [m/s]
            ang_vel_yaw = [-2.0, 2.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class El4090Tripod2CollectCfgPPO(El4090Tripod2CfgPPO):
    class runner(El4090Tripod2CfgPPO.runner):
        experiment_name = "el4090_tripod2_collect"
