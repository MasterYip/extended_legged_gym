from legged_gym.envs.el_4090.spider_nomal.el4090_high_stand_trot_config import (
    El4090HighStandTrotCfg,
    El4090HighStandTrotCfgPPO,
)


class El4090HighStandTrotCollectCfg(El4090HighStandTrotCfg):
    class collect:
        task_vec = [1.0, 4.0, 0.0]
    class commands(El4090HighStandTrotCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = True

        class ranges(El4090HighStandTrotCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]   # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]    # min max [rad/s]
            heading = [-3.14, 3.14]

class El4090HighStandTrotCollectCfgPPO(El4090HighStandTrotCfgPPO):
    class runner(El4090HighStandTrotCfgPPO.runner):
        experiment_name = "el4090_high_stand_trot_collect"
