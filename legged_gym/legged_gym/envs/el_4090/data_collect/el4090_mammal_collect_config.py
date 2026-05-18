from legged_gym.envs.el_4090.spider_nomal.el4090_mammal_config import (
    El4090MammalCfg,
    El4090MammalCfgPPO,
)


class El4090MammalCollectCfg(El4090MammalCfg):
    class collect:
        task_vec = [1.0, 5.0, 0.0]
    class commands(El4090MammalCfg.commands):
        curriculum = True
        max_curriculum = 1.5
        # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 4.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        small_command_radio = True

        class ranges(El4090MammalCfg.commands.ranges):
            lin_vel_x = [-1.5, 1.5]
            lin_vel_y = [-0.8, 0.8]
            ang_vel_yaw = [-1.5, 1.5]

class El4090MammalCollectCfgPPO(El4090MammalCfgPPO):
    class runner(El4090MammalCfgPPO.runner):
        experiment_name = "el4090_mammal_collect"
