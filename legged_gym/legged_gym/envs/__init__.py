# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.envs.a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from .base.legged_robot import LeggedRobot
from .batch_rollout.robot_batch_rollout import RobotBatchRollout

from .anymal_c.anymal import Anymal, LoadAdaptAnymal, StandAnymal, PoseAnymal
from .anymal_c.mixed_terrains.anymal_c_rough_config import AnymalCRoughCfg, AnymalCRoughCfgPPO
from .anymal_c.flat.anymal_c_flat_config import AnymalCFlatCfg, AnymalCFlatCfgPPO
from .anymal_c.flat.pose_anymal_c_flat_config import PoseAnymalCFlatCfg, PoseAnymalCFlatCfgPPO
from .anymal_c.flat.load_adapt_anymal_c_flat_config import LoadAdaptAnymalCFlatCfg, LoadAdaptAnymalCFlatCfgPPO
from .anymal_c.flat.stand_anymal_c_flat_config import StandAnymalCFlatCfg, StandAnymalCFlatCfgPPO
from .anymal_c.pose_adapt.anymal_c_base_pose_adapt import AnymalCBasePoseAdapt, AnymalCBasePoseAdaptCfg, AnymalCBasePoseAdaptCfgPPO
from .anymal_c.pose_adapt.anymal_c_base_pose_ctrl import AnymalCBasePoseCtrl, AnymalCBasePoseCtrlCfg, AnymalCBasePoseCtrlCfgPPO
from .anymal_b.anymal_b_config import AnymalBRoughCfg, AnymalBRoughCfgPPO
from .anymal_c.batch_rollout.anymal_c_batch_rollout_config import AnymalCBatchRolloutCfg, AnymalCBatchRolloutCfgPPO
from .anymal_c.batch_rollout.anymal_c_batch_rollout_flat_config import AnymalCBatchRolloutFlatCfg, AnymalCBatchRolloutFlatCfgPPO
from .anymal_c.batch_rollout.anymal_c_dialmpc_flat_config import AnymalCDialMPCFlatCfg, AnymalCDialMPCFlatCfgPPO
from .anymal_c.batch_rollout.anymal_c_batch_rollout import AnymalCBatchRollout
from .anymal_c.batch_rollout.anymal_c_traj_grad_sampling import AnymalCTrajGradSampling
from .anymal_c.batch_rollout.anymal_c_traj_grad_sampling_config import AnymalCTrajGradSamplingCfg, AnymalCTrajGradSamplingCfgPPO
from .anymal_c.batch_rollout.anymal_c_nav import AnymalCNav
from .anymal_c.batch_rollout.anymal_c_nav_config import AnymalCNavCfg, AnymalCNavCfgPPO
from .anymal_c.nav_tasks.anymal_c_barrier_cfg import AnymalCNavBarrierCfg, AnymalCNavBarrierCfgPPO
from .anymal_c.nav_tasks.anymal_c_timberpile_cfg import AnymalCNavTimberPileCfg, AnymalCNavTimberPileCfgPPO

from .go2.go2 import Go2, LoadAdaptGo2, StandGo2, PoseGo2
from .go2.flat.go2_rough_config import Go2RoughCfg, Go2RoughCfgPPO
from .go2.flat.go2_flat_config import Go2FlatCfg, Go2FlatCfgPPO
from .go2.flat.pose_go2_flat_config import PoseGo2FlatCfg, PoseGo2FlatCfgPPO
from .go2.flat.load_adapt_go2_flat_config import LoadAdaptGo2FlatCfg, LoadAdaptGo2FlatCfgPPO
from .go2.flat.stand_go2_flat_config import StandGo2FlatCfg, StandGo2FlatCfgPPO
from .go2.batch_rollout.go2_batch_rollout import Go2BatchRollout
from .go2.batch_rollout.go2_batch_rollout_config import Go2BatchRolloutCfg, Go2BatchRolloutCfgPPO
from .go2.batch_rollout.go2_batch_rollout_flat_config import Go2BatchRolloutFlatCfg, Go2BatchRolloutFlatCfgPPO
from .go2.batch_rollout.go2_traj_grad_sampling import Go2TrajGradSampling
from .go2.batch_rollout.go2_traj_grad_sampling_config import Go2TrajGradSamplingCfg, Go2TrajGradSamplingCfgPPO

from .cassie.cassie import Cassie
from .cassie.cassie_config import CassieRoughCfg, CassieRoughCfgPPO
from .cassie.cassie_traj_grad_sampling import CassieTrajGradSampling
from .cassie.cassie_traj_grad_sampling_config import CassieTrajGradSamplingCfg, CassieTrajGradSamplingCfgPPO
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO

from .elspider_air.elspider import ElSpider, PoseElSpider, FootTrackElSpider
from .elspider_air.mixed_terrains.elspider_air_rough_config import ElSpiderAirRoughCfg, ElSpiderAirRoughCfgPPO
from .elspider_air.mixed_terrains.elspider_air_rough_train_config import ElSpiderAirRoughTrainCfg, ElSpiderAirRoughTrainCfgPPO
from .elspider_air.mixed_terrains.elspider_air_rough_raycast_config import ElSpiderAirRoughRaycastCfg, ElSpiderAirRoughRaycastCfgPPO
from .elspider_air.flat.elspider_air_flat_config import ElSpiderAirFlatCfg, ElSpiderAirFlatCfgPPO
from .elspider_air.flat.pose_elspider_air_flat_config import PoseElSpiderAirFlatCfg, PoseElSpiderAirFlatCfgPPO
from .elspider_air.flat.foot_track_elspider_air_flat_config import FootTrackElSpiderAirFlatCfg, FootTrackElSpiderAirFlatCfgPPO
from .elspider_air.flat.foot_track_elspider_air_hang_config import FootTrackElSpiderAirHangCfg, FootTrackElSpiderAirHangCfgPPO
from .elspider_air.elspider_raycast import ElSpiderRayCast
from .elspider_air.pose_adapt.el_mini_base_pose_adapt import ElMiniBasePoseAdapt, ElMiniBasePoseAdaptCfg, ElMiniBasePoseAdaptCfgPPO
from .elspider_air.pose_adapt.el_mini_base_pose_ctrl import ElMiniBasePoseCtrl, ElMiniBasePoseCtrlCfg

from .elspider_air.batch_rollout.elspider_air_batch_rollout import ElSpiderAirBatchRollout
from .elspider_air.batch_rollout.elspider_air_batch_rollout_config import ElSpiderAirBatchRolloutCfg, ElSpiderAirBatchRolloutCfgPPO
from .elspider_air.batch_rollout.elspider_air_batch_rollout_flat_config import ElSpiderAirBatchRolloutFlatCfg, ElSpiderAirBatchRolloutFlatCfgPPO
from .elspider_air.batch_rollout.elspider_air_dialmpc_flat_config import ElSpiderAirDialMPCFlatCfg, ElSpiderAirDialMPCFlatCfgPPO
from .elspider_air.batch_rollout.elspider_air_dialmpc_config import ElSpiderAirDialMPCCfg, ElSpiderAirDialMPCCfgPPO
from .elspider_air.batch_rollout.elspider_air_traj_grad_sampling import ElSpiderAirTrajGradSampling
from .elspider_air.batch_rollout.elspider_air_traj_grad_sampling_config import ElSpiderAirTrajGradSamplingCfg, ElSpiderAirTrajGradSamplingCfgPPO
from .elspider_air.batch_rollout.elspider_air_plan_grad_sampling import ElSpiderAirPlanGradSampling
from .elspider_air.batch_rollout.elspider_air_plan_grad_sampling_config import ElSpiderAirPlanGradSamplingCfg, ElSpiderAirPlanGradSamplingCfgPPO
from .elspider_air.batch_rollout.elspider_air_nav import ElSpiderAirNav
from .elspider_air.batch_rollout.elspider_air_nav_config import ElSpiderAirNavCfg, ElSpiderAirNavCfgPPO
from .elspider_air.nav_tasks.elair_nav_barrier_cfg import ElAirNavBarrierCfg, ElAirNavBarrierCfgPPO
from .elspider_air.nav_tasks.elair_nav_timberpile_cfg import ElAirNavTimberPileCfg, ElAirNavTimberPileCfgPPO
from .elspider_air.nav_tasks.elair_nav_gap_cfg import ElAirNavGapCfg, ElAirNavGapCfgPPO

from .cyberdog2.c2_standdance_config import CyberStandDanceConfig, CyberStandDanceCfgPPO, CyberStandDanceCfgPPOAug, CyberStandDanceCfgPPOEMLP
from .cyberdog2.c2_standdance_env import CyberStandDanceEnv
from .cyberdog2.c2_walk_config import CyberWalkConfig, CyberWalkCfgPPO, CyberWalkCfgPPOAug, CyberWalkCfgPPOEMLP
from .cyberdog2.c2_walk_env import CyberWalkEnv

from .franka.franka import Franka
from .franka.franka_config import FrankaCfg, FrankaCfgPPO
from .franka.batch_rollout.franka_batch_rollout import FrankaBatchRollout
from .franka.batch_rollout.franka_batch_rollout_config import FrankaBatchRolloutCfg, FrankaBatchRolloutCfgPPO

from legged_gym.utils.task_registry import task_registry

# Import teacher-student configurations
from .anymal_c.mixed_terrains.anymal_c_rough_teacher_config import AnymalCRoughTeacherCfg, AnymalCRoughTeacherCfgPPO
from .anymal_c.mixed_terrains.anymal_c_rough_student_config import AnymalCRoughStudentCfg, AnymalCRoughStudentCfgPPO
from .anymal_c.anymal import AnymalStudent

task_registry.register("anymal_c_rough", Anymal, AnymalCRoughCfg(), AnymalCRoughCfgPPO())
task_registry.register("anymal_c_flat", Anymal, AnymalCFlatCfg(), AnymalCFlatCfgPPO())
task_registry.register("pose_anymal_c_flat", PoseAnymal, PoseAnymalCFlatCfg(), PoseAnymalCFlatCfgPPO())
task_registry.register("load_adapt_anymal_c_flat", LoadAdaptAnymal, LoadAdaptAnymalCFlatCfg(), LoadAdaptAnymalCFlatCfgPPO())
task_registry.register("stand_anymal_c_flat", StandAnymal, StandAnymalCFlatCfg(), StandAnymalCFlatCfgPPO())
task_registry.register("anymal_c_base_pose_adapt", AnymalCBasePoseAdapt, AnymalCBasePoseAdaptCfg(), AnymalCBasePoseAdaptCfgPPO())
task_registry.register("anymal_c_base_pose_ctrl", AnymalCBasePoseCtrl, AnymalCBasePoseCtrlCfg(), AnymalCBasePoseCtrlCfgPPO())
task_registry.register("anymal_c_batch_rollout", AnymalCBatchRollout, AnymalCBatchRolloutCfg(), AnymalCBatchRolloutCfgPPO())
task_registry.register("anymal_c_batch_rollout_flat", AnymalCBatchRollout,
                       AnymalCBatchRolloutFlatCfg(), AnymalCBatchRolloutFlatCfgPPO())
task_registry.register("anymal_c_dialmpc_flat", AnymalCBatchRollout, AnymalCDialMPCFlatCfg(), AnymalCDialMPCFlatCfgPPO())
task_registry.register("anymal_c_traj_grad_sampling", AnymalCTrajGradSampling,
                       AnymalCTrajGradSamplingCfg(), AnymalCTrajGradSamplingCfgPPO())
task_registry.register("anymal_c_nav", AnymalCNav, AnymalCNavCfg(), AnymalCNavCfgPPO())
task_registry.register("anymal_c_barrier_nav", AnymalCNav, AnymalCNavBarrierCfg(), AnymalCNavBarrierCfgPPO())
task_registry.register("anymal_c_timberpile_nav", AnymalCNav, AnymalCNavTimberPileCfg(), AnymalCNavTimberPileCfgPPO())

task_registry.register("anymal_b", Anymal, AnymalBRoughCfg(), AnymalBRoughCfgPPO())

# Go2 robot registrations
task_registry.register("go2_rough", Go2, Go2RoughCfg(), Go2RoughCfgPPO())
task_registry.register("go2_flat", Go2, Go2FlatCfg(), Go2FlatCfgPPO())
task_registry.register("pose_go2_flat", PoseGo2, PoseGo2FlatCfg(), PoseGo2FlatCfgPPO())
task_registry.register("load_adapt_go2_flat", LoadAdaptGo2, LoadAdaptGo2FlatCfg(), LoadAdaptGo2FlatCfgPPO())
task_registry.register("stand_go2_flat", StandGo2, StandGo2FlatCfg(), StandGo2FlatCfgPPO())
task_registry.register("go2_batch_rollout", Go2BatchRollout, Go2BatchRolloutCfg(), Go2BatchRolloutCfgPPO())
task_registry.register("go2_batch_rollout_flat", Go2BatchRollout,
                       Go2BatchRolloutFlatCfg(), Go2BatchRolloutFlatCfgPPO())
task_registry.register("go2_traj_grad_sampling", Go2TrajGradSampling,
                       Go2TrajGradSamplingCfg(), Go2TrajGradSamplingCfgPPO())

task_registry.register("a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO())
task_registry.register("cassie", Cassie, CassieRoughCfg(), CassieRoughCfgPPO())
task_registry.register("cassie_traj_grad_sampling", CassieTrajGradSampling,
                       CassieTrajGradSamplingCfg(), CassieTrajGradSamplingCfgPPO())


task_registry.register("elspider_air_rough", ElSpider, ElSpiderAirRoughTrainCfg(), ElSpiderAirRoughTrainCfgPPO())
task_registry.register("elspider_air_rough_raycast", ElSpiderRayCast,
                       ElSpiderAirRoughRaycastCfg(), ElSpiderAirRoughRaycastCfgPPO())
task_registry.register("elspider_air_flat", ElSpider, ElSpiderAirFlatCfg(), ElSpiderAirFlatCfgPPO())
task_registry.register("pose_elspider_air_flat", PoseElSpider, PoseElSpiderAirFlatCfg(), PoseElSpiderAirFlatCfgPPO())
task_registry.register("foot_track_elspider_air_flat", FootTrackElSpider,
                       FootTrackElSpiderAirFlatCfg(), FootTrackElSpiderAirFlatCfgPPO())
task_registry.register("foot_track_elspider_air_hang", FootTrackElSpider,
                       FootTrackElSpiderAirHangCfg(), FootTrackElSpiderAirHangCfgPPO())
task_registry.register("el_mini_base_pose_adapt", ElMiniBasePoseAdapt, ElMiniBasePoseAdaptCfg(), ElMiniBasePoseAdaptCfgPPO())
task_registry.register("el_mini_base_pose_ctrl", ElMiniBasePoseCtrl, ElMiniBasePoseCtrlCfg(), ElMiniBasePoseAdaptCfgPPO())
task_registry.register("elspider_air_batch_rollout", ElSpiderAirBatchRollout,
                       ElSpiderAirBatchRolloutCfg(), ElSpiderAirBatchRolloutCfgPPO())
task_registry.register("elspider_air_batch_rollout_flat", ElSpiderAirBatchRollout,
                       ElSpiderAirBatchRolloutFlatCfg(), ElSpiderAirBatchRolloutFlatCfgPPO())
task_registry.register("elspider_air_dialmpc_flat", ElSpiderAirBatchRollout,
                       ElSpiderAirDialMPCFlatCfg(), ElSpiderAirDialMPCFlatCfgPPO())
task_registry.register("elspider_air_dialmpc", ElSpiderAirBatchRollout,
                       ElSpiderAirDialMPCCfg(), ElSpiderAirDialMPCCfgPPO())
task_registry.register("elspider_air_traj_grad_sampling",
                       ElSpiderAirTrajGradSampling,
                       ElSpiderAirTrajGradSamplingCfg,
                       ElSpiderAirTrajGradSamplingCfgPPO)
task_registry.register("elspider_air_plan_grad_sampling",
                       ElSpiderAirPlanGradSampling,
                       ElSpiderAirPlanGradSamplingCfg(),
                       ElSpiderAirPlanGradSamplingCfgPPO())
task_registry.register("elspider_air_nav", ElSpiderAirNav,
                       ElSpiderAirNavCfg(), ElSpiderAirNavCfgPPO())
task_registry.register("elair_barrier_nav", ElSpiderAirNav,
                       ElAirNavBarrierCfg(), ElAirNavBarrierCfgPPO())
task_registry.register("elair_timberpile_nav", ElSpiderAirNav,
                       ElAirNavTimberPileCfg(), ElAirNavTimberPileCfgPPO())
task_registry.register("elair_gap_nav", ElSpiderAirNav,
                       ElAirNavGapCfg(), ElAirNavGapCfgPPO())

task_registry.register("cyber2_stand", CyberStandDanceEnv, CyberStandDanceConfig(), CyberStandDanceCfgPPOAug())
task_registry.register("cyber2_hop", CyberWalkEnv, CyberWalkConfig(), CyberWalkCfgPPO())
task_registry.register("cyber2_bounce", CyberWalkEnv, CyberWalkConfig(), CyberWalkCfgPPO())
task_registry.register("cyber2_walk", CyberWalkEnv, CyberWalkConfig(), CyberWalkCfgPPO())

# Register teacher-student tasks
task_registry.register("anymal_c_rough_teacher", Anymal, AnymalCRoughTeacherCfg(), AnymalCRoughTeacherCfgPPO())
task_registry.register("anymal_c_rough_student", AnymalStudent, AnymalCRoughStudentCfg(), AnymalCRoughStudentCfgPPO())

# Register franka environments
task_registry.register("franka", Franka, FrankaCfg(), FrankaCfgPPO())
task_registry.register("franka_batch_rollout", FrankaBatchRollout, FrankaBatchRolloutCfg(), FrankaBatchRolloutCfgPPO)