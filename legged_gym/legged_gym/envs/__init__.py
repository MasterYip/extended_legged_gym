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
from .base.legged_robot import LeggedRobot
from legged_gym.utils.task_registry import task_registry

from .el_4090.spider_nomal.el_4090 import EL_4090
from .el_4090.spider_nomal.el4090_spider_config import El4090SpiderCfg, El4090SpiderCfgPPO

from .el_4090.safe.el_4090_safe import EL_4090_Safe
from .el_4090.safe.el_4090_safe_config import El4090SafeCfg, El4090SafeCfgPPO

from .el_4090.spider_mammal.el_4090 import EL_4090_Mammal
from .el_4090.spider_mammal.el4090_spider_config import El4090MammalCfg,El4090MammalCfgPPO

from .el_4090.spider_both.el_4090 import EL_4090_BOTH
from .el_4090.spider_both.el4090_spider_config import El4090BothCfg,El4090BothCfgPPO

from .el_4090.spider_envelop.el_4090 import EL_4090_ENVELOP
from .el_4090.spider_envelop.el4090_spider_config import El4090EnvelopCfg, El4090EnvelopCfgPPO



# Register EL_4090 environments
task_registry.register("el4090_spider", EL_4090, El4090SpiderCfg(), El4090SpiderCfgPPO())
task_registry.register("el_4090_safe", EL_4090_Safe, El4090SafeCfg(), El4090SafeCfgPPO())
task_registry.register("el4090_mammal",EL_4090_Mammal,El4090MammalCfg(),El4090MammalCfgPPO())
task_registry.register("el4090_both",EL_4090_BOTH,El4090BothCfg(),El4090BothCfgPPO())
task_registry.register("el4090_envelop", EL_4090_ENVELOP, El4090EnvelopCfg(), El4090EnvelopCfgPPO())