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
from .base.legged_robot_abl_no_phase_inp import LeggedRobotNoPhaseInp
from .base.legged_robot_single_phase_input import LeggedRobotSinglePhaseInp
from .base.legged_robot_multi_vel import LeggedRobotMultiVel
from .base.legged_robot_multi_vel_with_joint_vel_reward import LeggedRobotMultiVelJointVel
from .a1.a1_config import A1RoughCfg, A1RoughCfgPPO
from legged_gym.envs.go1.go1_config import Go1RoughCfg, Go1RoughCfgPPO
from legged_gym.envs.aliengo.aliengo_config import AlienGoRoughCfg, AlienGoRoughCfgPPO
from legged_gym.envs.go2w.go2w_config import Go2wCfg, Go2wCfgPPO
from legged_gym.envs.go2w.go2w_config_abl_no_phase import Go2wCfgNoPhaseInp
from legged_gym.envs.go2w.go2w_config_single_phase_inp import Go2wCfgSinglePhaseInp
from legged_gym.envs.go2w.go2w_config_multi_vel import Go2wCfgMultiVel
from legged_gym.envs.go2w.go2w_config_multi_vel_joint_vel_tar import Go2wCfgMultiVelJointVel


import os

from legged_gym.utils.task_registry import task_registry

task_registry.register( "a1", LeggedRobot, A1RoughCfg(), A1RoughCfgPPO() )
task_registry.register( "go1", LeggedRobot, Go1RoughCfg(), Go1RoughCfgPPO() )
task_registry.register( "aliengo", LeggedRobot, AlienGoRoughCfg(), AlienGoRoughCfgPPO() )
task_registry.register( "go2w", LeggedRobot, Go2wCfg(), Go2wCfgPPO())
task_registry.register( "go2w-no-phase", LeggedRobotNoPhaseInp, Go2wCfgNoPhaseInp(), Go2wCfgPPO())
task_registry.register( "go2w-single-phase", LeggedRobotSinglePhaseInp, Go2wCfgSinglePhaseInp(), Go2wCfgPPO())
task_registry.register( "go2w-multi-vel", LeggedRobotMultiVel, Go2wCfgMultiVel(), Go2wCfgPPO())
task_registry.register( "go2w-multi-vel-joint-vel", LeggedRobotMultiVelJointVel, Go2wCfgMultiVelJointVel(), Go2wCfgPPO())




