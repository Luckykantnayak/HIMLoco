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

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class Go2wRoughCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.45] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
            
            'FL_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RR_foot_joint':0.0,

            # 'FL_hip_joint': 0.0,   # [rad]
            # 'RL_hip_joint': 0.0,   # [rad]
            # 'FR_hip_joint': 0.0 ,  # [rad]
            # 'RR_hip_joint': 0.0,   # [rad]

            # 'FL_thigh_joint': 0.67,     # [rad]
            # 'RL_thigh_joint': 0.67,   # [rad]
            # 'FR_thigh_joint': 0.67,     # [rad]
            # 'RR_thigh_joint': 0.67,   # [rad]

            # 'FL_calf_joint': -1.3,   # [rad]
            # 'RL_calf_joint': -1.3,    # [rad]
            # 'FR_calf_joint': -1.3,  # [rad]
            # 'RR_calf_joint': -1.3,    # [rad]
            
            # 'FL_foot_joint':0.0,
            # 'RL_foot_joint':0.0,
            # 'FR_foot_joint':0.0,
            # 'RR_foot_joint':0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip_joint': 40.0, 'thigh_joint': 40.0, 'calf_joint': 40.0, 'foot_joint': 20.}  # [N*m/rad]
        damping = {'hip_joint': 1.0, 'thigh_joint': 1.0, 'calf_joint': 1.0, 'foot_joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_reduction = 1.0
    
    class commands( LeggedRobotCfg.commands ):
            curriculum = True
            max_curriculum = 2.0
            num_commands = 4 # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
            resampling_time = 10. # time before command are changed[s]
            heading_command = True # if true: compute ang vel command from heading error
            class ranges( LeggedRobotCfg.commands.ranges):
                lin_vel_x = [-1.0, 1.0] # min max [m/s]
                lin_vel_y = [0.0, 0.0]   # min max [m/s]
                ang_vel_yaw = [-3.14, 3.14]    # min max [rad/s]
                heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/go2w/urdf/go2w.urdf'
        name = "go2w"
        foot_name = "foot"
        wheel_name = ["foot"]
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = ["base"]
        privileged_contacts_on = ["base", "thigh", "calf"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = True # Some .obj meshes must be flipped from y-up to z-up
  
    class rewards( LeggedRobotCfg.rewards ):
        class scales:
            termination = -0.0
            tracking_lin_vel = 1.0
            tracking_ang_vel = 0.5
            lin_vel_z = -2.0
            ang_vel_xy = -0.05
            orientation = -0.2
            dof_acc = -2.5e-7
            joint_power = -2e-5
            base_height = -1.0
            foot_clearance = 0.0
            action_rate = -0.01
            smoothness = -0.01
            feet_air_time =  0.0
            collision = -0.0
            feet_stumble = -0.0
            stand_still = -0.
            torques = -0.000001
            dof_vel = -0.0
            dof_pos_limits = -0.0
            dof_vel_limits = -0.0
            torque_limits = -0.0
            hip_action_l2 = -0.1


        only_positive_rewards = False # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 1. # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 1.
        soft_torque_limit = 1.
        base_height_target = 0.32
        max_contact_force = 100. # forces above this value are penalized
        clearance_height_target = -0.2

class Go2wRoughCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'rough_go2w'

  