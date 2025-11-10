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

import os
import cv2

from datetime import datetime
from tqdm import tqdm

# Single import block to avoid foundation conflicts
import isaacgym
from isaacgym import gymapi, gymtorch
from legged_gym.envs import *
from legged_gym.utils import get_args, task_registry

import numpy as np
import torch

def record_policy_video(args, save_path="./videos/", num_steps=1000, 
                       x_vel=1.0, y_vel=0.0, yaw_vel=0.0):
    """
    Video recording using Isaac Gym with proper simulation stepping
    """
    
    os.makedirs(save_path, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = f"policy_{args.task}_{timestamp}.mp4"
    video_path = os.path.join(save_path, video_name)
    
    print(f"Starting video recording for task: {args.task}")
    print(f"Video will be saved to: {video_path}")
    
    # Setup environment
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 50)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    
    # Create environment once
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    
    # Load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # Get initial observations
    obs = env.get_observations()
    
    # Set velocity commands AFTER getting observations
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel
    
    # Camera setup
    camera_props = gymapi.CameraProperties()
    camera_props.width = 1920
    camera_props.height = 1080
    rendering_camera = env.gym.create_camera_sensor(env.envs[0], camera_props)
    
    # Initial camera position
    env.gym.set_camera_location(rendering_camera, env.envs[0], 
                            gymapi.Vec3(1.5, 1, 3.0),
                            gymapi.Vec3(0, 0, 0))
    
    # Video writer setup
    fps = max(1, int(1.0 / env.dt))
    width, height = camera_props.width, camera_props.height
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    print(f"Recording {num_steps} steps at {fps} FPS...")

    if not video_writer.isOpened():
        print("ERROR: Could not open video writer")
        return None
    
    successful_frames = 0
    wheel_body_names = [n for n in env.body_names if "foot" in n]  # or asset specific substring
    wheel_body_indices = [env.body_names.index(n) for n in wheel_body_names]
    
    for i in tqdm(range(num_steps), desc="Recording"):
        wheel_vels = env.dof_vel[0, env.wheel_indices].cpu().numpy()
        for body_id, vel in zip(wheel_body_indices, wheel_vels):
            intensity = min(abs(vel) / 10.0, 1.0)
            color = gymapi.Vec3(intensity, 0.0, 0.0)
            env.gym.set_rigid_body_color(
                env.envs[0],
                env.actor_handles[0],
                body_id,
                gymapi.MESH_VISUAL,
                color
            )
        # CRITICAL: Get policy action from current observations
        with torch.no_grad():  # Ensure no gradient computation
            actions = policy(obs.detach())
        
        # Set commands before stepping (in case they get reset)
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        
        # Step environment and get new observations
        obs, _, _, _, _, _, _ = env.step(actions.detach())
        
        # Render and capture frame AFTER stepping
        try:
            # Get robot position for camera tracking
            robot_pos = env.root_states[0, :3].cpu().numpy()  # [x, y, z]
            
            # Camera follows behind and above the robot
            cam_offset_x = -2.0  # Behind robot
            cam_offset_y = 0.5   # Slightly to the side
            cam_offset_z = 1.5   # Above robot
            
            cam_pos = gymapi.Vec3(
                float(robot_pos[0] + cam_offset_x), 
                float(robot_pos[1] + cam_offset_y), 
                float(robot_pos[2] + cam_offset_z)
            )
            look_at = gymapi.Vec3(
                float(robot_pos[0]), 
                float(robot_pos[1]), 
                float(robot_pos[2])
            )
            
            env.gym.set_camera_location(rendering_camera, env.envs[0], cam_pos, look_at)
            
            # Render camera images
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            env.gym.start_access_image_tensors(env.sim)
            
            img_tensor = env.gym.get_camera_image(env.sim, env.envs[0], rendering_camera, gymapi.IMAGE_COLOR)
            
            img_array = np.frombuffer(img_tensor, dtype=np.uint8)
            img_reshaped = img_array.reshape((camera_props.height, camera_props.width, 4))  # RGBA
            img_rgb = img_reshaped[:, :, :3]  # Remove alpha
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            
            env.gym.end_access_image_tensors(env.sim)
            
            video_writer.write(img_bgr)
            successful_frames += 1
            
        except Exception as e:
            if i % 100 == 0:  # Print error occasionally
                print(f"Camera capture failed at step {i}: {e}")
    
    # Cleanup
    video_writer.release()
    cv2.destroyAllWindows()
    
    print(f"Video saved: {video_path}")
    print(f"Successfully captured {successful_frames}/{num_steps} frames")
    
    if successful_frames == 0:
        print("WARNING: No frames were captured successfully")
        return None
    
    return video_path

def play(args, x_vel=1.0, y_vel=0.0, yaw_vel=0.0):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.terrain.num_rows = 10
    env_cfg.terrain.num_cols = 8
    env_cfg.terrain.curriculum = True
    env_cfg.terrain.max_init_terrain_level = 9
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.disturbance = False
    env_cfg.domain_rand.randomize_payload_mass = False
    env_cfg.commands.heading_command = False
    # env_cfg.terrain.mesh_type = 'plane'
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.commands[:, 0] = x_vel
    env.commands[:, 1] = y_vel
    env.commands[:, 2] = yaw_vel

    obs = env.get_observations()
    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)


    # export policy as a jit module (used to run it from C++)
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    stop_state_log = 100 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards
    camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    camera_vel = np.array([1., 1., 0.])
    camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    img_idx = 0

    wheel_body_names = [n for n in env.body_names if "foot" in n]  # or asset specific substring
    wheel_body_indices = [env.body_names.index(n) for n in wheel_body_names]

    for i in range(10*int(env.max_episode_length)):
        print(f"wheel vels step {i}", env.dof_vel[:, env.wheel_indices])
        print(f"binary contact step {i}", env.contact_forces[:, env.feet_indices, 2] > 1.)

        wheel_vels = env.dof_vel[0, env.wheel_indices].cpu().numpy()
        for body_id, vel in zip(wheel_body_indices, wheel_vels):
            intensity = min(abs(vel) / 10.0, 1.0)
            color = gymapi.Vec3(intensity, 0.0, 0.0)
            env.gym.set_rigid_body_color(
                env.envs[0],
                env.actor_handles[0],
                body_id,
                gymapi.MESH_VISUAL,
                color
            )
    
        actions = policy(obs.detach())
        env.commands[:, 0] = x_vel
        env.commands[:, 1] = y_vel
        env.commands[:, 2] = yaw_vel
        obs, _, rews, dones, infos, _, _ = env.step(actions.detach())

        if RECORD_FRAMES:
            if i % 2:
                filename = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, 'exported', 'frames', f"{img_idx}.png")
                env.gym.write_viewer_image_to_file(env.viewer, filename)
                img_idx += 1 
        if MOVE_CAMERA:
            camera_position += camera_vel * env.dt
            env.set_camera(camera_position, camera_position + camera_direction)

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale + env.default_dof_pos[robot_index, joint_index].item(),
                    'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                    'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                    'dof_torque': env.torques[robot_index, joint_index].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
            )
        elif i==stop_state_log:
            logger.plot_states()
        if  0 < i < stop_rew_log:
            if infos["episode"]:
                num_episodes = torch.sum(env.reset_buf).item()
                if num_episodes>0:
                    logger.log_rewards(infos["episode"], num_episodes)
        elif i==stop_rew_log:
            logger.print_rewards()

if __name__ == '__main__':
    args = get_args()
    print("Isaac Gym Video Recorder")
    print(f"Task: {args.task}")
    
    # Try main recording method (direct to video)
    print("\n=== Method 1: Direct video recording ===")
    video_path = record_policy_video(
        args,
        save_path="./videos/",
        num_steps=1500,
        x_vel=1.0,
        y_vel=0.0,
        yaw_vel=0.0
    )
    
    # if video_path:
    #     print(f"SUCCESS: {video_path}")
    # else:
    #     print("Method 1 failed, trying frame-based method...")
        
    #     # Fallback: Save frames then create video
    #     print("\n=== Method 2: Frame-based recording ===")
    #     video_path = record_policy_video_simple(
    #         args,
    #         save_path="./videos/",
    #         num_steps=1000,
    #         x_vel=1.0,
    #         y_vel=0.0,
    #         yaw_vel=0.0,
    #         frame_skip=2
    #     )
        
    #     if video_path:
    #         print(f"SUCCESS: {video_path}")
    #     else:
    #         print("All methods failed")