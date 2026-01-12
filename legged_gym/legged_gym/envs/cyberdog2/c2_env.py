from isaacgym import gymtorch, gymapi
from legged_gym.envs.cyberdog2.legged_robot_config import LeggedRobotDiffuseLocoCfg
import torch
from collections import deque
from legged_gym.envs.cyberdog2.legged_robot import LeggedRobotDiffuseLoco, LEGGED_GYM_ROOT_DIR
from isaacgym.torch_utils import quat_apply, quat_rotate_inverse, to_torch, torch_rand_float
from legged_gym.utils.math_utils import quat_apply_yaw
import numpy as np
import os


class StackObsEnv(LeggedRobotDiffuseLoco):
    def __init__(self, cfg: LeggedRobotDiffuseLocoCfg, sim_params, physics_engine, sim_device, headless):
        self.cam_env_id = 0
        super().__init__(cfg, sim_params, physics_engine, sim_device, headless)

    def _init_buffers(self):
        super()._init_buffers()
        self.num_history = self.cfg.env.num_state_history
        self.num_stacked_obs = self.cfg.env.num_stacked_obs  # The common obs in RMA
        self.obs_history = deque(maxlen=self.cfg.env.num_state_history)
        for _ in range(self.cfg.env.num_state_history):
            self.obs_history.append(torch.zeros(self.num_envs, self.cfg.env.num_single_state,
                                    dtype=torch.float, device=self.device))
        self.num_env_factors = self.cfg.env.num_env_factors
        self.env_factor_buf = torch.zeros((self.num_envs, self.num_env_factors), dtype=torch.float, device=self.device)

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        for i in range(self.obs_history.maxlen):
            self.obs_history[i][env_ids] = 0.


class CyberEnv(StackObsEnv):
    def compute_observations(self):
        cur_obs_buf = self._compute_common_obs()
        # add noise if needed
        if self.add_noise:
            cur_obs_buf += (2 * torch.rand_like(cur_obs_buf) - 1) * self.noise_scale_vec
        self.obs_history.append(cur_obs_buf)
        self.obs_buf = torch.cat([self.obs_history[i] for i in range(len(self.obs_history))], dim=-1)
        self._compute_privileged_obs()

    def _apply_external_foot_force(self):
        force = 0.0 * (torch.norm(self.foot_velocities[:, :, :2], dim=-1) *
                       self.contact_forces[:, self.feet_indices, 2]).unsqueeze(dim=-1)
        direction = -self.foot_velocities[:, :, :2] / \
            torch.clamp(torch.norm(self.foot_velocities[:, :, :2], dim=-1), min=1e-5).unsqueeze(dim=-1)
        external_forces = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float32, device=self.device)
        external_forces[:, self.feet_indices, :2] = force * direction
        torque = 0.0 * self.contact_forces[:, self.feet_indices, 2].unsqueeze(dim=-1)
        direction = -self.foot_velocities_ang / \
            torch.clamp(torch.norm(self.foot_velocities_ang, dim=-1), min=1e-5).unsqueeze(dim=-1)
        external_torques = torch.zeros((self.num_envs, self.num_bodies, 3), dtype=torch.float32, device=self.device)
        external_torques[:, self.feet_indices] = torque * direction
        direction = quat_apply(self.base_quat, to_torch([0., -1., 0.], device=self.device).repeat((self.num_envs, 1)))
        pitch_vel = quat_rotate_inverse(self.base_quat, self.base_ang_vel)[:, 1]
        pitch_vel[pitch_vel > 0] = 0.
        torque = torch.clamp(0 * -pitch_vel.unsqueeze(dim=1) * torch_rand_float(0.8, 1.2,
                             (self.num_envs, 1), device=self.device), min=-50, max=50)
        mask = self.projected_gravity[:, 2] > -0.1
        external_torques[mask, 0] = (torque * direction)[mask]
        self.gym.apply_rigid_body_force_tensors(self.sim, gymtorch.unwrap_tensor(
            external_forces), gymtorch.unwrap_tensor(external_torques), gymapi.ENV_SPACE)

    def _compute_common_obs(self):
        raise NotImplementedError

    def _compute_privileged_obs(self):
        if self.cfg.env.num_privileged_obs is not None:
            privileged_obs = self.obs_buf.clone()
            if not self.cfg.env.obs_base_vel:
                privileged_obs = torch.cat([privileged_obs, self.base_lin_vel * self.obs_scales.lin_vel], dim=-1)
            if not self.cfg.env.obs_base_vela:
                privileged_obs = torch.cat([privileged_obs, self.base_ang_vel * self.obs_scales.ang_vel], dim=-1)
            if self.cfg.env.priv_obs_friction:
                privileged_obs = torch.cat([privileged_obs, normalize_range(
                    self.friction_coeffs.unsqueeze(dim=-1), self.cfg.domain_rand.friction_range)], dim=-1)
            if self.cfg.env.priv_obs_restitution:
                privileged_obs = torch.cat([privileged_obs, normalize_range(
                    self.restitution_coeffs.unsqueeze(dim=-1), self.cfg.domain_rand.restitution_range)], dim=-1)
            if self.cfg.env.priv_obs_joint_friction:
                privileged_obs = torch.cat([privileged_obs, normalize_range(
                    self.joint_friction, self.joint_friction_range)], dim=-1)
            if self.cfg.env.priv_obs_com:
                privileged_obs = torch.cat([privileged_obs, normalize_range(
                    self.com_displacement, self.cfg.domain_rand.com_displacement_range)], dim=-1)
            if self.cfg.env.priv_obs_mass:
                privileged_obs = torch.cat([privileged_obs, normalize_range(
                    self.mass_offset.unsqueeze(dim=-1), self.cfg.domain_rand.added_mass_range)], dim=-1)
            if self.cfg.env.priv_obs_contact:
                link_in_contact = (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1).float()
                privileged_obs = torch.cat([privileged_obs, link_in_contact], dim=-1)
            self.privileged_obs_buf[:] = privileged_obs

    def _reward_upright(self):
        forward = quat_apply(self.base_quat, self.forward_vec)
        upright_vec = quat_apply_yaw(self.base_quat, self.upright_vec)
        cosine_dist = torch.sum(forward * upright_vec, dim=-1) / torch.norm(upright_vec, dim=-1)
        # dot product with [0, 0, 1]
        # cosine_dist = forward[:, 2]
        reward = torch.square(0.5 * cosine_dist + 0.5)
        return reward

    def _reward_lift_up(self):
        root_height = self.root_states[:, 2]
        # four leg stand is ~0.28
        # sit height is ~0.385
        reward = torch.exp(root_height - self.cfg.rewards.lift_up_threshold) - 1
        return reward

    def _reward_collision(self):
        reward = super()._reward_collision()
        cond = self.episode_length_buf > self.cfg.rewards.allow_contact_steps
        reward = reward * cond.float()
        return reward

    def _reward_action_q_diff(self):
        condition = self.episode_length_buf <= self.cfg.rewards.allow_contact_steps
        reward = torch.sum(torch.square(self.q_diff_buf), dim=-1) * condition.float()
        return reward

    def _reward_feet_slip(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        # xy lin vel
        foot_velocities = torch.square(torch.norm(self.foot_velocities[:, :, 0:2], dim=2).view(self.num_envs, -1))
        # yaw ang vel
        foot_ang_velocities = torch.square(torch.norm(self.foot_velocities_ang[:, :, 2:] / np.pi, dim=2).view(self.num_envs, -1))
        rew_slip = torch.sum(contact_filt * (foot_velocities + foot_ang_velocities), dim=1)
        return rew_slip

    def system_id(self, motor_data_file):
        import pickle
        from PIL import Image as im
        import os
        from tqdm import tqdm
        with open(motor_data_file, "rb") as f:
            motor_data = pickle.load(f)
        start = int(motor_data['loop_rate']) * 2 + 0
        # start = int(motor_data['loop_rate']) * 4 + 240
        # start = int(motor_data["loop_rate"] * 103) + 40
        end = start + int(120.0 * motor_data['loop_rate'])
        motor_q = np.array(motor_data['q'])
        motor_dq = np.array(motor_data['qd'])
        motor_q_des = np.array(motor_data['q_des'])
        motor_kp = np.array(motor_data['kp'])
        motor_kd = np.array(motor_data['kd'])
        has_quat = 'quat' in motor_data
        if 'quat' in motor_data:
            motor_quat = np.array(motor_data['quat'])
            quat_real = torch.from_numpy(motor_quat)[start:end].float().to(self.device)
            real_forward_vec = quat_apply(torch.from_numpy(motor_quat[0: 1]).float().to(self.device), self.forward_vec[0: 1])
            projected_grav_real = quat_rotate_inverse(quat_real, self.gravity_vec[0: 1].tile(quat_real.shape[0], 1))
            projected_forward_real = quat_rotate_inverse(quat_real, real_forward_vec.tile(quat_real.shape[0], 1))
        q_real = torch.from_numpy(np.concatenate(
            [motor_q[:, 3:6], motor_q[:, 0:3], motor_q[:, 9:12], motor_q[:, 6:9]], axis=-1
        ) * np.array([1, -1, -1] * 4))[start:end].float().to(self.device)
        qd_real = torch.from_numpy(np.concatenate(
            [motor_dq[:, 3:6], motor_dq[:, 0:3], motor_dq[:, 9:12], motor_dq[:, 6:9]], axis=-1
        ) * np.array([1, -1, -1] * 4))[start:end].float().to(self.device)
        q_des = torch.from_numpy(np.concatenate(
            [motor_q_des[:, 3:6], motor_q_des[:, 0:3], motor_q_des[:, 9:12], motor_q_des[:, 6:9]], axis=-1
        ) * np.array([1, -1, -1] * 4))[start:end].float().to(self.device)
        kp_des = torch.from_numpy(motor_kp)[start:end].float().to(self.device)
        kd_des = torch.from_numpy(motor_kd)[start:end].float().to(self.device)
        # assert (motor_data['kp'][start] == 60).all()
        # assert (motor_data['kd'][start] == 3).all()

        # sample parameters
        damping_range = self.cfg.domain_rand.joint_damping_range
        friction_range = self.cfg.domain_rand.joint_friction_range
        rb_friction_range = self.cfg.domain_rand.friction_range
        rb_restitution_range = self.cfg.domain_rand.restitution_range
        # set parameters
        sampled_dampings = []
        sampled_frictions = []
        sampled_rb_frictions = []
        for i in range(self.num_envs):
            damping = np.random.uniform(damping_range[0], damping_range[1])
            friction = np.random.uniform(friction_range[0], friction_range[1])
            rb_friction = np.random.uniform(rb_friction_range[0], rb_friction_range[1])
            # feet_friction = np.random.uniform(rb_friction_range[0], rb_friction_range[1])
            rb_restitution = np.random.uniform(rb_restitution_range[0], rb_restitution_range[1])
            if i == 0:
                damping = 0.05
                friction = 0.2
            sampled_dampings.append(damping)
            sampled_frictions.append(friction)
            sampled_rb_frictions.append(rb_friction)
            rigid_shape_props = self.gym.get_actor_rigid_shape_properties(self.envs[i], self.actor_handles[i]).copy()
            for s in range(len(rigid_shape_props)):
                rigid_shape_props[s].friction = rb_friction
                rigid_shape_props[s].restitution = rb_restitution
            self.gym.set_actor_rigid_shape_properties(self.envs[i], self.actor_handles[i], rigid_shape_props)
            if abs(self.gym.get_actor_rigid_shape_properties(self.envs[i], self.actor_handles[i])[0].friction - rb_friction) > 1e-4:
                print(i, self.gym.get_actor_rigid_shape_properties(self.envs[i], self.actor_handles[i])[0].friction, rb_friction)
            dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
            for j in range(len(dof_props)):
                dof_props["damping"][j] = damping
                dof_props["friction"][j] = friction
            self.gym.set_actor_dof_properties(self.envs[i], self.actor_handles[i], dof_props)
        # check
        for i in range(self.num_envs):
            dof_props = self.gym.get_actor_dof_properties(self.envs[i], self.actor_handles[i])
            for j in range(len(dof_props)):
                assert abs(dof_props["damping"][j] - sampled_dampings[i]) < 1e-4
                assert abs(dof_props["friction"][j] - sampled_frictions[i]) < 1e-4
        sampled_dampings = np.array(sampled_dampings)
        sampled_frictions = np.array(sampled_frictions)
        sampled_rb_frictions = np.array(sampled_rb_frictions)
        # generating samples
        metric = 0
        sim_q = []
        sim_projected_grav = []
        # reset
        env_ids = torch.arange(self.num_envs).to(self.device)
        self.dof_pos[env_ids] = torch.tile(q_real[0], (self.num_envs, 1))
        self.dof_vel[env_ids] = torch.tile(qd_real[0], (self.num_envs, 1))

        # Important! should feed actor id, not env id
        env_ids_int32 = self.num_actors * env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self.root_states[env_ids] = self.base_init_state
        # self.root_states[env_ids, 2] = 0.35
        self.root_states[env_ids, :3] += self.env_origins[env_ids]
        if has_quat:
            self.root_states[env_ids, 3:7] = quat_real[0].unsqueeze(dim=0)
        # base velocities
        self.root_states[env_ids, 7:13] = 0.  # [7:10]: lin vel, [10:13]: ang vel
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        for i in tqdm(range(q_real.shape[0] - 1)):
            # apply action
            # print("des", q_des[i])
            # print("old dof pos", self.dof_pos[0], "old dof vel", self.dof_vel[0], "des", q_des[i])
            self.p_gains[:] = kp_des[i]
            self.d_gains[:] = kd_des[i]
            actions = ((q_des[i] - self.default_dof_pos) / self.cfg.control.action_scale).tile((self.num_envs, 1))
            self.torques = self._compute_torques(actions).view(self.torques.shape)
            # print("torques", self.torques[self.cam_env_id])
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            if self.device == 'cpu':
                self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.gym.refresh_actor_root_state_tensor(self.sim)
            self.gym.refresh_net_contact_force_tensor(self.sim)
            self.gym.refresh_rigid_body_state_tensor(self.sim)
            self.gym.refresh_jacobian_tensors(self.sim)
            self.foot_positions = self.rigid_body_state.view(self.num_envs, self.num_bodies, 13)[:, self.feet_indices, 0:3]
            # print("new dof pos", self.dof_pos[0])
            # print("dof_pos", torch.norm(self.dof_pos - self.dof_pos[0:1]), self.dof_pos[0],
            #       "dof vel", torch.norm(self.dof_vel - self.dof_vel[0:1]), self.dof_vel[0])
            # print("next q", q_real[i + 1], "next qd", qd_real[i + 1])
            if has_quat:
                projected_grav_sim = quat_rotate_inverse(self.base_quat, self.gravity_vec)
                projected_fwd_sim = quat_rotate_inverse(self.base_quat, real_forward_vec.tile((self.num_envs, 1)))
            metric = metric + torch.norm(self.dof_pos - q_real[i + 1].unsqueeze(dim=0), dim=-1) \
                + torch.norm(0.0 * (self.dof_vel - qd_real[i + 1].unsqueeze(dim=0)), dim=-1)
            if has_quat:
                metric = metric + 5 * 0 * torch.norm(projected_grav_real[i + 1] - projected_grav_sim, dim=-1) \
                    + 5 * 0 * torch.norm(projected_forward_real[i + 1] - projected_fwd_sim, dim=-1)
            sim_q.append(self.dof_pos.cpu().numpy())
            if has_quat:
                sim_projected_grav.append(projected_grav_sim.cpu().numpy())
            # print("metric", metric[:10])
            if self.cfg.record.record:
                image = self.get_camera_image()
                image = im.fromarray(image.astype(np.uint8))
                filename = os.path.join(self.cfg.record.folder, "%d.png" % i)
                image.save(filename)
        metric = metric.detach().cpu().numpy()
        datapoints = [(i, sampled_dampings[i], sampled_frictions[i], sampled_rb_frictions[i], metric[i])
                      for i in range(len(sampled_dampings))]
        # print("0", "damping", sampled_dampings[0], "friction", sampled_frictions[0],
        #       "rb_friction", sampled_rb_frictions[0],
        #       "rb_restitution", self.gym.get_actor_rigid_shape_properties(self.envs[0], self.actor_handles[0])[0].restitution,
        #       "mass", self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])[0].mass,
        #       "com", self.gym.get_actor_rigid_body_properties(self.envs[0], self.actor_handles[0])[0].com,
        #       "metric", metric[0]
        # )
        all_coms = []
        all_masses = []
        all_restitution = []
        for i in range(len(self.envs)):
            com = self.gym.get_actor_rigid_body_properties(self.envs[i], self.actor_handles[i])[0].com
            masses = [self.gym.get_actor_rigid_body_properties(self.envs[i], self.actor_handles[i])[j].mass for j in range(17)]
            all_coms.append(np.array([com.x, com.y, com.z]))
            all_masses.append(np.array(masses))
            all_restitution.append(self.gym.get_actor_rigid_shape_properties(self.envs[i], self.actor_handles[i])[0].restitution)
        all_coms = np.array(all_coms)
        all_masses = np.array(all_masses)
        all_restitution = np.array(all_restitution)
        print("Average metric", np.mean(metric))
        print("best", "damping", sampled_dampings[np.argmin(metric)],
              "friction", sampled_frictions[np.argmin(metric)],
              "rb_friction", sampled_rb_frictions[np.argmin(metric)],
              "feet_friction", self.gym.get_actor_rigid_shape_properties(
                  self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[7].friction,
              "rb_restitution", self.gym.get_actor_rigid_shape_properties(
                  self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[0].restitution,
              "mass", [self.gym.get_actor_rigid_body_properties(self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[
                  i].mass for i in range(17)],
              "com", self.gym.get_actor_rigid_body_properties(
                  self.envs[np.argmin(metric)], self.actor_handles[np.argmin(metric)])[0].com,
              metric[np.argmin(metric)])
        print("worst", "damping", sampled_dampings[np.argmax(metric)],
              "friction", sampled_frictions[np.argmax(metric)],
              "rb_friction", sampled_rb_frictions[np.argmax(metric)],
              "feet_friction", self.gym.get_actor_rigid_shape_properties(
                  self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[7].friction,
              "rb_restitution", self.gym.get_actor_rigid_shape_properties(
                  self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[0].restitution,
              "mass", [self.gym.get_actor_rigid_body_properties(self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[
                  i].mass for i in range(17)],
              "com", self.gym.get_actor_rigid_body_properties(
                  self.envs[np.argmax(metric)], self.actor_handles[np.argmax(metric)])[0].com,
              metric[np.argmax(metric)])
        datapoints = sorted(datapoints, key=lambda x: x[-1])
        print("elites", datapoints[:10], "worst", datapoints[-10:])
        best_idx = datapoints[0][0]
        best_sim_q = np.array([sim_q[i][best_idx] for i in range(len(sim_q))])
        best_sim_projected_grav = np.array([sim_projected_grav[i][best_idx] for i in range(len(sim_projected_grav))])
        top_idx = [datapoints[i][0] for i in range(len(datapoints) // 10)]
        top_friction = sampled_frictions[np.array(top_idx)]
        top_damping = sampled_dampings[np.array(top_idx)]
        print("top friction range", np.min(top_friction), np.max(top_friction))
        print("top damping range", np.min(top_damping), np.max(top_damping))
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        for i in range(3):
            ax[i].scatter(all_coms[:, i], metric, alpha=0.1)
        plt.savefig("tmp_com.png")
        plt.close(fig)
        fig, ax = plt.subplots(1, 5, figsize=(15, 3))
        for i in range(5):
            ax[i].scatter(all_masses[:, i], metric, alpha=0.1)
        plt.savefig("tmp_mass.png")
        plt.close(fig)
        fig, ax = plt.subplots(2, 2)
        ax[0][0].scatter(sampled_frictions, metric, alpha=0.1)
        with open("tmp_sysid_data.pkl", "wb") as f:
            pickle.dump({
                "sampled_joint_friction": sampled_frictions,
                "metric": metric,
                "best_sim_q": best_sim_q,
                "uncalib_sim_q": np.array([sim_q[i][0] for i in range(len(sim_q))]),
                "real_q": q_real.cpu().numpy()[1:],
            }, f)
        ax[0][0].set_title("joint friction")
        ax[0][1].scatter(sampled_dampings, metric, alpha=0.1)
        ax[0][1].set_title("joint damping")
        ax[1][0].scatter(sampled_rb_frictions, metric, alpha=0.1)
        ax[1][0].set_title("rb friction")
        ax[1][1].scatter(all_restitution, metric, alpha=0.1)
        ax[1][1].set_title("rb restitution")
        plt.savefig("tmp_env_factor.png")
        plt.close(fig)
        fig, ax = plt.subplots(4, 3)
        for r in range(4):
            for c in range(3):
                ax[r][c].plot(best_sim_q[:, 3 * r + c], label="sim")
                ax[r][c].plot(q_real.cpu().numpy()[1:, 3 * r + c], label="real")
                # ax[r][c].plot(q_des.cpu().numpy()[:, 3 * r + c], label="des")
                if r == 3 and c == 2:
                    ax[r][c].legend()
        plt.savefig("tmp_q.png")
        plt.close(fig)
        if has_quat:
            fig, ax = plt.subplots(1, 3)
            for c in range(3):
                ax[c].plot(best_sim_projected_grav[:, c], label="sim")
                ax[c].plot(projected_grav_real.cpu().numpy()[1:, c], label="real")
                # ax[r][c].plot(q_des.cpu().numpy()[:, 3 * r + c], label="des")
                if c == 2:
                    ax[c].legend()
            plt.savefig("tmp_gravity.png")

    def _get_cur_heading(self):
        heading_vec = quat_apply_yaw(self.base_quat, self.forward_vec)
        heading = torch.atan2(heading_vec[:, 1], heading_vec[:, 0])
        return heading


def normalize_range(x: torch.Tensor, limit):
    if isinstance(limit[0], list):
        low = torch.from_numpy(np.array(limit[0])).to(x.device)
        high = torch.from_numpy(np.array(limit[1])).to(x.device)
    else:
        low = limit[0]
        high = limit[1]
    mean = (low + high) / 2
    scale = (high - low) / 2
    if isinstance(scale, torch.Tensor):
        scale = torch.clamp(scale, min=1e-5)
    else:
        scale = max(scale, 1e-5)
    return (x - mean) / scale
