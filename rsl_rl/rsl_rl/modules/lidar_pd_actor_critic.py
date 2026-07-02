"""LidarPDActorCritic  零初始化双 GRU 感知 Actor-Critic。

近端 GRU: input=3, hidden=187, 每帧零初始化, seq_len=256。
远端 GRU: input=3, hidden=64, 每帧零初始化, seq_len=1280。
height_supervisor: Linear(187, 187) 近端特征   高度图, 辅助 MSE 监督。
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation, unpad_trajectories


def _euler_to_quat(roll: float, pitch: float, yaw: float) -> torch.Tensor:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return torch.tensor([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    return q * torch.tensor([-1, -1, -1, 1], device=q.device)


def _quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q_vec = q[:3]
    q_scalar = q[3]
    q_vec_exp = q_vec.expand(*v.shape[:-1], 3)
    t = 2.0 * torch.cross(q_vec_exp, v, dim=-1)
    return v + q_scalar * t + torch.cross(q_vec_exp, t, dim=-1)


class LidarPDActorCritic(nn.Module):
    """LidarPD Actor-Critic with dual zero-init GRU perception.

    Observation layout (from LidarWrapper):
      - proprio (48 dims)
      - proximal points (256   3 = 768 dims, sorted by spherical key)
      - distal points (1280   3 = 3840 dims, 10-frame concatenated + globally sorted)
      Total: 48 + 768 + 3840 = 4656

    Architecture:
      Proximal: (B, 256, 3)   GRU(3  187, zero-init)   h_n (B, 187)
      Distal:   (B, 1280, 3)   GRU(3  64, zero-init)    h_n (B, 64)
      Actor:    (B, 48+187+64=299)   MLP   12
      Critic:   (B, 299)   MLP   1
      Aux:      Linear(187, 187)   MSE with privileged heights
    """

    is_recurrent = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list[int] | tuple = (1024, 512, 256, 128),
        critic_hidden_dims: list[int] | tuple = (1024, 512, 256, 128),
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        proximal_points: int = 256,
        distal_history_length: int = 10,
        distal_points: int = 128,
        proximal_feature_dim: int = 187,
        distal_feature_dim: int = 64,
        proprio_obs_dim: int = 48,
        privileged_height_dim: int = 187,
        privileged_critic_dim: int = 235,
        privileged_supervision_coef: float = 1.0,
        sensor_offset_rpy: list[float] | None = None,
        sensor_offset_pos: list[float] | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "LidarPDActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        self.proximal_points = int(proximal_points)
        self.distal_history_length = int(distal_history_length)
        self.distal_points = int(distal_points)
        self.proximal_feature_dim = int(proximal_feature_dim)
        self.distal_feature_dim = int(distal_feature_dim)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.privileged_height_dim = int(privileged_height_dim)
        self.privileged_critic_dim = int(privileged_critic_dim)
        self.privileged_supervision_coef = float(privileged_supervision_coef)
        self.num_actions = num_actions

        self._sensor_conj: torch.Tensor | None = None
        if sensor_offset_rpy is not None and any(v != 0.0 for v in sensor_offset_rpy):
            self._sensor_conj = _quat_conjugate(_euler_to_quat(*sensor_offset_rpy))
        if sensor_offset_pos is not None and any(v != 0.0 for v in sensor_offset_pos):
            sensor_t = torch.tensor(sensor_offset_pos, dtype=torch.float32)
        else:
            sensor_t = torch.zeros(3, dtype=torch.float32)
        self.register_buffer("_sensor_translation", sensor_t, persistent=False)

        expected_obs = (
            self.proprio_obs_dim
            + self.proximal_points * 3
            + self.distal_history_length * self.distal_points * 3
        )
        if num_actor_obs < expected_obs:
            raise ValueError(
                f"LidarPDActorCritic expects at least {expected_obs} actor obs dims, "
                f"got {num_actor_obs}"
            )

        act_fn = resolve_nn_activation(activation)

        # GRU encoders (no PointNet, raw xyz input)
        self.proximal_gru = nn.GRU(
            input_size=3,
            hidden_size=self.proximal_feature_dim,
            batch_first=True,
        )
        self.distal_gru = nn.GRU(
            input_size=3,
            hidden_size=self.distal_feature_dim,
            batch_first=True,
        )

        # Height supervisor
        self.height_supervisor = nn.Linear(self.proximal_feature_dim, self.privileged_height_dim)

        # Actor / Critic heads
        actor_input_dim = self.proprio_obs_dim + self.proximal_feature_dim + self.distal_feature_dim
        self.actor = self._build_mlp(actor_input_dim, list(actor_hidden_dims), num_actions, act_fn)
        self.critic = self._build_mlp(actor_input_dim, list(critic_hidden_dims), 1, act_fn)

        # Noise parameterisation
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")

        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

        self._cached_proximal_feature: torch.Tensor | None = None
        self._cached_actor_latent: torch.Tensor | None = None

    @staticmethod
    def _build_mlp(in_dim: int, hidden_dims: list[int], out_dim: int, activation: nn.Module) -> nn.Sequential:
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_dims[0]), activation]
        for i in range(len(hidden_dims)):
            if i == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[i], out_dim))
            else:
                layers.append(nn.Linear(hidden_dims[i], hidden_dims[i + 1]))
                layers.append(activation)
        return nn.Sequential(*layers)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def _split_obs(self, observations: torch.Tensor):
        prox_len = self.proximal_points * 3
        dist_len = self.distal_history_length * self.distal_points * 3

        if observations.numel() == 0:
            if observations.dim() == 2:
                return (
                    torch.empty(0, self.proprio_obs_dim, device=observations.device),
                    torch.empty(0, self.proximal_points, 3, device=observations.device),
                    torch.empty(0, self.distal_history_length * self.distal_points, 3, device=observations.device),
                )
            elif observations.dim() == 3:
                t, b, _ = observations.shape
                return (
                    torch.empty(t, b, self.proprio_obs_dim, device=observations.device),
                    torch.empty(t, b, self.proximal_points, 3, device=observations.device),
                    torch.empty(t, b, self.distal_history_length * self.distal_points, 3, device=observations.device),
                )
            raise ValueError(f"Unexpected observations dim: {observations.dim()}")

        if observations.dim() == 2:
            proprio = observations[:, :self.proprio_obs_dim]
            prox = observations[:, self.proprio_obs_dim:self.proprio_obs_dim + prox_len]
            dist = observations[:, self.proprio_obs_dim + prox_len:self.proprio_obs_dim + prox_len + dist_len]
            return proprio, prox.reshape(-1, self.proximal_points, 3), dist.reshape(-1, self.distal_history_length * self.distal_points, 3)

        # 3D: (T, B, dim)
        t, b, _ = observations.shape
        obs_flat = observations.reshape(t * b, -1)
        proprio = obs_flat[:, :self.proprio_obs_dim].reshape(t, b, self.proprio_obs_dim)
        prox = obs_flat[:, self.proprio_obs_dim:self.proprio_obs_dim + prox_len]
        dist = obs_flat[:, self.proprio_obs_dim + prox_len:self.proprio_obs_dim + prox_len + dist_len]
        return (
            proprio,
            prox.reshape(t, b, self.proximal_points, 3),
            dist.reshape(t, b, self.distal_history_length * self.distal_points, 3),
        )

    def _sort_by_spherical(self, points: torch.Tensor) -> torch.Tensor:
        t = self._sensor_translation.to(device=points.device, dtype=points.dtype).view(1, 1, 3)
        pts = points - t
        if self._sensor_conj is not None:
            q = self._sensor_conj.to(device=points.device, dtype=points.dtype)
            pts = _quat_apply(q.unsqueeze(0).expand(pts.shape[0] * pts.shape[1], 4), pts.reshape(-1, 3))
            pts = pts.reshape(points.shape)

        x, y, z = pts[..., 0], pts[..., 1], pts[..., 2]
        r = torch.norm(pts, dim=-1)
        azimuth = torch.atan2(y, x)
        phi = torch.asin(z / (r + 1e-9))
        key = phi * (2.0 * math.pi) + azimuth
        order = torch.argsort(key, dim=1)
        return torch.gather(points, 1, order.unsqueeze(-1).expand_as(points))

    def _encode_proximal_chunked(self, prox_points: torch.Tensor) -> torch.Tensor:
        B, T_prox, P, _ = prox_points.shape
        out = torch.empty(B, T_prox, self.proximal_feature_dim,
                          device=prox_points.device, dtype=prox_points.dtype)
        chunk_size = 128
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = prox_points[start:end]
            c = end - start
            chunk_seq = chunk.reshape(c * T_prox, P, 3)
            _, h = self.proximal_gru(chunk_seq)
            out[start:end] = h.squeeze(0).reshape(c, T_prox, -1)
        return out

    def _encode_distal_chunked(self, dist_points: torch.Tensor) -> torch.Tensor:
        B, T_dist, D, _ = dist_points.shape
        out = torch.empty(B, T_dist, self.distal_feature_dim,
                          device=dist_points.device, dtype=dist_points.dtype)
        chunk_size = 128
        for start in range(0, B, chunk_size):
            end = min(start + chunk_size, B)
            chunk = dist_points[start:end]
            c = end - start
            chunk_seq = chunk.reshape(c * T_dist, D, 3)
            _, h = self.distal_gru(chunk_seq)
            out[start:end] = h.squeeze(0).reshape(c, T_dist, -1)
        return out

    def _build_actor_latent(self, observations: torch.Tensor, masks: torch.Tensor | None = None):
        proprio, proximal, distal = self._split_obs(observations)

        if masks is not None:
            # Training: flatten (T, B, ...) to (T*B, ...) for zero-init GRU per frame
            T_seq, B = proprio.shape[:2]
            prox_flat = proximal.reshape(T_seq * B, self.proximal_points, 3)
            _, h_prox = self.proximal_gru(prox_flat)
            prox_feat = h_prox.squeeze(0).reshape(T_seq, B, self.proximal_feature_dim)

            dist_flat = distal.reshape(T_seq * B, self.distal_history_length * self.distal_points, 3)
            _, h_dist = self.distal_gru(dist_flat)
            dist_feat = h_dist.squeeze(0).reshape(T_seq, B, self.distal_feature_dim)

            proprio = unpad_trajectories(proprio, masks)
            prox_feat = unpad_trajectories(prox_feat, masks)
            dist_feat = unpad_trajectories(dist_feat, masks)
        else:
            # Inference
            prox_sorted = self._sort_by_spherical(proximal)
            prox_feat_t = self._encode_proximal_chunked(prox_sorted.unsqueeze(1))
            prox_feat = prox_feat_t.squeeze(1)

            dist_feat_t = self._encode_distal_chunked(distal.unsqueeze(1))
            dist_feat = dist_feat_t.squeeze(1)

        actor_latent = torch.cat((proprio, prox_feat, dist_feat), dim=-1)

        self._cached_proximal_feature = prox_feat
        self._cached_actor_latent = actor_latent
        return actor_latent

    def update_distribution(self, observations, masks=None, hidden_states=None):
        actor_latent = self._build_actor_latent(observations, masks=masks)
        mean = self.actor(actor_latent)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}")
        self.distribution = Normal(mean, std)

    def reset(self, dones=None):
        pass

    def act(self, observations, masks=None, hidden_states=None, **kwargs):
        self.update_distribution(observations, masks=masks, hidden_states=hidden_states)
        return self.distribution.sample()

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations):
        actor_latent = self._build_actor_latent(observations)
        return self.actor(actor_latent)

    def evaluate(self, critic_observations, masks=None, hidden_states=None, **kwargs):
        if masks is not None:
            if self._cached_actor_latent is not None:
                return self.critic(self._cached_actor_latent)
            actor_latent = self._build_actor_latent(critic_observations, masks=masks)
            return self.critic(actor_latent)
        else:
            if self._cached_actor_latent is not None:
                return self.critic(self._cached_actor_latent)
            expected_dim = (
                self.proprio_obs_dim
                + self.proximal_points * 3
                + self.distal_history_length * self.distal_points * 3
            )
            if critic_observations.shape[-1] < expected_dim:
                raise ValueError(
                    f"evaluate() cold-start expects >= {expected_dim}-dim actor observations, "
                    f"got {critic_observations.shape[-1]}."
                )
            actor_latent = self._build_actor_latent(critic_observations)
            return self.critic(actor_latent)

    def get_auxiliary_loss(self, privileged_heights: torch.Tensor, masks: torch.Tensor | None = None) -> torch.Tensor:
        if self._cached_proximal_feature is None:
            return torch.zeros((), device=privileged_heights.device)
        if self._cached_proximal_feature.numel() == 0:
            return torch.zeros((), device=privileged_heights.device)

        if masks is not None and privileged_heights.dim() == 3:
            privileged_heights = unpad_trajectories(privileged_heights, masks)

        if self._cached_proximal_feature.dim() == 3:
            prox_feat = self._cached_proximal_feature[:, -1, :]
        else:
            prox_feat = self._cached_proximal_feature

        pred = self.height_supervisor(prox_feat)

        if privileged_heights.dim() == 3:
            priv_obs = privileged_heights[:, -1, :]
        else:
            priv_obs = privileged_heights

        actual_dim = priv_obs.shape[-1]
        if actual_dim == self.privileged_height_dim:
            height_target = priv_obs
        elif actual_dim == self.privileged_critic_dim:
            height_target = priv_obs[..., -self.privileged_height_dim:]
        else:
            if actual_dim >= self.privileged_height_dim:
                height_target = priv_obs[..., -self.privileged_height_dim:]
            else:
                print(f"[WARNING] Aux loss skip: actual dim {actual_dim}, expected {self.privileged_height_dim}")
                return torch.zeros((), device=privileged_heights.device)

        if pred.shape[-1] != height_target.shape[-1]:
            min_dim = min(pred.shape[-1], height_target.shape[-1])
            pred = pred[..., :min_dim]
            height_target = height_target[..., :min_dim]

        return self.privileged_supervision_coef * torch.mean(torch.square(pred - height_target))

    def load_state_dict(self, state_dict, strict=True):
        if 'critic.0.weight' in state_dict:
            expected = self.critic[0].weight.shape
            actual = state_dict['critic.0.weight'].shape
            if expected != actual:
                print(f"[LidarPDActorCritic] Critic weight shape mismatch "
                      f"(checkpoint {list(actual)} -> model {list(expected)}). "
                      f"Critic will be randomly initialized.")
                keys_to_remove = [k for k in state_dict if k.startswith('critic.')]
                for k in keys_to_remove:
                    del state_dict[k]

        if 'proximal_gru.weight_ih_l0' in state_dict:
            expected = self.proximal_gru.weight_ih_l0.shape
            actual = state_dict['proximal_gru.weight_ih_l0'].shape
            if expected != actual:
                print(f"[LidarPDActorCritic] Architecture changed "
                      f"(proximal_gru input_size: checkpoint={actual[1]}, model={expected[1]}). "
                      "Perception modules will be randomly initialized.")
                prefix_blacklist = (
                    'proximal_gru.', 'distal_gru.',
                    'proximal_pointnet.', 'distal_pointnet.',
                )
                keys_to_remove = [k for k in state_dict if k.startswith(prefix_blacklist)]
                for k in keys_to_remove:
                    del state_dict[k]

        return super().load_state_dict(state_dict, strict=False)
