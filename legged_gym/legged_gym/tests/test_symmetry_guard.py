"""Test that symmetry loss block is properly guarded by use_mirror_loss flag.

Bug: when use_mirror_loss=False, the symmetry loss block (Line 351 in ppo.py)
still executes act_inference, wasting GPU memory and compute.

Also tests that act_inference() does not overwrite training cache
(cached_proximal_feature / cached_actor_latent).
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint  # noqa: F401 — required by LidarPDActorCritic
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.modules import LidarPDActorCritic


OBS_DIM = 8
ACT_DIM = 2
NUM_ENVS = 4
NUM_STEPS = 4


class _SimplePolicy(nn.Module):
    """Minimal policy with all methods required by PPO.update()."""

    is_recurrent = False

    def __init__(self):
        super().__init__()
        self.actor = nn.Linear(OBS_DIM, ACT_DIM)
        self.critic = nn.Linear(OBS_DIM, 1)
        self.distribution: Normal | None = None

    def update_distribution(self, obs, masks=None, hidden_states=None):
        mean = self.actor(obs)
        std = torch.ones_like(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, masks=None, hidden_states=None, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        return self.actor(obs)

    def evaluate(self, obs, masks=None, hidden_states=None, **kwargs):
        return self.critic(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass


def _dummy_augmentation(obs, actions, env, obs_type):
    """Doubles batch size by concatenating with itself.  Handles obs=None."""
    if obs is not None:
        doubled_obs = torch.cat([obs, obs], dim=0)
    else:
        doubled_obs = None
    doubled_actions = (
        torch.cat([actions, actions], dim=0) if actions is not None else None
    )
    return doubled_obs, doubled_actions


def _build_ppo(use_mirror_loss: bool) -> PPO:
    """Create a PPO instance with specified symmetry settings."""
    policy = _SimplePolicy()
    symmetry_cfg = {
        "use_data_augmentation": False,
        "use_mirror_loss": use_mirror_loss,
        "data_augmentation_func": _dummy_augmentation,
        "mirror_loss_coeff": 1.0,
        "_env": None,
    }
    ppo = PPO(
        policy,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=2,
        symmetry_cfg=symmetry_cfg,
    )
    return ppo


def _populate_storage(ppo: PPO) -> None:
    """Fill rollout storage with dummy transitions and compute returns."""
    ppo.init_storage(
        "rl", NUM_ENVS, NUM_STEPS,
        [OBS_DIM], [OBS_DIM], [ACT_DIM],
        aux_obs_shape=None,
    )
    for _step in range(NUM_STEPS):
        obs = torch.randn(NUM_ENVS, OBS_DIM)
        ppo.transition.observations = obs
        ppo.transition.privileged_observations = obs
        ppo.transition.actions = ppo.policy.act(obs).detach()
        ppo.transition.values = ppo.policy.evaluate(obs).detach()
        ppo.transition.actions_log_prob = ppo.policy.get_actions_log_prob(
            ppo.transition.actions
        ).detach()
        ppo.transition.action_mean = ppo.policy.action_mean.detach()
        ppo.transition.action_sigma = ppo.policy.action_std.detach()
        ppo.transition.rewards = torch.randn(NUM_ENVS)
        ppo.transition.dones = torch.zeros(NUM_ENVS, dtype=torch.bool)
        ppo.storage.add_transitions(ppo.transition)
        ppo.transition.clear()

    last_obs = torch.randn(NUM_ENVS, OBS_DIM)
    ppo.compute_returns(last_obs)


def test_mirror_loss_false_skips_symmetry_block():
    """use_mirror_loss=False should NOT trigger act_inference during update()."""
    ppo = _build_ppo(use_mirror_loss=False)
    _populate_storage(ppo)

    call_count = 0
    original = ppo.policy.act_inference

    def tracked(obs):
        nonlocal call_count
        call_count += 1
        return original(obs)

    ppo.policy.act_inference = tracked
    loss_dict = ppo.update()

    assert call_count == 0, (
        f"act_inference called {call_count} times with use_mirror_loss=False, "
        f"expected 0. The symmetry loss block is not properly guarded."
    )
    # 对称损失关闭时，不应出现在 loss_dict 中
    assert "symmetry" not in loss_dict, (
        f"loss_dict should not contain 'symmetry' key when use_mirror_loss=False, "
        f"got {loss_dict.get('symmetry', 'MISSING')}"
    )


def test_mirror_loss_true_runs_symmetry_block():
    """use_mirror_loss=True SHOULD trigger act_inference during update()."""
    ppo = _build_ppo(use_mirror_loss=True)
    _populate_storage(ppo)

    call_count = 0
    original = ppo.policy.act_inference

    def tracked(obs):
        nonlocal call_count
        call_count += 1
        return original(obs)

    ppo.policy.act_inference = tracked
    loss_dict = ppo.update()

    assert call_count > 0, (
        f"act_inference called {call_count} times with use_mirror_loss=True, "
        f"expected > 0. The symmetry loss block should run when enabled."
    )
    # 对称损失开启时，应出现在 loss_dict 中
    assert "symmetry" in loss_dict, (
        "loss_dict should contain 'symmetry' key when use_mirror_loss=True"
    )


def test_act_inference_does_not_overwrite_cache():
    """act_inference() must not overwrite training cache set by act()."""
    policy = LidarPDActorCritic(
        num_actor_obs=128,
        num_critic_obs=128,
        num_actions=4,
        actor_hidden_dims=[32, 16],
        critic_hidden_dims=[32, 16],
        proximal_points=32,
        distal_history_length=2,
        distal_points=4,
        proximal_feature_dim=16,
        distal_feature_dim=8,
        proprio_obs_dim=8,
        privileged_height_dim=8,
    )

    # act() 设置缓存
    obs_small = torch.randn(3, 128)  # B=3
    policy.act(obs_small)
    proximal_before = policy.cached_proximal_feature.clone()
    latent_before = policy._cached_actor_latent.clone()
    assert proximal_before.shape == (3, 16)
    # actor_latent = cat(proprio, prox_feat, dist_feat) = 8+16+8 = 32
    assert latent_before.shape == (3, 8 + 16 + 8)

    # act_inference 使用不同的 batch size —— 不能覆写缓存
    obs_large = torch.randn(7, 128)  # B=7
    policy.act_inference(obs_large)

    proximal_after = policy.cached_proximal_feature
    latent_after = policy._cached_actor_latent

    assert proximal_after.shape == (3, 16), (
        f"cached_proximal_feature shape changed from (3,16) to {proximal_after.shape}"
    )
    assert latent_after.shape == (3, 8 + 16 + 8), (
        f"cached_actor_latent shape changed to {latent_after.shape}"
    )
    assert torch.equal(proximal_before, proximal_after), (
        "cached_proximal_feature was overwritten by act_inference"
    )
    assert torch.equal(latent_before, latent_after), (
        "cached_actor_latent was overwritten by act_inference"
    )


if __name__ == "__main__":
    print("Test 1: use_mirror_loss=False should skip symmetry block...")
    try:
        test_mirror_loss_false_skips_symmetry_block()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print("Test 2: use_mirror_loss=True should run symmetry block...")
    try:
        test_mirror_loss_true_runs_symmetry_block()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")

    print("Test 3: act_inference should not overwrite training cache...")
    try:
        test_act_inference_does_not_overwrite_cache()
        print("  PASSED")
    except AssertionError as e:
        print(f"  FAILED: {e}")
