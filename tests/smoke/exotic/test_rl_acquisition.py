"""
Unit tests for RL-MRI.
"""

import torch

from mriforge.models.reasoning.rl_acquisition import ActiveScanner, MRIEnv


def test_mri_env_step():
    """Verify Environment transitions."""
    env = MRIEnv(shape=(32, 32), max_steps=5)
    obs = env.reset()
    assert obs.shape == (32,)
    assert obs.sum() == 0

    # Step
    obs, reward, done = env.step(action=5)
    assert obs[5] == 1.0
    assert reward > 0
    assert not done

    # Step Re-acquire
    obs, reward, done = env.step(action=5)
    assert reward < 0  # Penalty


def test_agent_action():
    """Verify Agent selects valid actions."""
    agent = ActiveScanner(num_lines=32)
    mask = torch.zeros(32)

    action = agent.select_action(mask)
    assert 0 <= action < 32

    logits = agent(mask)
    assert logits.shape == (32,)
