import pytest
import torch

from spectramr.models.generators.surgery_loop_agent import (
    ActionPredictor,
    RewardModel,
    StateEncoder,
    SurgeryLoopAgent,
)


class TestSurgeryLoopAgent:
    @pytest.fixture
    def agent(self):
        return SurgeryLoopAgent(
            in_channels=1, out_channels=1, state_dim=32, action_dim=4
        )

    @pytest.fixture
    def input_tensor(self):
        # [B, C, H, W]
        # Note: The spatial decoder upsamples 3 times from 16x16.
        # 16 -> 32 -> 64 -> 128.
        # So input should probably be compatible with encoder downsampling.
        # Encoder: 3 layers of stride 2. 128 -> 64 -> 32 -> 16.
        return torch.randn(2, 1, 128, 128)

    def test_state_encoder(self):
        encoder = StateEncoder(in_channels=1, state_dim=32)
        x = torch.randn(2, 1, 128, 128)
        state = encoder(x)
        assert state.shape == (2, 32)

    def test_action_predictor(self):
        predictor = ActionPredictor(state_dim=32, action_dim=4)
        state = torch.randn(2, 32)
        action = predictor(state)
        assert action.shape == (2, 4)

    def test_reward_model(self):
        reward_model = RewardModel(state_dim=32, action_dim=4)
        state = torch.randn(2, 32)
        action = torch.randn(2, 4)
        reward = reward_model(state, action)
        assert reward.shape == (2, 1)

    def test_agent_initialization(self, agent):
        assert isinstance(agent.state_encoder, StateEncoder)
        assert isinstance(agent.action_predictor, ActionPredictor)
        assert isinstance(agent.reward_model, RewardModel)

    def test_encode_state(self, agent, input_tensor):
        state = agent.encode_state(input_tensor)
        assert state.shape == (2, 32)

    def test_predict_action(self, agent):
        state = torch.randn(2, 32)
        action = agent.predict_action(state)
        assert action.shape == (2, 4)

    def test_predict_outcome(self, agent):
        state = torch.randn(2, 32)
        action = torch.randn(2, 4)
        outcome = agent.predict_outcome(state, action)
        # Decoder: 16x16 -> 32x32 -> 64x64 -> 128x128
        assert outcome.shape == (2, 1, 128, 128)

    def test_forward_without_action(self, agent, input_tensor):
        # Should predict action internally
        outcome = agent(input_tensor)
        assert outcome.shape == (2, 1, 128, 128)

    def test_forward_with_action(self, agent, input_tensor):
        action = torch.randn(2, 4)
        outcome = agent(input_tensor, action=action)
        assert outcome.shape == (2, 1, 128, 128)

    def test_properties(self, agent):
        assert agent.name == "surgery_loop_agent"
        assert isinstance(agent.get_parameter_count(), int)
        assert agent.get_parameter_count() > 0
        assert agent.get_output_shape((1, 128, 128)) == (1, 128, 128)
