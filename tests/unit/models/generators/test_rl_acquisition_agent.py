import pytest
import torch

from mriforge.models.generators.rl_acquisition_agent import RLAcquisitionAgent


class TestRLAcquisitionAgent:
    @pytest.fixture
    def agent_dueling(self):
        return RLAcquisitionAgent(
            in_channels=1,
            img_size=32,  # Small size for speed
            hidden_dim=64,
            action_space=32,
            dueling_dqn=True,
        )

    @pytest.fixture
    def agent_standard(self):
        return RLAcquisitionAgent(
            in_channels=1,
            img_size=32,
            hidden_dim=64,
            action_space=32,
            dueling_dqn=False,
        )

    @pytest.fixture
    def input_tensors(self):
        batch_size = 2
        img_size = 32
        # Image: [B, C, H, W]
        image = torch.randn(batch_size, 1, img_size, img_size)
        # Mask: [B, 1, H, W]
        mask = torch.zeros(batch_size, 1, img_size, img_size)
        return image, mask

    def test_initialization_dueling(self, agent_dueling):
        assert agent_dueling.dueling_dqn
        assert hasattr(agent_dueling, "value_stream")
        assert hasattr(agent_dueling, "advantage_stream")
        assert not hasattr(agent_dueling, "q_head")

    def test_initialization_standard(self, agent_standard):
        assert not agent_standard.dueling_dqn
        assert hasattr(agent_standard, "q_head")
        assert not hasattr(agent_standard, "value_stream")
        assert not hasattr(agent_standard, "advantage_stream")

    def test_forward_dueling(self, agent_dueling, input_tensors):
        image, mask = input_tensors
        q_values = agent_dueling(image, mask)
        assert q_values.shape == (2, 32)  # [Batch, ActionSpace]

    def test_forward_standard(self, agent_standard, input_tensors):
        image, mask = input_tensors
        q_values = agent_standard(image, mask)
        assert q_values.shape == (2, 32)

    def test_select_action_greedy(self, agent_dueling, input_tensors):
        image, mask = input_tensors
        # Epsilon = 0.0 -> Greedy
        actions = agent_dueling.select_action(image, mask, epsilon=0.0)
        assert actions.shape == (2,)
        assert actions.dtype == torch.int64
        assert (actions >= 0).all() and (actions < 32).all()

    def test_select_action_random(self, agent_dueling, input_tensors):
        image, mask = input_tensors
        # Epsilon = 1.1 -> Always random
        actions = agent_dueling.select_action(image, mask, epsilon=1.1)
        assert actions.shape == (2,)
        assert (actions >= 0).all() and (actions < 32).all()

    def test_generate(self, agent_dueling, input_tensors):
        image, mask = input_tensors
        actions = agent_dueling.generate(image, mask)
        assert actions.shape == (2,)
        # Should be equivalent to select_action with epsilon=0.0

    def test_properties(self, agent_dueling):
        assert agent_dueling.name == "rl_acquisition_agent"
        assert isinstance(agent_dueling.get_parameter_count(), int)
        assert agent_dueling.get_parameter_count() > 0
        assert agent_dueling.get_output_shape((1, 32, 32)) == (32,)
