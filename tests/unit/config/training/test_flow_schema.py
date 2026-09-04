from spectramr.config.schemas.training.flow import FlowConfig, TrainingConfigFlow


class TestFlowConfig:
    def test_defaults(self):
        config = FlowConfig()
        assert config.sampling_method == "euler"
        assert config.num_inference_steps == 1
        assert config.reflow_iterations == 0

    def test_custom_values(self):
        config = FlowConfig(
            sampling_method="rk45", num_inference_steps=10, reflow_iterations=2
        )
        assert config.sampling_method == "rk45"
        assert config.num_inference_steps == 10
        assert config.reflow_iterations == 2


class TestTrainingConfigFlow:
    def test_defaults(self):
        # training_mode is deprecated; create config without it
        config = TrainingConfigFlow()
        assert isinstance(config.flow, FlowConfig)
        assert config.flow.sampling_method == "euler"

    def test_custom_values(self):
        flow_config = FlowConfig(sampling_method="rk45")
        config = TrainingConfigFlow(flow=flow_config)
        assert config.flow == flow_config
        assert config.flow.sampling_method == "rk45"
