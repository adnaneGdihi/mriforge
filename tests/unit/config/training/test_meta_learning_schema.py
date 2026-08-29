from mriforge.config.schemas.training.meta_learning import (
    AdaptationConfig,
    MetaLearningDataConfig,
    MetaLearningModelConfig,
    MetaLearningObjectivesConfig,
    TrainingConfigMetaLearning,
)


class TestMetaLearningConfig:
    def test_adaptation_config_defaults(self):
        config = AdaptationConfig()
        assert config.adaptation_steps == 5
        assert config.adaptation_lr == 0.01
        assert config.meta_lr == 0.001
        assert config.task_batch_size == 4
        assert config.adaptation_layers == "both"
        assert config.use_memory_bank is False
        assert config.memory_bank_size == 256

    def test_meta_learning_data_config_defaults(self):
        config = MetaLearningDataConfig(data_root="/data")
        assert config.dataset_type == "meta_learning"
        assert config.support_size == 4
        assert config.query_size == 4
        assert config.tasks_per_epoch == 100

    def test_meta_learning_model_config_defaults(self):
        config = MetaLearningModelConfig()
        assert config.model_type == "meta_learning_friendly"
        assert config.base_model_type == "standard_unet"
        assert isinstance(config.adaptation_config, AdaptationConfig)

    def test_training_config_meta_learning(self):
        model_config = MetaLearningModelConfig()
        data_config = MetaLearningDataConfig(data_root="/data")

        config = TrainingConfigMetaLearning(model=model_config, data=data_config)

        assert config.training_mode == "meta_learning"
        assert isinstance(config.objectives, MetaLearningObjectivesConfig)
        assert isinstance(
            config.objectives.reconstruction, object
        )  # ReconstructionLossesConfig

    def test_custom_values(self):
        adaptation_config = AdaptationConfig(adaptation_steps=10)
        model_config = MetaLearningModelConfig(
            base_model_type="resnet", adaptation_config=adaptation_config
        )
        data_config = MetaLearningDataConfig(data_root="/data", support_size=10)

        config = TrainingConfigMetaLearning(model=model_config, data=data_config)

        assert config.model.base_model_type == "resnet"
        assert config.model.adaptation_config.adaptation_steps == 10
        assert config.data.support_size == 10
