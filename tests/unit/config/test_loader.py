from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pytest

from mriforge.config.schemas.loader import (
    Config,
    ConfigFactory,
    JSONConfigLoader,
    YAMLConfigLoader,
    get_config_factory,
)


class TestYAMLConfigLoader:
    def test_load_from_file(self):
        loader = YAMLConfigLoader()
        yaml_content = "epochs: 100\nmodel_type: unet"
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            with patch(
                "yaml.safe_load", return_value={"epochs": 100, "model_type": "unet"}
            ):
                config = loader.load_from_file(Path("config.yaml"))
                assert isinstance(config, Config)
                assert config.epochs == 100
                assert config.model_type == "unet"

    def test_load_from_dict(self):
        loader = YAMLConfigLoader()
        config_dict = {"epochs": 50}
        config = loader.load_from_dict(config_dict)
        assert isinstance(config, Config)
        assert config.epochs == 50


class TestJSONConfigLoader:
    def test_load_from_file(self):
        loader = JSONConfigLoader()
        json_content = '{"epochs": 100, "model_type": "unet"}'
        with patch("builtins.open", mock_open(read_data=json_content)):
            with patch("json.load", return_value={"epochs": 100, "model_type": "unet"}):
                config = loader.load_from_file(Path("config.json"))
                assert isinstance(config, Config)
                assert config.epochs == 100
                assert config.model_type == "unet"

    def test_load_from_dict(self):
        loader = JSONConfigLoader()
        config_dict = {"epochs": 50}
        config = loader.load_from_dict(config_dict)
        assert isinstance(config, Config)
        assert config.epochs == 50


class TestConfig:
    def test_properties(self):
        data = {
            "epochs": 100,
            "batch_size": 16,
            "learning_rate": 0.001,
            "model_type": "resnet",
            "in_channels": 3,
            "out_channels": 3,
            "model_kwargs": {"depth": 5},
            "input_lr_dir": "lr",
            "input_hr_dir": "hr",
        }
        config = Config(data)
        assert config.epochs == 100
        assert config.batch_size == 16
        assert config.learning_rate == 0.001
        assert config.model_type == "resnet"
        assert config.in_channels == 3
        assert config.out_channels == 3
        assert config.model_kwargs == {"depth": 5}
        assert config.input_lr_dir == "lr"
        assert config.input_hr_dir == "hr"

    def test_defaults(self):
        config = Config({})
        assert config.epochs == 200
        assert config.batch_size == 4
        assert config.learning_rate == 1e-5
        assert config.model_type == "unet"
        assert config.in_channels == 1
        assert config.out_channels == 1
        assert config.model_kwargs == {}
        assert config.input_lr_dir == "./data_synthetic/A_LRSI"
        assert config.input_hr_dir == "./data_synthetic/A_HRSI"

    def test_update_from_dict(self):
        config = Config({"epochs": 10})
        config.update_from_dict({"epochs": 20, "batch_size": 8})
        assert config.epochs == 20
        assert config.batch_size == 8

    def test_get_section(self):
        config = Config({"section": {"key": "value"}})
        section = config.get_section("section")
        assert section == {"key": "value"}


class TestConfigFactory:
    def test_create_from_file_yaml(self):
        factory = ConfigFactory()
        with patch.object(YAMLConfigLoader, "load_from_file") as mock_load:
            factory.create_from_file(Path("config.yaml"))
            mock_load.assert_called_once()

    def test_create_from_file_json(self):
        factory = ConfigFactory()
        with patch.object(JSONConfigLoader, "load_from_file") as mock_load:
            factory.create_from_file(Path("config.json"))
            mock_load.assert_called_once()

    def test_create_from_file_unsupported(self):
        factory = ConfigFactory()
        with pytest.raises(ValueError):
            factory.create_from_file(Path("config.txt"))

    def test_create_from_dict(self):
        factory = ConfigFactory()
        config = factory.create_from_dict({"epochs": 10})
        assert isinstance(config, Config)
        assert config.epochs == 10

    def test_register_loader(self):
        factory = ConfigFactory()
        mock_loader = MagicMock()
        factory.register_loader(".txt", mock_loader)
        factory.create_from_file(Path("config.txt"))
        mock_loader.load_from_file.assert_called_once()


def test_get_config_factory():
    factory1 = get_config_factory()
    factory2 = get_config_factory()
    assert factory1 is factory2
    assert isinstance(factory1, ConfigFactory)
