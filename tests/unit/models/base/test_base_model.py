import pytest
import torch
from torch import nn

from spectramr.models.base.base_model import BaseDiscriminator, BaseGenerator


class ConcreteGenerator(BaseGenerator):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x * 2


class ConcreteDiscriminator(BaseDiscriminator):
    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return x.mean()

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class TestBaseGenerator:
    def test_init(self):
        gen = ConcreteGenerator(in_channels=1, out_channels=2)
        assert gen.in_channels == 1
        assert gen.out_channels == 2
        assert gen.name == "ConcreteGenerator"

    def test_forward_not_implemented(self):
        # Create a class that doesn't implement forward
        class IncompleteGenerator(BaseGenerator):
            pass

        # Instantiation is fine (it's a nn.Module)
        gen = IncompleteGenerator(1, 1)

        # Calling forward should raise NotImplementedError
        with pytest.raises(NotImplementedError):
            gen(torch.randn(1, 1, 10, 10))

    def test_generate(self):
        gen = ConcreteGenerator(1, 1)
        x = torch.ones(1, 1, 10, 10)
        out = gen.generate(x)
        assert torch.all(out == 2.0)

    def test_get_parameter_count(self):
        gen = ConcreteGenerator(1, 1)
        # Add a parameter
        gen.param = nn.Parameter(torch.ones(10))
        assert gen.get_parameter_count() == 10

        # Add a non-grad parameter
        gen.fixed = nn.Parameter(torch.ones(5), requires_grad=False)
        assert gen.get_parameter_count() == 10

    def test_get_output_shape(self):
        gen = ConcreteGenerator(in_channels=1, out_channels=3)
        input_shape = (4, 1, 32, 32)
        output_shape = gen.get_output_shape(input_shape)
        assert output_shape == (4, 3, 32, 32)


class TestBaseDiscriminator:
    def test_init(self):
        disc = ConcreteDiscriminator(in_channels=3)
        assert disc.in_channels == 3
        assert disc.name == "ConcreteDiscriminator"

    def test_forward_not_implemented(self):
        class IncompleteDiscriminator(BaseDiscriminator):
            def get_parameter_count(self):
                return 0

        disc = IncompleteDiscriminator(1)
        with pytest.raises(NotImplementedError):
            disc(torch.randn(1, 1, 10, 10))

    def test_discriminate(self):
        disc = ConcreteDiscriminator(1)
        x = torch.ones(1, 1, 10, 10)
        out = disc.discriminate(x)
        assert out == 1.0

    def test_get_feature_maps(self):
        disc = ConcreteDiscriminator(1)
        x = torch.randn(1, 1, 10, 10)
        features = disc.get_feature_maps(x)
        assert isinstance(features, dict)
        assert len(features) == 0
