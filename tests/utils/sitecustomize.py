import sys

try:
    from mock_torch import setup_mock_torch_instance
    setup_mock_torch_instance()
except Exception as e:
    pass

import types

# Mock jaxtyping
try:
    import jaxtyping
except ImportError:
    jaxtyping = types.ModuleType("jaxtyping")
    sys.modules["jaxtyping"] = jaxtyping
    jaxtyping.Complex = type("Complex", (), {"__getitem__": lambda cls, item: None})
    jaxtyping.Float = type("Float", (), {"__getitem__": lambda cls, item: None})

# Mock torchvision
try:
    import torchvision
except ImportError:
    sys.modules["torchvision"] = types.ModuleType("torchvision")
    sys.modules["torchvision.transforms"] = types.ModuleType("torchvision.transforms")

# Mock pydantic if needed (though it might be installed in .venv)
