from abc import ABCMeta


class ModuleABCMeta(ABCMeta):
    """Metaclass that combines ABCMeta and torch.nn.Module explicitly (if needed)
    or just acts as a marker.
    """

    pass
