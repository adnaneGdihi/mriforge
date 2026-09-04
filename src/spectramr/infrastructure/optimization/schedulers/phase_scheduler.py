import torch


class PhaseScheduler:
    """
    Manages optimization phases for LaNS/HoBS training.

    Phases:
    0: Warm-up (Static Geometry / M0) - Frozen Dynamics/Biophysics
    1: Dynamics/Biophysics Learning - Unfrozen specific params, increased weights
    2: Joint Fine-tuning - All Unfrozen, balanced weights
    """

    def __init__(self, warmup_epochs: int = 5, dynamics_epochs: int = 20, total_epochs: int = 100):
        """__init__.

        Args:
            warmup_epochs (int): Description.
            dynamics_epochs (int): Description.
            total_epochs (int): Description.
        """
        self.warmup_epochs = warmup_epochs
        self.dynamics_epochs = dynamics_epochs
        self.total_epochs = total_epochs

    def get_phase(self, current_epoch: int) -> str:
        """get_phase.

        Args:
            current_epoch (int): Description.
        Returns:
            str: Description.
        """
        if current_epoch < self.warmup_epochs:
            return "warmup"
        elif current_epoch < (self.warmup_epochs + self.dynamics_epochs):
            return "dynamics"
        else:
            return "joint"

    def update_model_state(self, model: torch.nn.Module, epoch: int):
        """update_model_state.

        Args:
            model (torch.nn.Module): Description.
            epoch (int): Description.
        Returns:
            Any: Description.
        """
        phase = self.get_phase(epoch)
        # Assuming model has methods to freeze/unfreeze
        # Or checking attribute names.

        # Generalized logic for LaNS/HoBS
        if phase == "warmup":
            # Freeze Dynamics (LaNS)
            for name, param in model.named_parameters():
                if "siren" in name or "velocity" in name:
                    param.requires_grad = False
                # Freeze T1/T2 (HoBS) - only learn M0?
                if "biophysics" in name:
                    # If biophysics is (M0, T1, T2), we might want to mask grads?
                    # Hard to split single tensor.
                    # Assuming for now we learn all biophysics or none.
                    # Let's freeze biophysics in warmup if intended.
                    pass

        elif phase == "dynamics":
            for name, param in model.named_parameters():
                if "siren" in name or "velocity" in name:
                    param.requires_grad = True

        elif phase == "joint":
            for param in model.parameters():
                param.requires_grad = True

    def get_loss_weights(self, epoch: int, base_config: dict[str, float]) -> dict[str, float]:
        """Returns modified loss weights based on phase."""
        phase = self.get_phase(epoch)
        weights = base_config.copy()

        if phase == "warmup":
            weights["lambda_smoothness"] = 0.0  # No flow to smooth
            weights["lambda_bloch"] = 0.0  # No biophysics constraints yet
        elif phase == "dynamics":
            # Ramp up smoothness
            pass

        return weights
