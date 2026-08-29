from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


def get_cameraman_tensor(sidelength):
    """Generate a simple test image tensor for demonstration purposes.

    Args:
        sidelength (int): Size of the square image

    Returns:
        torch.Tensor: Image tensor of shape [1, sidelength, sidelength]

    """
    # Create a simple checkerboard pattern as a test image
    x = torch.linspace(-1, 1, sidelength)
    y = torch.linspace(-1, 1, sidelength)
    X, Y = torch.meshgrid(x, y, indexing="ij")

    # Create a checkerboard pattern
    pattern = ((X > 0) ^ (Y > 0)).float()

    # Add some variation to make it more interesting
    pattern = pattern + 0.1 * torch.sin(10 * X) * torch.cos(10 * Y)
    pattern = torch.clamp(pattern, 0, 1)

    return pattern.unsqueeze(0)  # Add channel dimension


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for
    # discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see
    # supplement Sec. 1.5)

    """SineLayer class."""

    def __init__(
        self,
        in_features,
        out_features,
        bias=True,
        is_first=False,
        omega_0=30,
    ):
        """__init__.

        Args:
            in_features (Any): Description.
            out_features (Any): Description.
            bias (Any): Description.
            is_first (Any): Description.
            omega_0 (Any): Description.
        """
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        """init_weights.

        Returns:
            Any: Description.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        """forward.

        Args:
            input (Any): Description.
        Returns:
            Any: Description.

        forward method for SineLayer.

        Executes PyTorch tensor operations.

        Args:
            input (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        """forward_with_intermediate.

        Args:
            input (Any): Description.
        Returns:
            Any: Description.
        """
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    """Siren class."""

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        """__init__.

        Args:
            in_features (Any): Description.
            hidden_features (Any): Description.
            hidden_layers (Any): Description.
            out_features (Any): Description.
            outermost_linear (Any): Description.
            first_omega_0 (Any): Description.
            hidden_omega_0 (Any): Description.
        """
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            ),
        )

        for _i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                ),
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                ),
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        """forward.

        Args:
            coords (Any): Description.
        Returns:
            Any: Description.

        forward method for Siren.

        Executes PyTorch tensor operations.

        Args:
            coords (torch.Tensor, shape (B, C, H, W) [inferred]): Expected input tensor.

        Returns:
            torch.Tensor: Output tensor with shape matching the operation.

        Hardware/Device Context:
            Supports Mixed Precision (AMP) and CUDA streams if configured in DataStagingService."""
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!
        """
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for _i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations["_".join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def laplace(y, x):
    """laplace.

    Args:
        y (Any): Description.
        x (Any): Description.
    Returns:
        Any: Description.
    """
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    """divergence.

    Args:
        y (Any): Description.
        x (Any): Description.
    Returns:
        Any: Description.
    """
    div = 0.0
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(
            y[..., i],
            x,
            torch.ones_like(y[..., i]),
            create_graph=True,
        )[0][..., i : i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    """gradient.

    Args:
        y (Any): Description.
        x (Any): Description.
        grad_outputs (Any): Description.
    Returns:
        Any: Description.
    """
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


class ImageFitting(Dataset):
    """ImageFitting class."""

    def __init__(self, sidelength):
        """__init__.

        Args:
            sidelength (Any): Description.
        """
        super().__init__()
        img = get_cameraman_tensor(sidelength)
        self.pixels = img.permute(1, 2, 0).contiguous().view(-1, 1)
        self.coords = get_mgrid(sidelength, 2)

    def __len__(self):
        """__len__.

        Returns:
            Any: Description.
        """
        return 1

    def __getitem__(self, idx):
        """__getitem__.

        Args:
            idx (Any): Description.
        Returns:
            Any: Description.
        """
        if idx > 0:
            raise IndexError

        return self.coords, self.pixels


class DivergenceFreeSIREN(nn.Module):
    """
    A SIREN-based network that guarantees a divergence-free output vector field.

    Instead of predicting the velocity v directly, it predicts a vector potential Psi.
    The velocity is then computed as v = curl(Psi).

    Analytically: div(curl(Psi)) = 0.
    """

    def __init__(
        self,
        in_features=4,
        hidden_features=256,
        hidden_layers=3,
        out_features=3,
        **kwargs,
    ):
        """
        Args:
           in_features: Input dimension (e.g., x,y,z,t = 4).
           out_features: Output velocity dimension (must be 3 for curl).
        """
        super().__init__()
        # We predict a vector potential Psi
        # If we want 3D velocity, Psi should be 3D.
        self.net = Siren(
            in_features=in_features,
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            out_features=3,  # Psi is 3D
            **kwargs,
        )

    def forward(self, coords):
        """
        Args:
            coords: (B, N, 4) or (B, 4) Spacetime coordinates.

        Returns:
            velocity: (..., 3) Divergence-free velocity field.
            psi: (..., 3) The underlying vector potential.
        """
        coords = coords.clone().detach().requires_grad_(True)
        psi, _ = self.net(coords)

        # Compute Curl: curl(Psi) = (dPsi_z/dy - dPsi_y/dz,
        #                            dPsi_x/dz - dPsi_z/dx,
        #                            dPsi_y/dx - dPsi_x/dy)

        grad_psi_x = gradient(psi[..., 0], coords)
        grad_psi_y = gradient(psi[..., 1], coords)
        grad_psi_z = gradient(psi[..., 2], coords)

        # Assuming coords order is (t, x, y, z) or (x, y, z, t).
        # Let's assume (x, y, z, t) for spatial derivatives.
        # Indices: x=0, y=1, z=2.

        dPsi_x_dy = grad_psi_x[..., 1:2]
        dPsi_x_dz = grad_psi_x[..., 2:3]

        dPsi_y_dx = grad_psi_y[..., 0:1]
        dPsi_y_dz = grad_psi_y[..., 2:3]

        dPsi_z_dx = grad_psi_z[..., 0:1]
        dPsi_z_dy = grad_psi_z[..., 1:2]

        vx = dPsi_z_dy - dPsi_y_dz
        vy = dPsi_x_dz - dPsi_z_dx
        vz = dPsi_y_dx - dPsi_x_dy

        velocity = torch.cat([vx, vy, vz], dim=-1)

        return velocity, psi
