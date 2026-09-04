import math

import torch


def compute_b1_minus_biot_savart(
    grid_shape: tuple[int, int],
    num_coils: int = 8,
    coil_radius: float = 0.1,
    cylinder_radius: float = 0.15,
    fov: float = 0.2,
    num_segments: int = 64,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Computes analytical B1- receive sensitivity maps using the Biot-Savart Law.

    Simulates a cylindrical array of circular coil loops.

    Args:
        grid_shape: (H, W) tuple for the 2D grid resolution.
        num_coils: Number of circular coils arranged in a cylinder.
        coil_radius: Radius of each circular coil (in meters).
        cylinder_radius: Radius of the cylinder where coils are placed (in meters).
        fov: Field of view of the 2D grid (in meters).
        num_segments: Number of discrete segments to approximate the circular loop integration.
        device: Torch device.

    Returns:
        Complex tensor of shape (num_coils, H, W) representing the B1- receive fields.
    """
    if num_coils <= 0:
        raise ValueError(f"num_coils must be a positive integer, got {num_coils}")

    H, W = grid_shape
    y, x = torch.meshgrid(
        torch.linspace(-fov / 2, fov / 2, H, device=device),
        torch.linspace(-fov / 2, fov / 2, W, device=device),
        indexing="ij",
    )
    z = torch.zeros_like(x)

    # Points where we want to evaluate the field shape: (H, W, 3)
    points = torch.stack([x, y, z], dim=-1)

    # Pre-allocate sensitivities
    sensitivities = torch.zeros((num_coils, H, W), dtype=torch.complex64, device=device)

    mu_0 = 4 * math.pi * 1e-7
    I = 1.0  # Unit current

    for c in range(num_coils):
        # Angle of the coil center
        theta_c = 2 * math.pi * c / num_coils

        # Coil center
        r_c = torch.tensor(
            [
                cylinder_radius * math.cos(theta_c),
                cylinder_radius * math.sin(theta_c),
                0.0,
            ],
            device=device,
        )

        # Orthonormal basis for the coil plane
        # Normal points inwards towards origin
        n = torch.tensor([-math.cos(theta_c), -math.sin(theta_c), 0.0], device=device)
        # Tangent in xy plane
        t = torch.tensor([-math.sin(theta_c), math.cos(theta_c), 0.0], device=device)
        # Z axis
        z_hat = torch.tensor([0.0, 0.0, 1.0], device=device)

        # Discretize the coil loop into segments
        alpha = torch.linspace(0, 2 * math.pi, num_segments + 1, device=device)

        # Segment points: circle in the plane spanned by t and z_hat
        coil_points = r_c.view(1, 3) + coil_radius * (
            torch.outer(torch.cos(alpha), t) + torch.outer(torch.sin(alpha), z_hat)
        )

        # dl vectors for each segment
        dl = coil_points[1:] - coil_points[:-1]
        midpoints = (coil_points[1:] + coil_points[:-1]) / 2.0  # (num_segments, 3)

        # Compute Biot-Savart integral sum over all points
        # B = (mu_0 I / 4 pi) * Sum [ dl x (r - r') / |r - r'|^3 ]

        # Expand points to (H, W, 1, 3) and midpoints to (1, 1, num_segments, 3)
        r = points.unsqueeze(2)
        r_prime = midpoints.view(1, 1, num_segments, 3)

        diff = r - r_prime  # (H, W, N_seg, 3)
        dist = torch.norm(diff, dim=-1, keepdim=True)  # (H, W, N_seg, 1)

        # Cross product: dl (1, 1, N_seg, 3) x diff (H, W, N_seg, 3)
        dl_expanded = dl.view(1, 1, num_segments, 3).expand(H, W, num_segments, 3)
        cross_prod = torch.cross(dl_expanded, diff, dim=-1)  # (H, W, N_seg, 3)

        dB = (mu_0 * I / (4 * math.pi)) * cross_prod / (dist**3 + 1e-12)
        B = torch.sum(dB, dim=2)  # (H, W, 3)

        # Returns Bx + i By = conj(B1-), where the conventional receive sensitivity
        # is B1- = Bx - i By (Hoult convention, up to a constant). This is the
        # complex conjugate of B1-; phase is therefore defined only up to this
        # global conjugation, which is immaterial under the RSS magnitude
        # normalization applied below. Use conj() here if a consumer needs the
        # textbook B1- absolute phase.
        # We also want to include a phase wrap that depends on the coil position
        # to match realistic MRI phase combinations.
        B_comp = torch.complex(B[..., 0], B[..., 1])

        # Weight by the natural phase twist from standard volume alignment
        phase_shift = torch.exp(1j * torch.tensor(theta_c, device=device))

        sensitivities[c] = B_comp * phase_shift

    # Normalize roughly so max magnitude is 1.0
    rss = torch.sqrt(torch.sum(torch.abs(sensitivities) ** 2, dim=0, keepdim=True))
    sensitivities = sensitivities / (rss.max() + 1e-12)

    return sensitivities
