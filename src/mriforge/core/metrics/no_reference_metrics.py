"""No-Reference Image Quality Metrics.

Implements blind quality assessment metrics that do not require a reference image.
Critical for clinical MRI where ground truth is unavailable and for detecting
GAN hallucinations.

References:
    - Mittal, A., et al. "Making a 'Completely Blind' Image Quality Analyzer." IEEE SPL, 2012.
    - Mittal, A., et al. "No-Reference Image Quality Assessment in the Spatial Domain." IEEE TIP, 2012.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mriforge.core.metrics.registry import register_metric


@register_metric(
    "tenengrad_variance",
    aliases=["TenengradVariance", "tenengrad"],
    requires_reference=False,
)
class TenengradVariance:
    """Tenengrad Variance — No-reference sharpness metric.

    Computes the variance of the Sobel gradient magnitude, which serves
    as a computationally efficient proxy for image sharpness.

    .. math::

        \\text{TV} = \\text{Var}\\left(\\sqrt{G_x^2 + G_y^2}\\right)

    where :math:`G_x, G_y` are Sobel gradient responses.

    Higher values indicate sharper images. Blurry images have low gradient
    variance because smoothing suppresses high-frequency edges.
    """

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "tenengrad_variance"

    @property
    def higher_is_better(self) -> bool:
        """higher_is_better.

        Returns:
            bool: Description.
        """
        return True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """Compute Tenengrad variance on prediction (no-reference).

        Args:
            prediction: Input image tensor [B, C, H, W] or [H, W].
            target: Ignored (no-reference metric).

        Returns:
            Tenengrad variance (higher = sharper).
        """
        x = prediction.float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        # Use magnitude if complex (2-channel)
        if x.shape[1] == 2:
            x = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
        elif x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        # Sobel kernels
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=x.dtype,
            device=x.device,
        ).reshape(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=x.dtype,
            device=x.device,
        ).reshape(1, 1, 3, 3)

        gx = F.conv2d(x, sobel_x, padding=1)
        gy = F.conv2d(x, sobel_y, padding=1)
        gradient_mag = torch.sqrt(gx**2 + gy**2 + 1e-8)

        return gradient_mag.var().item()


@register_metric(
    "laplacian_variance",
    aliases=["LaplacianVariance", "lap_var"],
    requires_reference=False,
)
class LaplacianVariance:
    """Laplacian Variance — No-reference focus/sharpness metric.

    Applies a Laplacian filter and measures the variance of the response.
    Lower variance indicates a blurrier, more defocused image.

    .. math::

        \\text{LV} = \\text{Var}(\\nabla^2 I)

    where :math:`\\nabla^2` is the discrete Laplacian operator.
    """

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "laplacian_variance"

    @property
    def higher_is_better(self) -> bool:
        """higher_is_better.

        Returns:
            bool: Description.
        """
        return True

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """Compute Laplacian variance on prediction (no-reference).

        Args:
            prediction: Input image tensor [B, C, H, W] or [H, W].
            target: Ignored (no-reference metric).

        Returns:
            Laplacian variance (higher = sharper).
        """
        x = prediction.float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)

        if x.shape[1] == 2:
            x = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
        elif x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        # Laplacian kernel
        lap = torch.tensor(
            [[0, 1, 0], [1, -4, 1], [0, 1, 0]],
            dtype=x.dtype,
            device=x.device,
        ).reshape(1, 1, 3, 3)

        response = F.conv2d(x, lap, padding=1)
        return response.var().item()


@register_metric("niqe", aliases=["NIQE"], requires_reference=False)
class NIQE:
    """Natural Image Quality Evaluator (NIQE) — Blind quality metric.

    Fits a multivariate Gaussian to Mean Subtracted Contrast Normalized (MSCN)
    coefficients and measures the distance to a pristine natural image model.

    .. math::

        NIQE = \\sqrt{(\\nu_1 - \\nu_2)^T \\left(\\frac{\\Sigma_1 + \\Sigma_2}{2}\\right)^{-1} (\\nu_1 - \\nu_2)}

    where :math:`(\\nu_1, \\Sigma_1)` are the fitted parameters from the test
    image and :math:`(\\nu_2, \\Sigma_2)` from the pristine model.

    Lower NIQE values indicate better perceptual quality (closer to natural images).

    References:
        Mittal, A., et al. "Making a 'Completely Blind' Image Quality Analyzer."
        IEEE Signal Processing Letters, 2012.
    """

    def __init__(self, patch_size: int = 96, stride: int = 48) -> None:
        """__init__.

        Args:
            patch_size (int): Description.
            stride (int): Description.
        """
        self.patch_size = patch_size
        self.stride = stride

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "niqe"

    @property
    def higher_is_better(self) -> bool:
        """higher_is_better.

        Returns:
            bool: Description.
        """
        return False

    def _compute_mscn(self, img: torch.Tensor) -> torch.Tensor:
        """Compute Mean Subtracted Contrast Normalized coefficients.

        Args:
            img: Single-channel image [1, 1, H, W].

        Returns:
            MSCN coefficients [1, 1, H, W].
        """
        # Local mean via average pooling (7×7 window)
        k = 7
        pad = k // 2
        kernel = torch.ones(1, 1, k, k, device=img.device, dtype=img.dtype) / (k * k)
        mu = F.conv2d(img, kernel, padding=pad)

        # Local variance
        mu_sq = F.conv2d(img**2, kernel, padding=pad)
        sigma = torch.sqrt(torch.clamp(mu_sq - mu**2, min=1e-10))

        return (img - mu) / (sigma + 1.0)

    def _fit_generalized_gaussian(self, data: torch.Tensor) -> tuple[float, float, float, float]:
        """Fit asymmetric generalized Gaussian to MSCN data.

        Returns mean, variance, left shape, right shape as features.
        """
        flat = data.flatten()
        mean_val = flat.mean().item()
        var_val = flat.var().item()

        # Left and right variances for asymmetry
        left = flat[flat < mean_val]
        right = flat[flat >= mean_val]
        left_var = left.var().item() if left.numel() > 1 else var_val
        right_var = right.var().item() if right.numel() > 1 else var_val

        return mean_val, var_val, left_var, right_var

    def _extract_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract NSS feature vector from patches.

        Args:
            img: [1, 1, H, W] image.

        Returns:
            Feature vector [num_patches, num_features].
        """
        mscn = self._compute_mscn(img)
        h, w = mscn.shape[-2], mscn.shape[-1]
        ps = self.patch_size
        st = self.stride

        features_list = []
        for y in range(0, h - ps + 1, st):
            for x in range(0, w - ps + 1, st):
                patch = mscn[:, :, y : y + ps, x : x + ps]
                feats = list(self._fit_generalized_gaussian(patch))

                # Paired products (horizontal, vertical, diag)
                shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
                for dy, dx in shifts:
                    shifted = torch.roll(patch, shifts=(dy, dx), dims=(2, 3))
                    product = patch * shifted
                    feats.extend(list(self._fit_generalized_gaussian(product)))

                features_list.append(feats)

        if not features_list:
            return torch.zeros(1, 20, device=img.device)

        return torch.tensor(features_list, device=img.device, dtype=img.dtype)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """Compute NIQE score (no-reference).

        Uses the test image's NSS features compared against an internal
        pristine-image model. Without a pre-trained model, we compute
        the Mahalanobis distance of the patch feature distribution.

        Args:
            prediction: Image tensor [B, C, H, W] or [H, W].
            target: Ignored (no-reference metric).

        Returns:
            NIQE score (lower = better).
        """
        x = prediction.float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 2:
            x = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
        elif x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        features = self._extract_features(x)

        if features.shape[0] < 2:
            return 0.0

        # Without a pre-trained pristine model, we measure the
        # "concentration" of NSS features: tightly clustered features
        # indicate a natural, high-quality image.
        mu = features.mean(dim=0)
        cov = torch.cov(features.T)

        # Regularize covariance
        cov += torch.eye(cov.shape[0], device=cov.device) * 1e-6

        # Mean Mahalanobis distance of features from their own centroid
        diff = features - mu.unsqueeze(0)
        try:
            cov_inv = torch.linalg.inv(cov)
            mahal = torch.sqrt(
                torch.clamp(
                    (diff @ cov_inv * diff).sum(dim=1),
                    min=0.0,
                )
            )
            return mahal.mean().item()
        except torch.linalg.LinAlgError:
            return float(features.var().item())


@register_metric("brisque", aliases=["BRISQUE"], requires_reference=False)
class BRISQUE:
    """BRISQUE — Blind/Referenceless Image Spatial Quality Evaluator.

    Extracts NSS features from MSCN coefficients and computes quality
    from the deviation of the fitted generalized Gaussian parameters
    from a pristine image model.

    Lower BRISQUE scores indicate better perceptual quality.

    .. math::

        \\text{BRISQUE} = f(\\alpha, \\sigma_l^2, \\sigma_r^2, \\eta, \\ldots)

    where :math:`\\alpha` is the shape parameter,
    :math:`\\sigma_l^2, \\sigma_r^2` are left/right variances of the AGGD fit,
    and :math:`\\eta` captures paired product statistics.

    References:
        Mittal, A., et al. "No-Reference Image Quality Assessment in the
        Spatial Domain." IEEE TIP, 2012.
    """

    @property
    def name(self) -> str:
        """name.

        Returns:
            str: Description.
        """
        return "brisque"

    @property
    def higher_is_better(self) -> bool:
        """higher_is_better.

        Returns:
            bool: Description.
        """
        return False

    def _compute_mscn(self, img: torch.Tensor) -> torch.Tensor:
        """Compute MSCN coefficients."""
        k = 7
        pad = k // 2
        kernel = torch.ones(1, 1, k, k, device=img.device, dtype=img.dtype) / (k * k)
        mu = F.conv2d(img, kernel, padding=pad)
        mu_sq = F.conv2d(img**2, kernel, padding=pad)
        sigma = torch.sqrt(torch.clamp(mu_sq - mu**2, min=1e-10))
        return (img - mu) / (sigma + 1.0)

    def _extract_nss_features(self, img: torch.Tensor) -> torch.Tensor:
        """Extract 36 NSS features at two scales.

        Args:
            img: [1, 1, H, W] image.

        Returns:
            Feature vector [36].
        """
        features = []
        for scale in range(2):
            if scale > 0:
                img = F.avg_pool2d(img, kernel_size=2)
                if img.shape[-1] < 8 or img.shape[-2] < 8:
                    break

            mscn = self._compute_mscn(img)

            # MSCN statistics: shape (from variance ratio) and variance
            flat = mscn.flatten()
            features.append(flat.var().item())
            features.append(flat.mean().item())
            # Standardised 4th moment (kurtosis). torch.Tensor has no
            # ``.kurtosis`` method, so the previous ``hasattr`` guard was
            # never True and this slot was a dead constant 0.0 — compute it
            # explicitly so the BRISQUE feature reflects MSCN tail behaviour.
            fc = flat - flat.mean()
            fstd = fc.std().clamp_min(1e-8)
            features.append(((fc / fstd) ** 4).mean().item())

            # Paired products in 4 orientations
            shifts = [(0, 1), (1, 0), (1, 1), (-1, 1)]
            for dy, dx in shifts:
                shifted = torch.roll(mscn, shifts=(dy, dx), dims=(2, 3))
                product = (mscn * shifted).flatten()
                features.append(product.mean().item())
                features.append(product.var().item())

                # Asymmetry
                left = product[product < 0]
                right = product[product >= 0]
                features.append(left.var().item() if left.numel() > 1 else 0.0)
                features.append(right.var().item() if right.numel() > 1 else 0.0)

        return torch.tensor(features, device=img.device, dtype=torch.float32)

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        **kwargs: object,
    ) -> float:
        """Compute BRISQUE score (no-reference).

        Without a pre-trained SVR model, we use the L2 norm of the NSS
        feature deviation from an ideal zero-mean unit-variance model.

        Args:
            prediction: Image tensor [B, C, H, W] or [H, W].
            target: Ignored (no-reference metric).

        Returns:
            BRISQUE score (lower = better quality).
        """
        x = prediction.float()
        if x.dim() == 2:
            x = x.unsqueeze(0).unsqueeze(0)
        elif x.dim() == 3:
            x = x.unsqueeze(0)
        if x.shape[1] == 2:
            x = torch.sqrt(x[:, 0:1] ** 2 + x[:, 1:2] ** 2)
        elif x.shape[1] > 1:
            x = x.mean(dim=1, keepdim=True)

        features = self._extract_nss_features(x)

        # Score is the norm of feature deviation from ideal natural stats
        # Natural images have specific NSS distributions; deviation = quality loss
        return features.norm().item()


# ---------------------------------------------------------------------------
# No-reference metric battery extensions (spec §3, ``no_reference_metrics``).
#
# House contract (see ``mriforge.core.metrics.context``): plain class, ``name``
# property, *self-declared* ``higher_is_better``, ``__call__(self, prediction,
# target=None, *, context=None, **kwargs) -> float``. NEVER raise
# ``ValueError`` — return ``float('nan')`` on missing context / degenerate
# input. Read context via ``resolve_context``; derive a foreground threshold
# mask from ``|prediction|`` when ``ctx.foreground_mask is None``.
# ---------------------------------------------------------------------------


def _nr_magnitude(prediction: torch.Tensor) -> torch.Tensor | None:
    """Reduce an NR ``prediction`` to a real magnitude image ``[H, W]``.

    Handles ``[B, C, H, W]`` (C even ⇒ interleaved real/imag, else channel
    mean), ``[B, 1, H, W]``, ``[C, H, W]``, ``[H, W]`` and complex tensors;
    averages over the batch so the metric returns one float. ``None`` on
    un-handleable input.
    """
    x = prediction
    if x.dim() == 2:
        x = x[None, None]
    elif x.dim() == 3:
        x = x[None]
    if x.dim() != 4:
        return None
    if x.is_complex():
        mag = x.abs().mean(dim=1, keepdim=True)
    elif x.shape[1] == 2:
        mag = torch.sqrt(x[:, 0:1].float() ** 2 + x[:, 1:2].float() ** 2 + 1e-12)
    elif x.shape[1] % 2 == 0 and x.shape[1] > 2:
        xr = x.float().view(x.shape[0], x.shape[1] // 2, 2, x.shape[2], x.shape[3])
        mag = torch.sqrt(xr[:, :, 0] ** 2 + xr[:, :, 1] ** 2 + 1e-12).mean(dim=1, keepdim=True)
    else:
        mag = x.float().mean(dim=1, keepdim=True)
    return mag.mean(dim=0).squeeze(0)


def _nr_fg_mask(mag2d: torch.Tensor, ctx_mask: torch.Tensor | None) -> torch.Tensor:
    """Resolve a boolean foreground mask ``[H, W]`` from context or threshold."""
    if ctx_mask is not None:
        m = ctx_mask
        while m.dim() > 2:
            m = m[0] if m.shape[0] == 1 else m.mean(dim=0)
        if m.shape == mag2d.shape:
            return m > 0.5
    thr = 0.05 * mag2d.abs().max()
    return mag2d.abs() > thr


@register_metric(
    "ecfd",
    aliases=["ECFD", "empirical_cf_rician_deviation"],
    requires_reference=False,
    needs=("foreground_mask",),
    direction="lower",
)
class EmpiricalCharacteristicFunctionRicianDeviation:
    r"""Empirical-Characteristic-Function Rician Deviation (ECFD) — lower is better.

    Fits a Rician :math:`(\hat\nu, \hat\sigma)` to the foreground magnitude via
    the moment relations
    :math:`\mathbb{E}[x^2] = \nu^2 + 2\sigma^2`,
    :math:`\operatorname{Var}[x^2] = 4\sigma^2(\nu^2 + \sigma^2)`, then measures
    the squared :math:`L_2` gap between the empirical characteristic function
    :math:`\hat\phi_n(t) = \tfrac1N \sum_p e^{i t x_p}` and the Rician model CF
    over a small frequency grid:

    .. math::

        \mathrm{ECFD} = \int |\hat\phi_n(t) - \phi_\mathrm{Ric}(t; \hat\nu, \hat\sigma)|^2\, w(t)\, dt

    :math:`\approx 0` when the magnitude follows the fitted Rician / Rayleigh
    law; rises under non-Rician (e.g. structured / heavy-tailed) corruption.
    The Rician CF has no closed form, so :math:`\phi_\mathrm{Ric}` is evaluated
    by Gauss-Legendre quadrature of its integral representation.
    """

    @property
    def name(self) -> str:
        return "ecfd"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: object | None = None,
        **kwargs: object,
    ) -> float:
        from mriforge.core.metrics.context import resolve_context

        ctx = resolve_context(context, kwargs)
        mag = _nr_magnitude(prediction)
        if mag is None or mag.numel() < 16:
            return float("nan")
        fg = _nr_fg_mask(mag, ctx.foreground_mask)
        x = mag[fg].double()
        if x.numel() < 16:
            return float("nan")

        # --- moment-based Rician (nu, sigma) fit ---------------------------
        m2 = (x**2).mean()
        v2 = (x**2).var(unbiased=False)
        # Var(x^2) = 4 sigma^2 (nu^2 + sigma^2);  E[x^2] = nu^2 + 2 sigma^2.
        # => sigma^2 = (E[x^2] - sqrt(max(2 E[x^2]^2 - Var(x^2), 0))) / 2  is
        # numerically fragile; solve the quadratic in sigma^2 directly.
        # Let s = sigma^2: Var = 4 s (m2 - 2 s + s) = 4 s (m2 - s)  → 4 s^2 -4 m2 s + Var = 0.
        a, b, c = 4.0, -4.0 * float(m2), float(v2)
        disc = b * b - 4 * a * c
        if disc < 0:
            disc = 0.0
        s = (-b - math.sqrt(disc)) / (2 * a)  # smaller root → physical sigma^2
        s = max(s, 1e-12)
        nu_sq = max(float(m2) - 2 * s, 0.0)
        nu = math.sqrt(nu_sq)
        sigma = math.sqrt(s)
        if sigma <= 1e-9:
            return float("nan")

        # --- frequency grid (scaled by data spread) ------------------------
        scale = float(x.std()) + 1e-8
        t = torch.linspace(0.1, 3.0, 12, dtype=torch.float64, device=x.device) / scale
        w = torch.exp(-0.5 * (t * scale) ** 2)  # taper down the high-t noise

        # Empirical CF.
        tx = t[:, None] * x[None, :]  # [T, N]
        phi_emp = torch.complex(torch.cos(tx), torch.sin(tx)).mean(dim=1)  # [T]

        # Rician CF by quadrature over its angular integral representation:
        # phi(t) = (1/2pi) ∫_0^{2pi} E[exp(i t |z|)] ... — evaluate via the
        # pdf integral phi(t)=∫_0^∞ p_Ric(r) e^{itr} dr on a fine r-grid.
        rmax = float(x.max()) * 1.5 + 6 * sigma
        r = torch.linspace(1e-6, rmax, 512, dtype=torch.float64, device=x.device)
        dr = r[1] - r[0]
        # Rician pdf: (r/s) exp(-(r^2+nu^2)/(2s)) I0(r nu / s).
        z = r * nu / s
        i0 = torch.special.i0e(z) * torch.exp(z.abs())  # I0 via scaled form
        pdf = (r / s) * torch.exp(-(r**2 + nu_sq) / (2 * s)) * i0
        pdf = torch.nan_to_num(pdf, nan=0.0, posinf=0.0, neginf=0.0)
        norm = (pdf.sum() * dr).clamp_min(1e-12)
        pdf = pdf / norm
        tr_ = t[:, None] * r[None, :]  # [T, R]
        cf_r = torch.complex(torch.cos(tr_), torch.sin(tr_)) * pdf[None, :]
        phi_ric = cf_r.sum(dim=1) * dr  # [T]

        integrand = (phi_emp - phi_ric).abs() ** 2 * w
        ecfd = float((integrand.mean()).item())
        if not math.isfinite(ecfd):
            return float("nan")
        return ecfd


@register_metric(
    "bnac",
    aliases=["BNAC", "background_noise_autocorrelation_colour"],
    requires_reference=False,
    needs=("foreground_mask",),
    direction="lower",
)
class BackgroundNoiseAutocorrelationColour:
    r"""Background-Noise Autocorrelation Colour (BNAC) — lower is better.

    Over the background region :math:`\Omega^c` (the complement of the
    foreground mask), the lagged spatial autocorrelation is

    .. math::

        \rho(\mathbf r) = \frac{\operatorname{Cov}(x(\mathbf p), x(\mathbf p + \mathbf r))}
                               {\operatorname{Var}(x)},
        \qquad
        \mathrm{BNAC} = \sum_{\mathbf r \ne \mathbf 0} w(\mathbf r)\,|\rho(\mathbf r)|

    summed over a small set of lags. :math:`\approx 0` for spatially white
    background noise (all off-zero correlations vanish); rises when the noise is
    *coloured* by g-factor spatial correlation or by regularisation smoothing.
    Lags are weighted by :math:`w(\mathbf r) = 1/\|\mathbf r\|` so near
    neighbours dominate.
    """

    def __init__(self, max_lag: int = 3) -> None:
        self.max_lag = int(max_lag)

    @property
    def name(self) -> str:
        return "bnac"

    @property
    def higher_is_better(self) -> bool:
        return False

    def __call__(
        self,
        prediction: torch.Tensor,
        target: torch.Tensor | None = None,
        *,
        context: object | None = None,
        **kwargs: object,
    ) -> float:
        from mriforge.core.metrics.context import resolve_context

        ctx = resolve_context(context, kwargs)
        mag = _nr_magnitude(prediction)
        if mag is None or mag.numel() < 64:
            return float("nan")
        fg = _nr_fg_mask(mag, ctx.foreground_mask)
        bg = ~fg  # background region Omega^c

        img = mag.double()
        bg_d = bg.double()
        n_bg = float(bg_d.sum())
        if n_bg < 16:
            return float("nan")

        # Background mean / variance (masked).
        mu = (img * bg_d).sum() / n_bg
        xc = (img - mu) * bg_d  # zero outside background
        var = (xc**2).sum() / n_bg
        if float(var) <= 1e-12:
            return float("nan")

        total = 0.0
        _h, _w = img.shape
        for dy in range(-self.max_lag, self.max_lag + 1):
            for dx in range(0, self.max_lag + 1):
                if dy == 0 and dx == 0:
                    continue
                if dx == 0 and dy < 0:
                    continue  # avoid double-counting symmetric lags
                # Overlap region where both p and p+r are valid background.
                shifted = torch.roll(xc, shifts=(dy, dx), dims=(0, 1))
                shifted_mask = torch.roll(bg_d, shifts=(dy, dx), dims=(0, 1))
                # Zero out wrap-around rows/cols.
                valid = bg_d * shifted_mask
                if dy > 0:
                    valid[:dy, :] = 0
                elif dy < 0:
                    valid[dy:, :] = 0
                if dx > 0:
                    valid[:, :dx] = 0
                n_pair = float(valid.sum())
                if n_pair < 8:
                    continue
                cov = (xc * shifted * valid).sum() / n_pair
                rho = float((cov / var).item())
                weight = 1.0 / math.sqrt(dy * dy + dx * dx)
                total += weight * abs(rho)

        if not math.isfinite(total):
            return float("nan")
        return float(total)
