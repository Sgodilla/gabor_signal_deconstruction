import torch
import torch.nn.functional as F
import numpy as np

__all__ = [
    "compute_gabor_old",
    "compute_gabor_pre_smooth",
    "compute_gabor_optimized",
]

# ------------------------------------------------------------
# Helper utilities
# ------------------------------------------------------------


def _apply_fft_smoothing_along_dim(
    data: torch.Tensor, sigma: float, dim: int
) -> torch.Tensor:
    """Gaussian smoothing along a single dimension using direct convolution.

    This version replaces the original FFT-based implementation to avoid
    boundary artifacts (rippling). It uses a direct convolution with
    reflection padding for superior handling of signal edges.
    """
    if sigma <= 0:
        return data

    device, dtype = data.device, data.dtype
    dim_size = data.shape[dim]

    # --- 1. Create the Gaussian kernel ---
    ksize = int(6 * sigma) + 1
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, dim_size)

    x = torch.arange(ksize, dtype=dtype, device=device) - ksize // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # --- 2. Reshape data for 1D convolution ---
    # Move the target dimension to the end for easier processing.
    dims = list(range(data.ndim))
    dims[dim], dims[-1] = dims[-1], dims[dim]
    data_perm = data.permute(*dims)
    orig_shape = data_perm.shape

    # Flatten all dimensions except the last one (the convolution dimension).
    # Shape becomes (N_batch_flat, Length)
    flat_data = data_perm.reshape(-1, orig_shape[-1])
    # Add a channel dimension for conv1d: (N_batch_flat, 1, Length)
    conv_input = flat_data.unsqueeze(1)

    # Prepare the kernel for conv1d: (C_out, C_in/groups, K_size)
    conv_kernel = kernel.view(1, 1, -1)

    # --- 3. Pad and convolve ---
    # Use 'reflect' padding to minimize edge artifacts. The amount of
    # padding on each side is (ksize - 1) / 2 for a 'same' convolution.
    padding_size = (ksize - 1) // 2
    padded_input = F.pad(conv_input, (padding_size, padding_size), mode="reflect")

    # Perform the 1D convolution with 'valid' padding as we have already padded.
    conv_output = F.conv1d(padded_input, conv_kernel, padding="valid")

    # --- 4. Reshape back to original dimensions ---
    # Remove the channel dimension and reshape to the original (permuted) shape.
    output_perm = conv_output.squeeze(1).reshape(orig_shape)

    # Permute dimensions back to their original order.
    final_output = output_perm.permute(*dims)

    return final_output


# ------------------------------------------------------------
# Core convolution engine
# ------------------------------------------------------------


def _fft_convolve_real_imag(
    signal: torch.Tensor, filt_r: torch.Tensor, filt_i: torch.Tensor
):
    """1‑D FFT convolution of *K* complex filters with signal."""
    L = signal.shape[0]
    K, Ksize = filt_r.shape
    total = L + Ksize - 1
    nfft = 2 ** int(np.ceil(np.log2(total)))

    sig_pad = F.pad(signal, (0, nfft - L))
    r_pad = F.pad(filt_r, (0, nfft - Ksize))
    i_pad = F.pad(filt_i, (0, nfft - Ksize))

    S = torch.fft.rfft(sig_pad, n=nfft)
    R = torch.fft.rfft(r_pad, n=nfft)
    I = torch.fft.rfft(i_pad, n=nfft)

    conv_r = torch.fft.irfft(R * S.unsqueeze(0), n=nfft)
    conv_i = torch.fft.irfft(I * S.unsqueeze(0), n=nfft)

    start = (Ksize - 1) // 2
    conv_r = conv_r[:, start : start + L]
    conv_i = conv_i[:, start : start + L]
    return conv_r, conv_i


def _envelope_from_conv(conv_r, conv_i, filt_r, filt_i, peak_enhancement, n_f, n_s):
    env = torch.sqrt(conv_r**2 + conv_i**2)
    energy = torch.sqrt(torch.sum(filt_r**2 + filt_i**2, dim=1, keepdim=True))
    env = env / (energy + 1e-8)
    if peak_enhancement != 1.0:
        env = env.pow(peak_enhancement)
    L = env.shape[1]
    return env.reshape(n_f, n_s, L)


# ------------------------------------------------------------
# Filter builder
# ------------------------------------------------------------


def _build_gabor_filters(x, frequencies, sigmas):
    freq_grid, sigma_grid = torch.meshgrid(frequencies, sigmas, indexing="ij")
    f_flat = freq_grid.reshape(-1)
    s_flat = sigma_grid.reshape(-1)
    x_exp = x.unsqueeze(0).repeat(f_flat.shape[0], 1)
    norm = 1.0 / torch.sqrt(2 * np.pi * s_flat.unsqueeze(1))
    gauss = torch.exp(-(x_exp**2) / (2 * s_flat.unsqueeze(1) ** 2))
    real = norm * gauss * torch.cos(2 * np.pi * f_flat.unsqueeze(1) * x_exp)
    imag = norm * gauss * torch.sin(2 * np.pi * f_flat.unsqueeze(1) * x_exp)
    return real, imag


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------


def compute_gabor_old(
    signal, x, frequencies, sigmas, peak_enhancement=2.0, position_smoothing_sigma=0.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal, x = signal.to(device), x.to(device)
    frequencies, sigmas = frequencies.to(device), sigmas.to(device)

    filt_r, filt_i = _build_gabor_filters(x, frequencies, sigmas)
    conv_r, conv_i = _fft_convolve_real_imag(signal, filt_r, filt_i)
    env = _envelope_from_conv(
        conv_r, conv_i, filt_r, filt_i, peak_enhancement, len(frequencies), len(sigmas)
    )

    if position_smoothing_sigma > 0:
        env = _apply_fft_smoothing_along_dim(env, position_smoothing_sigma, dim=2)
    return env


def compute_gabor_pre_smooth(
    signal, x, frequencies, sigmas, peak_enhancement=2.0, position_smoothing_sigma=0.0
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal, x = signal.to(device), x.to(device)
    frequencies, sigmas = frequencies.to(device), sigmas.to(device)

    filt_r, filt_i = _build_gabor_filters(x, frequencies, sigmas)
    if position_smoothing_sigma > 0:
        filt_r = _apply_fft_smoothing_along_dim(filt_r, position_smoothing_sigma, dim=1)
        filt_i = _apply_fft_smoothing_along_dim(filt_i, position_smoothing_sigma, dim=1)

    conv_r, conv_i = _fft_convolve_real_imag(signal, filt_r, filt_i)
    return _envelope_from_conv(
        conv_r, conv_i, filt_r, filt_i, peak_enhancement, len(frequencies), len(sigmas)
    )


def compute_gabor_optimized(
    signal, x, frequencies, sigmas, peak_enhancement=2.0, position_smoothing_sigma=0.0
):
    """Single convolution with analytic (Gabor ✱ Gaussian) kernel."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal, x = signal.to(device), x.to(device)
    frequencies, sigmas = frequencies.to(device), sigmas.to(device)

    # Composite sigma after Gaussian–Gaussian convolution
    sigma_c = torch.sqrt(sigmas**2 + position_smoothing_sigma**2)  # (|σ|,)

    # Build parameter grids (freq × sigma)
    freq_grid, sigc_grid = torch.meshgrid(
        frequencies, sigma_c, indexing="ij"
    )  # both (|f|,|σ|)
    f_flat = freq_grid.reshape(-1)
    sc_flat = sigc_grid.reshape(-1)

    # Attenuation depends only on frequency but must broadcast across sigma as well
    att_flat = torch.exp(
        -2 * (np.pi**2) * f_flat**2 * position_smoothing_sigma**2
    )  # (K,)

    x_exp = x.unsqueeze(0).repeat(f_flat.shape[0], 1)  # (K, |x|)
    norm = 1.0 / torch.sqrt(2 * np.pi * sc_flat.unsqueeze(1))
    gauss = torch.exp(-(x_exp**2) / (2 * sc_flat.unsqueeze(1) ** 2))

    real = (
        norm
        * gauss
        * torch.cos(2 * np.pi * f_flat.unsqueeze(1) * x_exp)
        * att_flat.unsqueeze(1)
    )
    imag = (
        norm
        * gauss
        * torch.sin(2 * np.pi * f_flat.unsqueeze(1) * x_exp)
        * att_flat.unsqueeze(1)
    )

    conv_r, conv_i = _fft_convolve_real_imag(signal, real, imag)
    return _envelope_from_conv(
        conv_r, conv_i, real, imag, peak_enhancement, len(frequencies), len(sigmas)
    )


def compute_gabor_analytic(
    signal: torch.Tensor,
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigmas: torch.Tensor,
    *,
    peak_enhancement: float = 2.0,
    position_smoothing_sigma: float = 0.0,
):
    """
    Convolve `signal` with a bank of complex Gabors, *pre*-smoothed by a
    Gaussian of width `position_smoothing_sigma`, using a **single FFT**
    and an exact closed-form kernel.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if position_smoothing_sigma == 0:
        # shortcut to the raw Gabor path (no carrier scaling, no attenuation)
        return compute_gabor_old(
            signal,
            x,
            frequencies,
            sigmas,
            peak_enhancement,
            position_smoothing_sigma=0.0,
        )

    device, dtype = signal.device, signal.dtype
    σ = sigmas.to(device)  # (|σ|)
    σ_s = torch.tensor(position_smoothing_sigma, device=device, dtype=dtype)
    σ_c = torch.sqrt(σ**2 + σ_s**2)  # (|σ|)

    # ---------------- grids --------------------------------------------------
    f_grid, σ_grid = torch.meshgrid(
        frequencies.to(device), σ, indexing="ij"
    )  # each (|f|,|σ|)
    σc_grid = torch.sqrt(σ_grid**2 + σ_s**2)
    α_grid = σ_grid**2 / σc_grid**2
    amp_grid = (σ_grid / σc_grid) * torch.exp(
        -2 * (np.pi**2) * (f_grid**2) * (σ_grid**2) * (σ_s**2) / (σc_grid**2)
    )

    # ---------------- build kernel in space domain --------------------------
    K = f_grid.numel()  # total filters
    σc_flat = σc_grid.reshape(-1)  # (K,)
    αf_flat = (α_grid * f_grid).reshape(-1)  # (K,)
    amp_flat = amp_grid.reshape(-1)  # (K,)

    x_exp = x.to(device).unsqueeze(0).repeat(K, 1)  # (K,|x|)
    norm = 1.0 / torch.sqrt(2 * np.pi * σc_flat[:, None])
    gauss = torch.exp(-(x_exp**2) / (2 * σc_flat[:, None] ** 2))

    real = (
        amp_flat[:, None]
        * norm
        * gauss
        * torch.cos(2 * np.pi * αf_flat[:, None] * x_exp)
    )
    imag = (
        amp_flat[:, None]
        * norm
        * gauss
        * torch.sin(2 * np.pi * αf_flat[:, None] * x_exp)
    )

    # ---------------- convolution & envelope --------------------------------
    conv_r, conv_i = _fft_convolve_real_imag(signal, real, imag)
    env = _envelope_from_conv(
        conv_r, conv_i, real, imag, peak_enhancement, len(frequencies), len(sigmas)
    )
    return env
