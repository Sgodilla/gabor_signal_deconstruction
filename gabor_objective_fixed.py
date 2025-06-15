import torch
import torch.nn.functional as F
import numpy as np


# Fix 1: Use reflection padding for smoother boundaries
def _apply_fft_smoothing_along_dim_reflected(
    data: torch.Tensor, sigma: float, dim: int
) -> torch.Tensor:
    """FFT‑based Gaussian smoothing with reflection padding to reduce edge artifacts."""
    if sigma <= 0:
        return data

    device, dtype = data.device, data.dtype
    dim_size = data.shape[dim]

    ksize = int(6 * sigma) + 1
    if ksize % 2 == 0:
        ksize += 1

    pad_size = ksize // 2

    # Manual reflection padding since PyTorch's F.pad can be problematic
    def reflect_pad_1d(tensor_1d, pad_left, pad_right):
        """Apply reflection padding to a 1D tensor."""
        if pad_left > 0:
            left_pad = tensor_1d[1 : pad_left + 1].flip(0)
            tensor_1d = torch.cat([left_pad, tensor_1d], dim=0)
        if pad_right > 0:
            right_pad = tensor_1d[-(pad_right + 1) : -1].flip(0)
            tensor_1d = torch.cat([tensor_1d, right_pad], dim=0)
        return tensor_1d

    # Apply reflection padding manually along the specified dimension
    if dim == 0:
        # Pad along first dimension (each row)
        padded_rows = []
        for i in range(data.shape[1]):
            row = data[:, i]
            padded_row = reflect_pad_1d(row, pad_size, pad_size)
            padded_rows.append(padded_row)
        data_padded = torch.stack(padded_rows, dim=1)
    elif dim == 1:
        # Pad along second dimension (each column)
        padded_cols = []
        for i in range(data.shape[0]):
            col = data[i, :]
            padded_col = reflect_pad_1d(col, pad_size, pad_size)
            padded_cols.append(padded_col)
        data_padded = torch.stack(padded_cols, dim=0)
    else:
        raise NotImplementedError(
            f"Reflection padding not implemented for dim={dim} with {data.ndim}D tensors"
        )

    # Now apply smoothing to the padded data
    dims = list(range(data_padded.ndim))
    dims[dim], dims[-1] = dims[-1], dims[dim]
    data_perm = data_padded.permute(dims)
    orig_shape = data_perm.shape
    flat = data_perm.reshape(-1, orig_shape[-1])

    # Create Gaussian kernel
    x = torch.arange(ksize, dtype=dtype, device=device) - ksize // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    total = orig_shape[-1] + ksize - 1
    nfft = 2 ** int(np.ceil(np.log2(total)))

    flat_pad = F.pad(flat, (0, nfft - orig_shape[-1]))
    kern_pad = F.pad(kernel, (0, nfft - ksize))

    flat_f = torch.fft.rfft(flat_pad, n=nfft)
    kern_f = torch.fft.rfft(kern_pad, n=nfft)
    conv = torch.fft.irfft(flat_f * kern_f.unsqueeze(0), n=nfft)

    start = (ksize - 1) // 2
    conv = conv[:, start : start + orig_shape[-1]]
    conv = conv.reshape(orig_shape)
    conv_unperm = conv.permute(dims)

    # Remove the padding to get back to original size
    if dim == 0:
        return conv_unperm[pad_size:-pad_size]
    elif dim == 1:
        return conv_unperm[:, pad_size:-pad_size]
    else:
        # General case for higher dimensions
        slices = [slice(None)] * conv_unperm.ndim
        slices[dim] = slice(pad_size, -pad_size)
        return conv_unperm[tuple(slices)]


# Fix 2: Build longer filters with smooth tapering
def _build_gabor_filters_with_tapering(x, frequencies, sigmas, taper_fraction=0.1):
    """Build Gabor filters with smooth tapering at edges to reduce artifacts."""
    freq_grid, sigma_grid = torch.meshgrid(frequencies, sigmas, indexing="ij")
    f_flat = freq_grid.reshape(-1)
    s_flat = sigma_grid.reshape(-1)
    x_exp = x.unsqueeze(0).repeat(f_flat.shape[0], 1)

    norm = 1.0 / torch.sqrt(2 * np.pi * s_flat.unsqueeze(1))
    gauss = torch.exp(-(x_exp**2) / (2 * s_flat.unsqueeze(1) ** 2))

    # Add Tukey window for smooth tapering
    L = x.shape[0]
    taper_len = int(taper_fraction * L)
    window = torch.ones_like(x)

    if taper_len > 0:
        # Left taper
        left_taper = 0.5 * (
            1
            + torch.cos(
                np.pi * (torch.arange(taper_len, device=x.device) / taper_len - 1)
            )
        )
        window[:taper_len] = left_taper

        # Right taper
        right_taper = 0.5 * (
            1 + torch.cos(np.pi * torch.arange(taper_len, device=x.device) / taper_len)
        )
        window[-taper_len:] = right_taper

    window_exp = window.unsqueeze(0).repeat(f_flat.shape[0], 1)

    real = (
        norm * gauss * torch.cos(2 * np.pi * f_flat.unsqueeze(1) * x_exp) * window_exp
    )
    imag = (
        norm * gauss * torch.sin(2 * np.pi * f_flat.unsqueeze(1) * x_exp) * window_exp
    )

    return real, imag


# Fix 3: Improved pre-smooth function using symmetric extension (most reliable)
def compute_gabor_pre_smooth_fixed(
    signal, x, frequencies, sigmas, peak_enhancement=2.0, position_smoothing_sigma=0.0
):
    """Fixed version using symmetric extension to reduce edge artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal, x = signal.to(device), x.to(device)
    frequencies, sigmas = frequencies.to(device), sigmas.to(device)

    filt_r, filt_i = _build_gabor_filters(x, frequencies, sigmas)

    if position_smoothing_sigma > 0:
        filt_r = _apply_fft_smoothing_along_dim_symmetric(
            filt_r, position_smoothing_sigma, dim=1
        )
        filt_i = _apply_fft_smoothing_along_dim_symmetric(
            filt_i, position_smoothing_sigma, dim=1
        )

    conv_r, conv_i = _fft_convolve_real_imag(signal, filt_r, filt_i)
    return _envelope_from_conv(
        conv_r, conv_i, filt_r, filt_i, peak_enhancement, len(frequencies), len(sigmas)
    )


# Fix 4: Simple symmetric extension (most reliable)
def _apply_fft_smoothing_along_dim_symmetric(
    data: torch.Tensor, sigma: float, dim: int
) -> torch.Tensor:
    """FFT‑based Gaussian smoothing with symmetric extension to reduce edge artifacts."""
    if sigma <= 0:
        return data

    device, dtype = data.device, data.dtype

    ksize = int(6 * sigma) + 1
    if ksize % 2 == 0:
        ksize += 1

    pad_size = ksize // 2

    # Simple symmetric extension: repeat edge values
    if dim == 1:
        # Extend each filter (row) symmetrically
        left_ext = data[:, :1].repeat(1, pad_size)  # repeat first column
        right_ext = data[:, -1:].repeat(1, pad_size)  # repeat last column
        data_extended = torch.cat([left_ext, data, right_ext], dim=1)
    elif dim == 0:
        # Extend along first dimension
        top_ext = data[:1, :].repeat(pad_size, 1)
        bottom_ext = data[-1:, :].repeat(pad_size, 1)
        data_extended = torch.cat([top_ext, data, bottom_ext], dim=0)
    else:
        raise NotImplementedError(f"Symmetric extension not implemented for dim={dim}")

    # Apply smoothing
    dims = list(range(data_extended.ndim))
    dims[dim], dims[-1] = dims[-1], dims[dim]
    data_perm = data_extended.permute(dims)
    orig_shape = data_perm.shape
    flat = data_perm.reshape(-1, orig_shape[-1])

    # Create Gaussian kernel
    x = torch.arange(ksize, dtype=dtype, device=device) - ksize // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    total = orig_shape[-1] + ksize - 1
    nfft = 2 ** int(np.ceil(np.log2(total)))

    flat_pad = F.pad(flat, (0, nfft - orig_shape[-1]))
    kern_pad = F.pad(kernel, (0, nfft - ksize))

    flat_f = torch.fft.rfft(flat_pad, n=nfft)
    kern_f = torch.fft.rfft(kern_pad, n=nfft)
    conv = torch.fft.irfft(flat_f * kern_f.unsqueeze(0), n=nfft)

    start = (ksize - 1) // 2
    conv = conv[:, start : start + orig_shape[-1]]
    conv = conv.reshape(orig_shape)
    conv_unperm = conv.permute(dims)

    # Remove the extension to get back to original size
    if dim == 0:
        return conv_unperm[pad_size:-pad_size]
    elif dim == 1:
        return conv_unperm[:, pad_size:-pad_size]
    else:
        slices = [slice(None)] * conv_unperm.ndim
        slices[dim] = slice(pad_size, -pad_size)
        return conv_unperm[tuple(slices)]


# Fix 5: Even simpler - use constant extension
def compute_gabor_pre_smooth_simple_fix(
    signal, x, frequencies, sigmas, peak_enhancement=2.0, position_smoothing_sigma=0.0
):
    """Simplest fix using symmetric extension."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal, x = signal.to(device), x.to(device)
    frequencies, sigmas = frequencies.to(device), sigmas.to(device)

    filt_r, filt_i = _build_gabor_filters(x, frequencies, sigmas)

    if position_smoothing_sigma > 0:
        filt_r = _apply_fft_smoothing_along_dim_symmetric(
            filt_r, position_smoothing_sigma, dim=1
        )
        filt_i = _apply_fft_smoothing_along_dim_symmetric(
            filt_i, position_smoothing_sigma, dim=1
        )

    conv_r, conv_i = _fft_convolve_real_imag(signal, filt_r, filt_i)
    return _envelope_from_conv(
        conv_r, conv_i, filt_r, filt_i, peak_enhancement, len(frequencies), len(sigmas)
    )


# Fix 6: Alternative using tapered filters
def compute_gabor_pre_smooth_tapered(
    signal,
    x,
    frequencies,
    sigmas,
    peak_enhancement=2.0,
    position_smoothing_sigma=0.0,
    taper_fraction=0.1,
):
    """Alternative version using tapered filters to reduce edge artifacts."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal, x = signal.to(device), x.to(device)
    frequencies, sigmas = frequencies.to(device), sigmas.to(device)

    filt_r, filt_i = _build_gabor_filters_with_tapering(
        x, frequencies, sigmas, taper_fraction
    )

    if position_smoothing_sigma > 0:
        # Use original smoothing function since filters are already tapered
        filt_r = _apply_fft_smoothing_along_dim(filt_r, position_smoothing_sigma, dim=1)
        filt_i = _apply_fft_smoothing_along_dim(filt_i, position_smoothing_sigma, dim=1)

    conv_r, conv_i = _fft_convolve_real_imag(signal, filt_r, filt_i)
    return _envelope_from_conv(
        conv_r, conv_i, filt_r, filt_i, peak_enhancement, len(frequencies), len(sigmas)
    )


# Helper functions (you'll need these from your original code)
def _fft_convolve_real_imag(signal, filt_r, filt_i):
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


def _apply_fft_smoothing_along_dim(data, sigma, dim):
    """Original smoothing function for comparison."""
    if sigma <= 0:
        return data

    device, dtype = data.device, data.dtype
    dim_size = data.shape[dim]

    ksize = int(6 * sigma) + 1
    if ksize % 2 == 0:
        ksize += 1
    ksize = min(ksize, dim_size)

    x = torch.arange(ksize, dtype=dtype, device=device) - ksize // 2
    kernel = torch.exp(-(x**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    dims = list(range(data.ndim))
    dims[dim], dims[-1] = dims[-1], dims[dim]
    data_perm = data.permute(dims)
    orig_shape = data_perm.shape
    flat = data_perm.reshape(-1, orig_shape[-1])

    total = orig_shape[-1] + ksize - 1
    nfft = 2 ** int(np.ceil(np.log2(total)))

    flat_pad = F.pad(flat, (0, nfft - orig_shape[-1]))
    kern_pad = F.pad(kernel, (0, nfft - ksize))

    flat_f = torch.fft.rfft(flat_pad, n=nfft)
    kern_f = torch.fft.rfft(kern_pad, n=nfft)
    conv = torch.fft.irfft(flat_f * kern_f.unsqueeze(0), n=nfft)

    start = (ksize - 1) // 2
    conv = conv[:, start : start + orig_shape[-1]]
    conv = conv.reshape(orig_shape)
    return conv.permute(dims)
