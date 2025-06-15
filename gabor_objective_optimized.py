import torch
import torch.nn.functional as F
import numpy as np


def compute_gabor_objective_fast(
    signal,
    x,
    frequencies,
    sigmas,
    peak_enhancement=2.0,
    position_smoothing_sigma=0.0,
):
    """
    Optimized Gabor objective using analytic composite kernel h = g * K_t
    (no separate smoothing step, single FFT-based convolution).

    Args:
        signal:       1D torch.Tensor of shape (L,)
        x:            1D torch.Tensor of positions for the kernel (K,)
        frequencies:  1D torch.Tensor of frequencies (N_f,)
        sigmas:       1D torch.Tensor of scales  (N_s,)
        peak_enhancement:      exponent p
        position_smoothing_sigma: smoothing sigma σ_t

    Returns:
        fitness: 3D tensor (N_f, N_s, L)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    signal = signal.to(device)
    x = x.to(device)
    frequencies = frequencies.to(device)
    sigmas = sigmas.to(device)

    signal_length = signal.shape[0]
    num_frequencies = frequencies.shape[0]
    num_sigmas = sigmas.shape[0]

    # Create parameter grids
    freq_grid, sigma_grid = torch.meshgrid(frequencies, sigmas, indexing="ij")
    freq_grid = freq_grid.reshape(-1)
    sigma_grid = sigma_grid.reshape(-1)
    num_kernels = freq_grid.shape[0]

    # Generate complex Gabor filters (quadrature pair)
    x_expanded = x.unsqueeze(0).repeat(num_kernels, 1)
    normalization = 1.0 / torch.sqrt(2 * np.pi * sigma_grid.unsqueeze(1))
    gaussians = torch.exp(-(x_expanded**2) / (2 * sigma_grid.unsqueeze(1) ** 2))

    # Real and imaginary components
    real_part = gaussians * torch.cos(2 * np.pi * freq_grid.unsqueeze(1) * x_expanded)
    imag_part = gaussians * torch.sin(2 * np.pi * freq_grid.unsqueeze(1) * x_expanded)

    gabor_real = normalization * real_part
    gabor_imag = normalization * imag_part

    # FFT-based convolution
    kernel_size = x_expanded.shape[1]
    total_length = signal_length + kernel_size - 1
    fft_length = 2 ** int(np.ceil(np.log2(total_length)))

    # Pad signal and kernels
    signal_padded = F.pad(signal, (0, fft_length - signal_length))
    gabor_real_padded = F.pad(gabor_real, (0, fft_length - kernel_size))
    gabor_imag_padded = F.pad(gabor_imag, (0, fft_length - kernel_size))

    # FFT convolution
    signal_fft = torch.fft.rfft(signal_padded, n=fft_length)
    gabor_real_fft = torch.fft.rfft(gabor_real_padded, n=fft_length)
    gabor_imag_fft = torch.fft.rfft(gabor_imag_padded, n=fft_length)

    conv_real_fft = gabor_real_fft * signal_fft.unsqueeze(0)
    conv_imag_fft = gabor_imag_fft * signal_fft.unsqueeze(0)

    conv_real = torch.fft.irfft(conv_real_fft, n=fft_length)
    conv_imag = torch.fft.irfft(conv_imag_fft, n=fft_length)

    # Extract valid region
    start = (kernel_size - 1) // 2
    end = start + signal_length
    conv_real = conv_real[:, start:end]
    conv_imag = conv_imag[:, start:end]

    # Compute envelope (magnitude of complex response)
    envelope = torch.sqrt(conv_real**2 + conv_imag**2)

    # Normalize by filter energy for scale invariance
    filter_energy = torch.sqrt(
        torch.sum(gabor_real**2 + gabor_imag**2, dim=1, keepdim=True)
    )
    envelope = envelope / (filter_energy + 1e-8)

    # Peak enhancement
    if peak_enhancement != 1.0:
        envelope = torch.pow(envelope, peak_enhancement)

    # Reshape to 3D: (freq, sigma, position)
    envelope = envelope.reshape(num_frequencies, num_sigmas, signal_length)

    if position_smoothing_sigma > 0:
        smoothed = envelope.contiguous()
        smoothed = _apply_fft_smoothing_along_dim(
            smoothed, position_smoothing_sigma, dim=2
        )
        return smoothed
    else:
        return envelope


def _apply_fft_smoothing_along_dim(data, sigma, dim):
    """
    Apply FFT-based Gaussian smoothing along a specific dimension.

    Args:
        data: Input tensor
        sigma: Gaussian smoothing sigma
        dim: Dimension to smooth along

    Returns:
        Smoothed tensor
    """
    if sigma <= 0:
        return data

    device = data.device
    dtype = data.dtype

    # Get dimension size
    dim_size = data.shape[dim]

    # Create Gaussian kernel
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, dim_size)  # Don't exceed dimension size

    x_kernel = torch.arange(kernel_size, dtype=dtype, device=device)
    x_kernel = x_kernel - kernel_size // 2
    kernel = torch.exp(-(x_kernel**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Move target dimension to last position for efficient FFT processing
    dims = list(range(data.ndim))
    dims[dim], dims[-1] = dims[-1], dims[dim]
    data_permuted = data.permute(dims)

    # Reshape for batch processing
    original_shape = data_permuted.shape
    data_flat = data_permuted.reshape(-1, original_shape[-1])

    # FFT-based convolution
    signal_length = original_shape[-1]
    total_length = signal_length + kernel_size - 1
    fft_length = 2 ** int(np.ceil(np.log2(total_length)))

    # Pad data and kernel
    data_padded = F.pad(data_flat, (0, fft_length - signal_length))
    kernel_padded = F.pad(kernel, (0, fft_length - kernel_size))

    # FFT convolution
    data_fft = torch.fft.rfft(data_padded, n=fft_length)
    kernel_fft = torch.fft.rfft(kernel_padded, n=fft_length)

    conv_fft = data_fft * kernel_fft.unsqueeze(0)
    conv_result = torch.fft.irfft(conv_fft, n=fft_length)

    # Extract valid region (same size as input)
    start = (kernel_size - 1) // 2
    end = start + signal_length
    conv_result = conv_result[:, start:end]

    # Reshape back
    conv_result = conv_result.reshape(original_shape)

    # Permute back to original dimension order
    conv_result = conv_result.permute(dims)

    return conv_result
