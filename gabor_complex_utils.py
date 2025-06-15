import torch
import torch.nn.functional as F
import numpy as np


def compute_gabor_fitness_function(
    signal,
    x,
    frequencies,
    sigmas,
    phases,
    fitness_type="normalized_envelope",
    peak_enhancement=2.0,
    spatial_smoothing_sigma=0.0,
    freq_smoothing_sigma=0.0,
    scale_smoothing_sigma=0.0,
    position_smoothing_sigma=0.0,
):
    """
    Computes a convex fitness function for Gabor parameter optimization.

    The function peaks when Gabor filter parameters match signal components
    and is smooth/convex to avoid spurious local maxima.

    Args:
        signal: 1D input signal tensor
        x: 1D position tensor for Gabor filter generation
        frequencies: 1D tensor of frequencies to test
        sigmas: 1D tensor of sigma values to test
        phases: 1D tensor of phase values to test
        fitness_type: Type of fitness function:
            - "normalized_envelope": Normalized envelope energy (recommended)
            - "complex_envelope": Phase-invariant complex Gabor envelope
            - "peak_enhanced": Envelope with peak enhancement
        peak_enhancement: Power to raise fitness (>1 enhances peaks, reduces noise)
        spatial_smoothing_sigma: Apply Gaussian smoothing in position (0 = no smoothing)

    Returns:
        fitness: 4D tensor (num_frequencies, num_sigmas, num_phases, signal_length)
                Values are higher where Gabor parameters match signal components
    """

    if fitness_type == "complex_envelope":
        return _compute_complex_gabor_fitness(
            signal,
            x,
            frequencies,
            sigmas,
            peak_enhancement,
            freq_smoothing_sigma,
            scale_smoothing_sigma,
            position_smoothing_sigma,
        )
    elif fitness_type == "normalized_envelope":
        return _compute_normalized_envelope_fitness(
            signal,
            x,
            frequencies,
            sigmas,
            phases,
            peak_enhancement,
            spatial_smoothing_sigma,
        )
    elif fitness_type == "peak_enhanced":
        return _compute_peak_enhanced_fitness(
            signal,
            x,
            frequencies,
            sigmas,
            phases,
            peak_enhancement,
            spatial_smoothing_sigma,
        )
    else:
        raise ValueError(f"Unknown fitness_type: {fitness_type}")


def _compute_complex_gabor_fitness(
    signal,
    x,
    frequencies,
    sigmas,
    peak_enhancement=2.0,
    freq_smoothing_sigma=0.0,
    scale_smoothing_sigma=0.0,
    position_smoothing_sigma=0.0,
):
    """
    Phase-invariant fitness using complex Gabor filters.
    This is often the most convex option.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal = signal.to(device)
    x = x.to(device)
    frequencies = frequencies.to(device)
    sigmas = sigmas.to(device)

    signal_length = signal.shape[0]
    num_frequencies = frequencies.shape[0]
    num_sigmas = sigmas.shape[0]

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

    signal_padded = F.pad(signal, (0, fft_length - signal_length))
    gabor_real_padded = F.pad(gabor_real, (0, fft_length - kernel_size))
    gabor_imag_padded = F.pad(gabor_imag, (0, fft_length - kernel_size))

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

    # Spatial smoothing if requested
    # if position_smoothing_sigma > 0:
    #     envelope = _apply_spatial_smoothing(envelope, position_smoothing_sigma)

    # Reshape to include phase dimension (since this is phase-invariant, broadcast)
    # envelope = envelope.view(num_frequencies, num_sigmas, 1, signal_length)
    # Broadcast across phase dimension - all phases give same result
    # envelope = envelope.expand(-1, -1, 1, -1)

    # Reshape to 4D: (freq, sigma, phase=1, position)
    envelope = envelope.reshape(num_frequencies, num_sigmas, signal_length)
    envelope = envelope.unsqueeze(2)  # Add phase dimension

    # Apply smoothing if requested
    if (
        freq_smoothing_sigma > 0
        or scale_smoothing_sigma > 0
        or position_smoothing_sigma > 0
    ):
        envelope = smooth_envelope_3d(
            envelope,
            freq_smoothing_sigma,
            scale_smoothing_sigma,
            position_smoothing_sigma,
        )

    return envelope


def _compute_normalized_envelope_fitness(
    signal,
    x,
    frequencies,
    sigmas,
    phases,
    peak_enhancement=2.0,
    spatial_smoothing_sigma=0.0,
):
    """
    Normalized envelope fitness that tests all phase combinations.
    """
    # Get envelope from original function
    envelope = compute_full_conv_filtered_heatmap_fft(
        signal, x, frequencies, sigmas, phases
    )

    # Normalize by signal and filter energies
    signal_energy = torch.norm(signal)

    # Compute filter energies for normalization
    device = signal.device
    x = x.to(device)
    frequencies = frequencies.to(device)
    sigmas = sigmas.to(device)
    phases = phases.to(device)

    freq_grid, sigma_grid, phase_grid = torch.meshgrid(
        frequencies, sigmas, phases, indexing="ij"
    )
    freq_grid = freq_grid.reshape(-1)
    sigma_grid = sigma_grid.reshape(-1)
    phase_grid = phase_grid.reshape(-1)

    x_expanded = x.unsqueeze(0).repeat(len(freq_grid), 1)
    normalization = 1.0 / torch.sqrt(2 * np.pi * sigma_grid.unsqueeze(1))
    gaussians = torch.exp(-(x_expanded**2) / (2 * sigma_grid.unsqueeze(1) ** 2))
    sinusoids = torch.cos(
        2 * np.pi * freq_grid.unsqueeze(1) * x_expanded + phase_grid.unsqueeze(1)
    )
    gabor_kernels = normalization * gaussians * sinusoids

    filter_energies = torch.norm(gabor_kernels, dim=1).view(
        len(frequencies), len(sigmas), len(phases), 1
    )

    # Normalize envelope
    envelope = envelope / (signal_energy * filter_energies + 1e-8)

    # Peak enhancement
    if peak_enhancement != 1.0:
        envelope = torch.pow(envelope, peak_enhancement)

    # Spatial smoothing
    if spatial_smoothing_sigma > 0:
        envelope = _apply_spatial_smoothing(
            envelope.view(-1, envelope.shape[-1]), spatial_smoothing_sigma
        )
        envelope = envelope.view(len(frequencies), len(sigmas), len(phases), -1)

    return envelope


def _compute_peak_enhanced_fitness(
    signal,
    x,
    frequencies,
    sigmas,
    phases,
    peak_enhancement=2.0,
    spatial_smoothing_sigma=0.0,
):
    """
    Simple peak-enhanced envelope fitness.
    """
    envelope = compute_full_conv_filtered_heatmap_fft(
        signal, x, frequencies, sigmas, phases
    )

    # Peak enhancement
    if peak_enhancement != 1.0:
        envelope = torch.pow(envelope, peak_enhancement)

    # Spatial smoothing
    if spatial_smoothing_sigma > 0:
        envelope = _apply_spatial_smoothing(
            envelope.view(-1, envelope.shape[-1]), spatial_smoothing_sigma
        )
        envelope = envelope.view(len(frequencies), len(sigmas), len(phases), -1)

    return envelope


def _apply_spatial_smoothing(data, sigma):
    """
    Apply Gaussian smoothing along the spatial dimension.
    """
    if sigma <= 0:
        return data

    # Create 1D Gaussian kernel
    kernel_size = int(6 * sigma) + 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    x_kernel = torch.arange(kernel_size, dtype=data.dtype, device=data.device)
    x_kernel = x_kernel - kernel_size // 2
    kernel = torch.exp(-(x_kernel**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()

    # Apply convolution
    data_padded = F.pad(data, (kernel_size // 2, kernel_size // 2), mode="reflect")
    smoothed = F.conv1d(
        data_padded.unsqueeze(1), kernel.unsqueeze(0).unsqueeze(0), padding=0, groups=1
    )

    return smoothed.squeeze(1)


def smooth_envelope_3d(envelope, freq_sigma=0.0, scale_sigma=0.0, position_sigma=0.0):
    """
    Efficiently apply Gaussian smoothing to envelope along frequency, sigma, and position dimensions.
    Assumes smoothing will be applied - no dimension checks for maximum efficiency.

    Args:
        envelope: 4D tensor (num_frequencies, num_sigmas, num_phases, signal_length)
        freq_sigma: Smoothing sigma for frequency dimension (dim 0)
        scale_sigma: Smoothing sigma for scale/sigma dimension (dim 1)
        position_sigma: Smoothing sigma for position dimension (dim 3)

    Returns:
        Smoothed envelope tensor of same shape
    """
    if freq_sigma <= 0 and scale_sigma <= 0 and position_sigma <= 0:
        return envelope

    device = envelope.device
    dtype = envelope.dtype
    smoothed = envelope.contiguous()  # Ensure contiguous for efficient operations

    # Smooth along frequency dimension (dim 0)
    if freq_sigma > 0:
        kernel_size = int(6 * freq_sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        kernel = torch.exp(-(x**2) / (2 * freq_sigma**2))
        kernel = kernel / kernel.sum()

        # Reshape to (batch_size, freq) for efficient conv1d
        original_shape = smoothed.shape
        smoothed = smoothed.permute(1, 2, 3, 0).reshape(-1, original_shape[0])

        # Apply convolution with clamped padding
        pad_size = min(kernel_size // 2, original_shape[0] - 1)
        padded = F.pad(smoothed, (pad_size, pad_size), mode="reflect")
        smoothed = F.conv1d(
            padded.unsqueeze(1), kernel.view(1, 1, -1), padding=0
        ).squeeze(1)

        # Reshape back
        smoothed = smoothed.reshape(
            original_shape[1], original_shape[2], original_shape[3], original_shape[0]
        )
        smoothed = smoothed.permute(3, 0, 1, 2)

    # Smooth along sigma dimension (dim 1)
    if scale_sigma > 0:
        kernel_size = int(6 * scale_sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        kernel = torch.exp(-(x**2) / (2 * scale_sigma**2))
        kernel = kernel / kernel.sum()

        # Reshape to (batch_size, sigma) for efficient conv1d
        original_shape = smoothed.shape
        smoothed = smoothed.permute(0, 2, 3, 1).reshape(-1, original_shape[1])

        # Apply convolution with clamped padding
        pad_size = min(kernel_size // 2, original_shape[1] - 1)
        padded = F.pad(smoothed, (pad_size, pad_size), mode="reflect")
        smoothed = F.conv1d(
            padded.unsqueeze(1), kernel.view(1, 1, -1), padding=0
        ).squeeze(1)

        # Reshape back
        smoothed = smoothed.reshape(
            original_shape[0], original_shape[2], original_shape[3], original_shape[1]
        )
        smoothed = smoothed.permute(0, 3, 1, 2)

    # Smooth along position dimension (dim 3)
    if position_sigma > 0:
        kernel_size = int(6 * position_sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        x = torch.arange(kernel_size, dtype=dtype, device=device) - kernel_size // 2
        kernel = torch.exp(-(x**2) / (2 * position_sigma**2))
        kernel = kernel / kernel.sum()

        # Reshape to (batch_size, position) for efficient conv1d
        original_shape = smoothed.shape
        smoothed = smoothed.reshape(-1, original_shape[-1])

        # Apply convolution with clamped padding
        pad_size = min(kernel_size // 2, original_shape[-1] - 1)
        padded = F.pad(smoothed, (pad_size, pad_size), mode="reflect")
        smoothed = F.conv1d(
            padded.unsqueeze(1), kernel.view(1, 1, -1), padding=0
        ).squeeze(1)

        # Reshape back
        smoothed = smoothed.reshape(original_shape)

    return smoothed


def compute_full_conv_filtered_heatmap_fft(signal, x, frequencies, sigmas, phases):
    """
    Original envelope function from before - reused here.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    signal = signal.to(device)
    x = x.to(device)
    frequencies = frequencies.to(device)
    sigmas = sigmas.to(device)
    phases = phases.to(device)

    signal_length = signal.shape[0]
    num_frequencies = frequencies.shape[0]
    num_sigmas = sigmas.shape[0]
    num_phases = phases.shape[0]

    # Create meshgrid
    freq_grid, sigma_grid, phase_grid = torch.meshgrid(
        frequencies, sigmas, phases, indexing="ij"
    )
    freq_grid = freq_grid.reshape(-1)
    sigma_grid = sigma_grid.reshape(-1)
    phase_grid = phase_grid.reshape(-1)
    num_kernels = freq_grid.shape[0]

    # Generate Gabor filters
    x = x.unsqueeze(0).repeat(num_kernels, 1)
    normalization = 1.0 / torch.sqrt(2 * np.pi * sigma_grid.unsqueeze(1))
    gaussians = torch.exp(-(x**2) / (2 * sigma_grid.unsqueeze(1) ** 2))
    sinusoids = torch.cos(
        2 * np.pi * freq_grid.unsqueeze(1) * x + phase_grid.unsqueeze(1)
    )
    gabor_kernels = normalization * gaussians * sinusoids

    kernel_size = x.shape[1]
    total_length = signal_length + kernel_size - 1
    fft_length = 2 ** int(np.ceil(np.log2(total_length)))

    # Pad signal and kernels to fft_length
    signal_padded = F.pad(signal, (0, fft_length - signal_length))
    kernels_padded = F.pad(gabor_kernels, (0, fft_length - kernel_size))

    # FFT
    signal_fft = torch.fft.rfft(signal_padded, n=fft_length)
    kernel_fft = torch.fft.rfft(kernels_padded, n=fft_length)

    # Multiply in frequency domain
    conv_fft = kernel_fft * signal_fft.unsqueeze(0)
    full_conv = torch.fft.irfft(conv_fft, n=fft_length)

    # Extract center region
    start = (kernel_size - 1) // 2
    end = start + signal_length
    conv_result = full_conv[:, start:end]

    # Apply Hilbert transform to get envelope
    envelope = hilbert_envelope_fft(conv_result)

    return envelope.view(num_frequencies, num_sigmas, num_phases, signal_length)


def hilbert_envelope_fft(signals):
    """
    Compute envelope using Hilbert transform via FFT.
    """
    num_signals, signal_length = signals.shape
    device = signals.device

    fft_length = 2 ** int(np.ceil(np.log2(signal_length)))

    if fft_length > signal_length:
        signals_padded = F.pad(signals, (0, fft_length - signal_length))
    else:
        signals_padded = signals

    signal_fft = torch.fft.fft(signals_padded, dim=1)

    # Hilbert transform filter
    h = torch.zeros(fft_length, device=device, dtype=torch.float32)
    if fft_length % 2 == 0:
        h[0] = 1.0
        h[1 : fft_length // 2] = 2.0
        h[fft_length // 2] = 1.0
    else:
        h[0] = 1.0
        h[1 : (fft_length + 1) // 2] = 2.0

    analytic_fft = signal_fft * h.unsqueeze(0)
    analytic_signal = torch.fft.ifft(analytic_fft, dim=1)

    if fft_length > signal_length:
        analytic_signal = analytic_signal[:, :signal_length]

    envelope = torch.abs(analytic_signal)
    return envelope
