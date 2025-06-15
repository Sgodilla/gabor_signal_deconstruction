import itertools
import torch
import torch.nn.functional as F
import numpy as np


def create_gabor_wavelet(
    x: torch.Tensor,
    amplitude=1.0,
    shift=0.0,
    frequency=1.0,
    sigma=1.0,
    phase=0.0,
    normalize=False,
):
    """
    Generate 1D Gabor wavelet.

    A Gabor wavelet is a sinusoidal plane wave modulated by a Gaussian envelope.
    This function computes the wavelet based on the provided parameters.

    Parameters:
        x (torch.Tensor): Input tensor representing the domain over which to compute the wavelet.
        amplitude (float, optional): Peak amplitude of the wavelet. Defaults to 1.0.
        shift (float, optional): Horizontal shift of the wavelet's center. Defaults to 0.0.
        frequency (float, optional): Frequency of the sinusoidal component. Defaults to 1.0.
        sigma (float, optional): Standard deviation of the Gaussian envelope. Defaults to 1.0.
        phase (float, optional): Phase offset of the sinusoidal component, in radians. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed Gabor wavelet as a tensor.

    """
    normalization = 1 / np.sqrt(2 * np.pi * sigma) if normalize else 1
    gaussian = torch.exp(-((x - shift) ** 2) / (2 * sigma**2))
    cosine = torch.cos(2 * np.pi * frequency * (x - shift) + phase)
    gabor_wavelet = normalization * amplitude * gaussian * cosine
    return gabor_wavelet


def compute_conv_frequency_heatmap(
    signal: torch.Tensor,
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigma: float = 1.0,
    phase: float = 0.0,
) -> torch.Tensor:
    """
    Convolve a 1D signal with a bank of Gabor wavelets at varying frequencies.

    Parameters:
        signal (torch.Tensor): Input signal tensor of shape (signal_length,).
        x (torch.Tensor): 1D tensor representing the domain over which to compute the wavelets.
        frequencies (torch.Tensor): 1D tensor of frequencies for the Gabor wavelets.
        sigma (float, optional): Standard deviation of the Gaussian envelope. Defaults to 1.0.
        phase (float, optional): Phase offset of the sinusoidal component, in radians. Defaults to 0.0.

    Returns:
        torch.Tensor: A tensor of shape (num_frequencies, signal_length) containing the convolution results.
    """
    # Ensure signal is 1D
    if signal.dim() != 1:
        raise ValueError("Input signal must be a 1D tensor.")

    # Reshape signal to (1, 1, signal_length) for convolution
    signal = signal.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, signal_length)

    # Determine kernel size (should be odd to maintain symmetry)
    kernel_size = x.numel()
    if kernel_size % 2 == 0:
        raise ValueError("Length of x must be odd to ensure 'same' padding.")

    # Compute padding to maintain the same output length
    padding = kernel_size // 2

    # Create Gabor filters for each frequency
    filters = []
    for freq in frequencies:
        gaussian = torch.exp(-((x) ** 2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * freq * x + phase)
        gabor = gaussian * sinusoid
        filters.append(gabor)

    # Stack filters and reshape to (num_filters, 1, kernel_size)
    filters_tensor = torch.stack(filters).unsqueeze(
        1
    )  # Shape: (num_filters, 1, kernel_size)

    num_filters = filters_tensor.size(0)

    # Repeat the signal for each filter
    signal_repeated = signal.repeat(
        1, num_filters, 1
    )  # Shape: (1, num_filters, signal_length)

    # Perform grouped convolution
    conv_result = F.conv1d(
        signal_repeated, filters_tensor, padding=padding, groups=filters_tensor.size(0)
    )

    # Remove batch dimension and return result
    return conv_result.squeeze(1)  # Shape: (num_filters, signal_length)


def compute_conv_sigma_heatmap(
    signal: torch.Tensor,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> torch.Tensor:
    """
    Convolve a 1D signal with a bank of Gabor wavelets at varying sigmas.

    Parameters:
        signal (torch.Tensor): Input signal tensor of shape (signal_length,).
        x (torch.Tensor): 1D tensor representing the domain over which to compute the wavelets.
        sigmas (torch.Tensor): 1D tensor of sigma for the Gabor wavelets.
        frequencies (float, optional): Frequncy of the Gaussian envelope. Defaults to 1.0.
        phase (float, optional): Phase offset of the sinusoidal component, in radians. Defaults to 0.0.

    Returns:
        torch.Tensor: A tensor of shape (num_frequencies, signal_length) containing the convolution results.
    """
    # Ensure signal is 1D
    if signal.dim() != 1:
        raise ValueError("Input signal must be a 1D tensor.")

    # Reshape signal to (1, 1, signal_length) for convolution
    signal = signal.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, signal_length)

    # Determine kernel size (should be odd to maintain symmetry)
    kernel_size = x.numel()
    if kernel_size % 2 == 0:
        raise ValueError("Length of x must be odd to ensure 'same' padding.")

    # Compute padding to maintain the same output length
    padding = kernel_size // 2

    # Create Gabor filters for each frequency
    filters = []
    for sigma in sigmas:
        normalization = 1 / np.sqrt(2 * np.pi * sigma)
        gaussian = torch.exp(-((x) ** 2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * x + phase)
        gabor = normalization * gaussian * sinusoid
        filters.append(gabor)

    # Stack filters and reshape to (num_filters, 1, kernel_size)
    filters_tensor = torch.stack(filters).unsqueeze(
        1
    )  # Shape: (num_filters, 1, kernel_size)

    num_filters = filters_tensor.size(0)

    # Repeat the signal for each filter
    signal_repeated = signal.repeat(
        1, num_filters, 1
    )  # Shape: (1, num_filters, signal_length)

    # Perform grouped convolution
    conv_result = F.conv1d(
        signal_repeated, filters_tensor, padding=padding, groups=filters_tensor.size(0)
    )

    # Remove batch dimension and return result
    return conv_result.squeeze(1)  # Shape: (num_filters, signal_length)


def compute_conv_sigma_heatmap_fft(
    signal: torch.Tensor,
    x: torch.Tensor,
    sigmas: torch.Tensor,
    frequency: float = 1.0,
    phase: float = 0.0,
) -> torch.Tensor:
    """
    Convolve a 1D signal with a bank of Gabor wavelets at varying sigmas using FFT.

    Parameters:
        signal (torch.Tensor): Input signal tensor of shape (signal_length,).
        x (torch.Tensor): 1D tensor representing the domain over which to compute the wavelets.
        sigmas (torch.Tensor): 1D tensor of sigma for the Gabor wavelets.
        frequency (float, optional): Frequency of the Gaussian envelope. Defaults to 1.0.
        phase (float, optional): Phase offset of the sinusoidal component, in radians. Defaults to 0.0.

    Returns:
        torch.Tensor: A tensor of shape (num_sigmas, signal_length) containing the convolution results.
    """
    # Ensure signal is 1D
    if signal.dim() != 1:
        raise ValueError("Input signal must be a 1D tensor.")

    # Determine kernel size (should be odd to maintain symmetry)
    kernel_size = x.numel()
    if kernel_size % 2 == 0:
        raise ValueError("Length of x must be odd to ensure 'same' padding.")

    signal_length = signal.shape[0]
    num_sigmas = sigmas.shape[0]

    # Create Gabor filters for each sigma
    filters = []
    for sigma in sigmas:
        normalization = 1 / np.sqrt(2 * np.pi * sigma)
        gaussian = torch.exp(-((x) ** 2) / (2 * sigma**2))
        sinusoid = torch.cos(2 * np.pi * frequency * x + phase)
        gabor = normalization * gaussian * sinusoid
        filters.append(gabor)

    # Stack filters
    filters_tensor = torch.stack(filters)  # Shape: (num_sigmas, kernel_size)

    # Calculate FFT size (power of 2 for efficiency)
    total_length = signal_length + kernel_size - 1
    fft_length = 2 ** int(np.ceil(np.log2(total_length)))

    # Pad signal and filters to fft_length
    signal_padded = F.pad(signal, (0, fft_length - signal_length))
    filters_padded = F.pad(filters_tensor, (0, fft_length - kernel_size))

    # Compute FFT
    signal_fft = torch.fft.rfft(signal_padded, n=fft_length)
    filters_fft = torch.fft.rfft(filters_padded, n=fft_length)

    # Multiply in frequency domain (equivalent to convolution in time domain)
    conv_fft = filters_fft * signal_fft.unsqueeze(0)

    # Inverse FFT to get back to time domain
    conv_result = torch.fft.irfft(conv_fft, n=fft_length)

    # Extract the relevant part corresponding to the original signal length
    # For 'same' padding, we need to take the center portion
    start = (kernel_size - 1) // 2
    end = start + signal_length
    result = conv_result[:, start:end]  # Shape: (num_sigmas, signal_length)
    result = result.unsqueeze(0)

    return result


def compute_full_conv_heatmap_fft_v2(signal, x, frequencies, sigmas, phases):
    """
    Compute the 4D convolution heatmap of a signal with Gabor wavelets using FFT.
    This version creates Gabor filters using for loops for clarity.

    Parameters:
        signal (torch.Tensor): Input signal of shape (signal_length,).
        x (torch.Tensor): 1D tensor representing the domain over which to compute the wavelets.
        frequencies (torch.Tensor): 1D tensor of frequencies.
        sigmas (torch.Tensor): 1D tensor of sigma values.
        phases (torch.Tensor): 1D tensor of phase values.

    Returns:
        torch.Tensor: 4D tensor of shape (num_frequencies, num_sigmas, num_phases, signal_length).
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
    kernel_size = x.numel()

    # List to store all Gabor kernels
    filters = []
    # Generate Gabor filters with nested for loops
    for frequency in frequencies:
        for sigma in sigmas:
            for phase in phases:
                normalization = 1 / np.sqrt(2 * np.pi * sigma.cpu())
                gaussian = torch.exp(-((x) ** 2) / (2 * sigma**2))
                sinusoid = torch.cos(2 * np.pi * frequency * x + phase)
                gabor = normalization * gaussian * sinusoid
                filters.append(gabor)

    # Stack filters
    filters_tensor = torch.stack(filters)  # Shape: (num_sigmas, kernel_size)

    # Calculate FFT size (power of 2 for efficiency)
    total_length = signal_length + kernel_size - 1
    fft_length = 2 ** int(np.ceil(np.log2(total_length)))

    # Pad signal and filters to fft_length
    signal_padded = F.pad(signal, (0, fft_length - signal_length))
    filters_padded = F.pad(filters_tensor, (0, fft_length - kernel_size))

    # Compute FFT
    signal_fft = torch.fft.rfft(signal_padded, n=fft_length)
    filters_fft = torch.fft.rfft(filters_padded, n=fft_length)

    # Multiply in frequency domain (equivalent to convolution in time domain)
    conv_fft = filters_fft * signal_fft.unsqueeze(0)

    # Inverse FFT to get back to time domain
    conv_result = torch.fft.irfft(conv_fft, n=fft_length)

    # Extract center region (same size as signal)
    start = (kernel_size - 1) // 2
    end = start + signal_length
    result = conv_result[:, start:end]  # shape: (num_kernels, signal_length)

    # Reshape to 4D tensor
    return result.view(num_frequencies, num_sigmas, num_phases, signal_length)


def compute_full_conv_heatmap_fft(signal, x, frequencies, sigmas, phases):
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
    x = x.unsqueeze(0).repeat(num_kernels, 1)  # (num_kernels, kernel_size)
    normalization = 1.0 / torch.sqrt(2 * np.pi * sigma_grid.unsqueeze(1))
    gaussians = torch.exp(-(x**2) / (2 * sigma_grid.unsqueeze(1) ** 2))
    sinusoids = torch.cos(
        2 * np.pi * freq_grid.unsqueeze(1) * x + phase_grid.unsqueeze(1)
    )
    gabor_kernels = normalization * gaussians * sinusoids  # (num_kernels, kernel_size)

    kernel_size = x.shape[1]
    total_length = signal_length + kernel_size - 1
    fft_length = 2 ** int(np.ceil(np.log2(total_length)))  # next power of 2 for FFT

    # Pad signal and kernels to fft_length
    signal_padded = F.pad(signal, (0, fft_length - signal_length))
    kernels_padded = F.pad(gabor_kernels, (0, fft_length - kernel_size))

    # FFT
    signal_fft = torch.fft.rfft(signal_padded, n=fft_length)
    kernel_fft = torch.fft.rfft(kernels_padded, n=fft_length)

    # Multiply in frequency domain
    conv_fft = kernel_fft * signal_fft.unsqueeze(0)
    full_conv = torch.fft.irfft(
        conv_fft, n=fft_length
    )  # shape: (num_kernels, fft_length)

    # Extract center region (same size as signal)
    start = (kernel_size - 1) // 2
    end = start + signal_length
    conv_result = full_conv[:, start:end]  # shape: (num_kernels, signal_length)

    return conv_result.view(num_frequencies, num_sigmas, num_phases, signal_length)


def compute_full_conv_filtered_heatmap_fft(signal, x, frequencies, sigmas, phases):
    """
    Convolves a 1D signal with a bank of Gabor filters and returns the envelope
    of the convolved signals using Hilbert transform via FFT.

    Args:
        signal: 1D input signal tensor
        x: 1D position tensor for Gabor filter generation
        frequencies: 1D tensor of frequencies
        sigmas: 1D tensor of sigma values
        phases: 1D tensor of phase values

    Returns:
        4D tensor of shape (num_frequencies, num_sigmas, num_phases, signal_length)
        containing the envelope of each convolved signal
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
    x = x.unsqueeze(0).repeat(num_kernels, 1)  # (num_kernels, kernel_size)
    normalization = 1.0 / torch.sqrt(2 * np.pi * sigma_grid.unsqueeze(1))
    gaussians = torch.exp(-(x**2) / (2 * sigma_grid.unsqueeze(1) ** 2))
    sinusoids = torch.cos(
        2 * np.pi * freq_grid.unsqueeze(1) * x + phase_grid.unsqueeze(1)
    )
    gabor_kernels = normalization * gaussians * sinusoids  # (num_kernels, kernel_size)

    kernel_size = x.shape[1]
    total_length = signal_length + kernel_size - 1
    fft_length = 2 ** int(np.ceil(np.log2(total_length)))  # next power of 2 for FFT

    # Pad signal and kernels to fft_length
    signal_padded = F.pad(signal, (0, fft_length - signal_length))
    kernels_padded = F.pad(gabor_kernels, (0, fft_length - kernel_size))

    # FFT for convolution
    signal_fft = torch.fft.rfft(signal_padded, n=fft_length)
    kernel_fft = torch.fft.rfft(kernels_padded, n=fft_length)

    # Multiply in frequency domain
    conv_fft = kernel_fft * signal_fft.unsqueeze(0)
    full_conv = torch.fft.irfft(
        conv_fft, n=fft_length
    )  # shape: (num_kernels, fft_length)

    # Extract center region (same size as signal)
    start = (kernel_size - 1) // 2
    end = start + signal_length
    conv_result = full_conv[:, start:end]  # shape: (num_kernels, signal_length)

    # Apply Hilbert transform to get envelope
    envelope = hilbert_envelope_fft(conv_result)

    return envelope.view(num_frequencies, num_sigmas, num_phases, signal_length)


def hilbert_envelope_fft(signals):
    """
    Compute the envelope of signals using Hilbert transform via FFT.

    Args:
        signals: 2D tensor of shape (num_signals, signal_length)

    Returns:
        2D tensor of the same shape containing the envelope of each signal
    """
    num_signals, signal_length = signals.shape
    device = signals.device

    # Choose FFT length (next power of 2 for efficiency)
    fft_length = 2 ** int(np.ceil(np.log2(signal_length)))

    # Pad signals if necessary
    if fft_length > signal_length:
        signals_padded = F.pad(signals, (0, fft_length - signal_length))
    else:
        signals_padded = signals

    # Take FFT
    signal_fft = torch.fft.fft(signals_padded, dim=1)

    # Create Hilbert transform filter
    h = torch.zeros(fft_length, device=device, dtype=torch.float32)
    if fft_length % 2 == 0:
        # Even length
        h[0] = 1.0  # DC component
        h[1 : fft_length // 2] = 2.0  # Positive frequencies
        h[fft_length // 2] = 1.0  # Nyquist frequency
        # Negative frequencies remain zero
    else:
        # Odd length
        h[0] = 1.0  # DC component
        h[1 : (fft_length + 1) // 2] = 2.0  # Positive frequencies
        # Negative frequencies remain zero

    # Apply Hilbert filter
    analytic_fft = signal_fft * h.unsqueeze(0)  # Broadcasting

    # Inverse FFT to get analytic signal
    analytic_signal = torch.fft.ifft(analytic_fft, dim=1)

    # Extract original length if we padded
    if fft_length > signal_length:
        analytic_signal = analytic_signal[:, :signal_length]

    # Compute envelope (magnitude of analytic signal)
    envelope = torch.abs(analytic_signal)

    return envelope


def compute_adaptive_lowpass_filtered_heatmap(signal, x, frequencies, sigmas, phases):
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
    x_rep = x.unsqueeze(0).repeat(num_kernels, 1)
    normalization = 1.0 / torch.sqrt(2 * np.pi * sigma_grid.unsqueeze(1))
    gaussians = torch.exp(-(x_rep**2) / (2 * sigma_grid.unsqueeze(1) ** 2))
    sinusoids = torch.cos(
        2 * np.pi * freq_grid.unsqueeze(1) * x_rep + phase_grid.unsqueeze(1)
    )
    gabor_kernels = normalization * gaussians * sinusoids

    kernel_size = x.shape[0]
    total_length = signal_length + kernel_size - 1
    fft_length = 2 ** int(np.ceil(np.log2(total_length)))

    # Pad
    signal_padded = F.pad(signal, (0, fft_length - signal_length))
    kernels_padded = F.pad(gabor_kernels, (0, fft_length - kernel_size))

    # FFT convolution
    signal_fft = torch.fft.rfft(signal_padded, n=fft_length)
    kernel_fft = torch.fft.rfft(kernels_padded, n=fft_length)
    conv_fft = kernel_fft * signal_fft.unsqueeze(0)
    full_conv = torch.fft.irfft(conv_fft, n=fft_length)

    # Extract center portion
    start = (kernel_size - 1) // 2
    end = start + signal_length
    conv_result = full_conv[:, start:end]  # (num_kernels, signal_length)

    # Low-pass filter in position domain
    pos_fft = torch.fft.rfft(conv_result, n=signal_length)
    dx = x[1] - x[0]
    freqs_pos = torch.fft.rfftfreq(signal_length, d=dx.item()).to(device)

    # Create per-kernel mask in position domain
    # Use a more aggressive low-pass: e.g., cutoff = f_gabor / 2
    lowpass_cutoff = freq_grid / 1.3
    mask = (freqs_pos.unsqueeze(0) <= lowpass_cutoff.unsqueeze(1)).to(pos_fft.dtype)
    pos_fft_filtered = pos_fft * mask

    # Back to time domain
    result_filtered = torch.fft.irfft(pos_fft_filtered, n=signal_length)

    return result_filtered.view(num_frequencies, num_sigmas, num_phases, signal_length)


def compute_gaussian_smoothed_heatmap_fft_batched(
    signal, x, frequencies, sigmas, phases, smoothing=1.0, batch_size=16
):
    """
    Memory-efficient version that processes parameter combinations in batches.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move base data to device
    signal = signal.to(device)
    x = x.to(device)

    signal_length = signal.shape[0]
    num_frequencies = frequencies.shape[0]
    num_sigmas = sigmas.shape[0]
    num_phases = phases.shape[0]

    # Create output tensor on CPU to save GPU memory
    result = torch.zeros(
        (num_frequencies, num_sigmas, num_phases, signal_length),
        dtype=torch.float32,
        device="cpu",
    )

    # Process in batches of parameter combinations
    total_combinations = num_frequencies * num_sigmas * num_phases
    indices = list(
        itertools.product(range(num_frequencies), range(num_sigmas), range(num_phases))
    )

    for batch_start in range(0, total_combinations, batch_size):
        batch_end = min(batch_start + batch_size, total_combinations)
        batch_indices = indices[batch_start:batch_end]

        # Extract parameters for this batch
        batch_freq = torch.tensor(
            [frequencies[i] for i, _, _ in batch_indices], device=device
        )
        batch_sigma = torch.tensor(
            [sigmas[j] for _, j, _ in batch_indices], device=device
        )
        batch_phase = torch.tensor(
            [phases[k] for _, _, k in batch_indices], device=device
        )

        # Create Gabor kernels for this batch
        batch_size = len(batch_indices)
        x_rep = x.unsqueeze(0).repeat(batch_size, 1)
        normalization = 1.0 / torch.sqrt(2 * np.pi * batch_sigma.unsqueeze(1))
        gaussians = torch.exp(-(x_rep**2) / (2 * batch_sigma.unsqueeze(1) ** 2))
        sinusoids = torch.cos(
            2 * np.pi * batch_freq.unsqueeze(1) * x_rep + batch_phase.unsqueeze(1)
        )
        gabor_kernels = normalization * gaussians * sinusoids

        # Convolution via FFT
        kernel_size = x.shape[0]
        total_length = signal_length + kernel_size - 1

        # Use next power of 2 only if it's not much larger
        fft_length = total_length
        next_power_of_2 = 2 ** int(np.ceil(np.log2(total_length)))
        if next_power_of_2 < 1.5 * total_length:
            fft_length = next_power_of_2

        signal_padded = F.pad(signal, (0, fft_length - signal_length))
        kernels_padded = F.pad(gabor_kernels, (0, fft_length - kernel_size))

        signal_fft = torch.fft.rfft(signal_padded, n=fft_length)
        kernel_fft = torch.fft.rfft(kernels_padded, n=fft_length)

        # Free memory
        del kernels_padded
        torch.cuda.empty_cache()

        conv_fft = kernel_fft * signal_fft.unsqueeze(0)

        # Free memory
        del kernel_fft
        torch.cuda.empty_cache()

        full_conv = torch.fft.irfft(conv_fft, n=fft_length)

        # Free memory
        del conv_fft
        torch.cuda.empty_cache()

        # Crop valid portion
        start = (kernel_size - 1) // 2
        end = start + signal_length
        conv_result = full_conv[:, start:end]

        # Free memory
        del full_conv
        torch.cuda.empty_cache()

        # Compute smoothing sigmas based on wavelength
        wavelength = 1.0 / batch_freq.clamp(min=1e-6)
        sigma_smooths = smoothing * wavelength

        # Apply Gaussian smoothing
        batch_smoothed = apply_gaussian_smoothing_fft_batch_optimized(
            conv_result, sigma_smooths
        )

        # Place results in the correct position in the output tensor
        for idx, (i, j, k) in enumerate(batch_indices):
            result[i, j, k] = batch_smoothed[idx].cpu()

        # Free memory
        del batch_smoothed, conv_result
        torch.cuda.empty_cache()

    return result


def apply_gaussian_smoothing_fft_batch_optimized(
    conv_result: torch.Tensor, sigma_smooths: torch.Tensor
):
    """
    Memory-optimized version of Gaussian smoothing via FFT
    """
    device = conv_result.device
    dtype = conv_result.dtype
    num_kernels, signal_length = conv_result.shape

    # Instead of one large FFT, process each kernel separately or in small batches
    smoothed = torch.zeros_like(conv_result)

    # Process in mini-batches to save memory
    mini_batch_size = 4  # Adjust based on your GPU

    for i in range(0, num_kernels, mini_batch_size):
        end_idx = min(i + mini_batch_size, num_kernels)
        current_batch = conv_result[i:end_idx]
        current_sigmas = sigma_smooths[i:end_idx]

        # Individual kernel sizes for each signal
        kernel_sizes = (6 * current_sigmas + 1).int()
        kernel_sizes = kernel_sizes + (1 - kernel_sizes % 2)  # Ensure odd sizes

        for j, (signal, kernel_size, sigma) in enumerate(
            zip(current_batch, kernel_sizes, current_sigmas)
        ):
            # Only for this specific signal
            half = (kernel_size.item() - 1) // 2
            x_kernel = torch.arange(-half, half + 1, device=device, dtype=dtype)

            # Create kernel for this specific sigma
            kernel = torch.exp(-0.5 * (x_kernel / sigma) ** 2)
            kernel /= kernel.sum()  # Normalize

            # Calculate optimal FFT size - use just large enough
            total_length = signal_length + kernel_size.item() - 1
            fft_length = total_length
            next_power_of_2 = 2 ** int(np.ceil(np.log2(total_length)))
            if next_power_of_2 < 1.2 * total_length:
                fft_length = next_power_of_2

            # Pad signal and kernel
            signal_padded = F.pad(signal, (0, int(fft_length - signal_length)))
            kernel_padded = F.pad(kernel, (0, int(fft_length - kernel_size.item())))

            # FFT
            signal_fft = torch.fft.rfft(signal_padded)
            kernel_fft = torch.fft.rfft(kernel_padded)

            # Multiply in frequency domain
            result_fft = signal_fft * kernel_fft

            # Inverse FFT
            result = torch.fft.irfft(result_fft, n=fft_length)

            # Extract valid portion
            start = half
            end = start + signal_length
            smoothed[i + j] = result[start:end]

    return smoothed
