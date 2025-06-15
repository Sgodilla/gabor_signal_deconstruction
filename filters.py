import torch
import torch.nn.functional as F
import numpy as np


def apply_adaptive_lowpass_filter(
    conv_heatmap: torch.Tensor,
    frequencies: torch.Tensor,
    base_cutoff_freq: float = 0.1,
    frequency_scaling: float = 1.0,
):
    """
    Apply an adaptive low-pass filter to the full convolution tensor,
    where the cutoff frequency increases proportionally to the frequency layer.

    Parameters:
        conv_heatmap (torch.Tensor): 4D tensor of shape [freq, sigma, phase, position]
        frequencies (torch.Tensor): Frequencies tensor used for scaling the cutoff
        base_cutoff_freq (float): Base cutoff frequency for the lowest frequency layer (0.0-1.0)
        frequency_scaling (float): Factor controlling how much cutoff increases with frequency

    Returns:
        torch.Tensor: Filtered convolution tensor of the same shape
    """
    import numpy as np

    # Get dimensions of the convolution tensor
    num_freqs, num_sigmas, num_phases, signal_length = conv_heatmap.shape

    # Create a device-matched output tensor
    device = conv_heatmap.device
    filtered_heatmap = torch.zeros_like(conv_heatmap)

    # Normalize frequencies to 0-1 range for scaling purposes
    freq_min = frequencies.min()
    freq_max = frequencies.max()
    freq_norm = (frequencies - freq_min) / (freq_max - freq_min)

    # Apply filter to each frequency and phase slice
    for freq_idx in range(num_freqs):
        for phase_idx in range(num_phases):
            # Get the 2D slice for this frequency and phase
            slice_2d = conv_heatmap[
                freq_idx, :, phase_idx, :
            ]  # shape: [num_sigmas, signal_length]

            # Calculate adaptive cutoff frequency based on the current frequency layer
            # Higher frequency layers get higher cutoff frequencies
            cutoff = base_cutoff_freq + frequency_scaling * freq_norm[freq_idx]
            cutoff = torch.clamp(cutoff, max=0.99)  # Ensure cutoff stays below 1.0

            # Apply FFT along the position dimension
            slice_fft = torch.fft.rfft(slice_2d, dim=1)

            # Create a low-pass filter mask
            fft_length = slice_fft.shape[1]
            filter_mask = torch.ones(fft_length, device=device)

            # Calculate the cutoff point in FFT space
            cutoff_idx = int(cutoff.item() * fft_length)

            # Soft transition around cutoff for a smoother filter (Hann window)
            transition_width = max(1, int(0.1 * fft_length))
            for i in range(cutoff_idx, min(fft_length, cutoff_idx + transition_width)):
                # Apply a smooth cosine transition from 1 to 0
                t = (i - cutoff_idx) / transition_width
                filter_mask[i] = 0.5 * (1 + np.cos(np.pi * t))

            # Zero out frequencies above cutoff
            if cutoff_idx + transition_width < fft_length:
                filter_mask[cutoff_idx + transition_width :] = 0

            # Apply filter in frequency domain
            filtered_fft = slice_fft * filter_mask

            # Transform back to spatial domain
            filtered_slice = torch.fft.irfft(filtered_fft, n=signal_length, dim=1)

            # Store the filtered slice
            filtered_heatmap[freq_idx, :, phase_idx, :] = filtered_slice

    return filtered_heatmap


def remove_gabor_ripples(
    heatmap: torch.Tensor,
    frequencies: torch.Tensor,
    phase: float = 0.0,
    threshold: float = 0.1,
) -> torch.Tensor:
    """
    Remove rippling artifacts from a Gabor wavelet convolution heatmap using vectorized operations.
    """
    # Get dimensions
    batch_size, n_freqs, signal_length = heatmap.shape

    # Create normalized time domain
    signal_domain = torch.linspace(0, 1, signal_length, device=heatmap.device)

    # Initialize output tensor
    filtered_result = torch.zeros_like(heatmap)

    # Create all cosine waves at once
    # Shape: (n_freqs, 1, signal_length)
    cosine_waves = torch.zeros(n_freqs, 1, signal_length, device=heatmap.device)

    for i, freq in enumerate(frequencies):
        cosine_waves[i, 0, :] = torch.cos(2 * np.pi * freq * signal_domain + phase)

    # Create masks for near-zero values
    masks = 0.5 * (torch.tanh((abs(cosine_waves) - threshold) * 10) + 1)

    # Safety for division by zero
    epsilon = 1e-10
    denominators = cosine_waves + epsilon * torch.sign(cosine_waves)

    # Reshape heatmap for division
    # Original: (batch_size, n_freqs, signal_length)
    # Transposed: (n_freqs, batch_size, signal_length)
    heatmap_t = heatmap.transpose(0, 1)

    # Perform division with broadcasting
    result_t = heatmap_t / denominators

    # Apply masks
    result_t = result_t * masks

    # Transpose back to original shape
    filtered_result = result_t.transpose(0, 1)

    return filtered_result
