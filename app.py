import streamlit as st
import time
import torch
import numpy as np
from gabor_objective import compute_gabor_objective
from gabor_objective_fixed import (
    compute_gabor_pre_smooth_fixed,
    compute_gabor_pre_smooth_simple_fix,
    compute_gabor_pre_smooth_tapered,
)
from gabor_objective_optimized import compute_gabor_objective_fast
from gabor_objective_tests import (
    compute_gabor_analytic,
    compute_gabor_old,
    compute_gabor_optimized,
    compute_gabor_pre_smooth,
)
from gabor_utils import (
    compute_conv_frequency_heatmap,
    compute_full_conv_heatmap_fft,
)
from gabor_visualizer import (
    visualize_conv_heatmap,
    visualize_full_conv_slice,
    visualize_gabor_wavelet,
    visualize_sigma_conv_heatmap,
    visualize_signal,
)
from filters import remove_gabor_ripples
from objective_visualizer import (
    visualize_gabor_objective_slice,
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"Using device: {device}")

gabor_1 = visualize_gabor_wavelet("Gabor Wavelet 1", "c-")
st.html("</br>")
gabor_2 = visualize_gabor_wavelet("Gabor Wavelet 2", "y-")
st.html("</br>")
gabor_3 = visualize_gabor_wavelet("Gabor Wavelet 3", "m-")
st.html("</br>")
signal = visualize_signal([gabor_1, gabor_2, gabor_3])
st.html("</br>")

n = signal.shape[0]
x = torch.linspace(-10, 10, steps=n)

frequencies = torch.linspace(0.0, 10.0, steps=1000)

start = time.time()
frequency_conv_heatmap = compute_conv_frequency_heatmap(
    signal, x, frequencies, sigma=1.0, phase=0.0
)
end = time.time()
st.write(f"Frequency Conv execution time: {end - start:.4f} seconds")

# Step 3: Apply the ripple removal filter
filtered_frequency_heatmap = remove_gabor_ripples(
    heatmap=frequency_conv_heatmap,
    frequencies=frequencies,
    phase=0.0,
)

# visualize_frequency_conv_heatmap(signal, "Frequency Conv")
visualize_conv_heatmap(
    x=x,
    variations=frequencies,
    conv_heatmap=frequency_conv_heatmap,
    var_label="Frequency",
)
st.html("</br>")

# visualize_conv_heatmap(
#     x=x,
#     variations=frequencies,
#     conv_heatmap=filtered_frequency_heatmap,
#     var_label="Filtered Frequency",
# )
# st.html("</br>")

visualize_sigma_conv_heatmap(signal, "Sigma Conv")
st.html("</br>")


# log_vals = torch.logspace(start=-4, end=0, steps=100)
# frequencies = 10.0 * (1.0 - log_vals / log_vals.max())
frequencies = torch.linspace(0.0, 10.0, steps=100)
sigmas = torch.linspace(0.01, 10.0, steps=10)
# phases = torch.linspace(0.01, 2 * np.pi, steps=8)
# phases = torch.linspace(0.01, 2 * np.pi, steps=12)
phases = torch.linspace(0.01, 2 * np.pi, steps=2)

start = time.time()
conv_heatmap = compute_full_conv_heatmap_fft(signal, x, frequencies, sigmas, phases)
end = time.time()
st.write(f"Full Conv execution time: {end - start:.4f} seconds")
print(f"\nFull Conv execution time: {end - start:.4f} seconds")

# start = time.time()
# filtered_conv_heatmap = compute_full_conv_filtered_heatmap_fft(
#     signal, x, frequencies, sigmas, phases
# )
# end = time.time()
# st.write(f"Full Filtered Conv execution time: {end - start:.4f} seconds")
# print(f"Full Filtered Conv execution time: {end - start:.4f} seconds")

# start = time.time()
# fitness_function = compute_gabor_objective(
#     signal,
#     x,
#     frequencies,
#     sigmas,
#     # phases,
#     # fitness_type="complex_envelope",
#     # fitness_type="normalized_envelope",
#     # fitness_type="peak_enhanced",
#     peak_enhancement=2.0,  # Enhances peaks vs noise
#     freq_smoothing_sigma=0.0,  # Reduces spurious maxima
#     scale_smoothing_sigma=0.0,  # Reduces spurious maxima
#     position_smoothing_sigma=20.0,  # Reduces spurious maxima
# )
# end = time.time()
# st.write(f"Fitness execution time: {end - start:.4f} seconds")
# print(f"Fitness execution time: {end - start:.4f} seconds\n")

start = time.time()
fitness_function_old = compute_gabor_old(
    signal,
    x,
    frequencies,
    sigmas,
    peak_enhancement=2.0,  # Enhances peaks vs noise
    position_smoothing_sigma=20.0,  # Reduces spurious maxima
)
end = time.time()
st.write(f"Old fitness execution time: {end - start:.4f} seconds")
print(f"Old fitness execution time: {end - start:.4f} seconds\n")

start = time.time()
fitness_function_pre_smooth = compute_gabor_pre_smooth(
    signal,
    x,
    frequencies,
    sigmas,
    peak_enhancement=2.0,  # Enhances peaks vs noise
    position_smoothing_sigma=20.0,  # Reduces spurious maxima
)
end = time.time()
st.write(f"Pre-smoooth fitness execution time: {end - start:.4f} seconds")
print(f"Pre-smoooth fitness execution time: {end - start:.4f} seconds\n")

start = time.time()
fitness_function_optimized = compute_gabor_optimized(
    signal,
    x,
    frequencies,
    sigmas,
    peak_enhancement=2.0,  # Enhances peaks vs noise
    position_smoothing_sigma=20.0,  # Reduces spurious maxima
)
end = time.time()
st.write(f"Optimized fitness execution time: {end - start:.4f} seconds")
print(f"Optimized fitness execution time: {end - start:.4f} seconds\n")

start = time.time()
fitness_function_analytic = compute_gabor_analytic(
    signal,
    x,
    frequencies,
    sigmas,
    peak_enhancement=2.0,  # Enhances peaks vs noise
    position_smoothing_sigma=20.0,  # Reduces spurious maxima
)
end = time.time()
st.write(f"Analytic fitness execution time: {end - start:.4f} seconds")
print(f"Analytic fitness execution time: {end - start:.4f} seconds\n")

# start = time.time()
# filtered_conv_heatmap = apply_adaptive_lowpass_filter(conv_heatmap, frequencies)
# end = time.time()
# st.write(f"Filter execution time: {end - start:.4f} seconds")


visualize_full_conv_slice(x, frequencies, sigmas, phases, conv_heatmap, "Full Conv")

# visualize_gabor_objective_slice(
#     x, frequencies, sigmas, fitness_function, "Gabor Fitness Function"
# )

visualize_gabor_objective_slice(
    x,
    frequencies,
    sigmas,
    fitness_function_old,
    "Gabor Fitness Function - Old",
)

visualize_gabor_objective_slice(
    x,
    frequencies,
    sigmas,
    fitness_function_pre_smooth,
    "Gabor Fitness Function - Pre-smoooth",
)

visualize_gabor_objective_slice(
    x,
    frequencies,
    sigmas,
    fitness_function_analytic,
    "Gabor Fitness Function - Analytic",
)

# visualize_gabor_objective_slice(
#     x,
#     frequencies,
#     sigmas,
#     fitness_function_optimized,
#     "Gabor Fitness Function - Optimized",
# )

# visualize_gabor_objective_volume(
#     x,
#     frequencies,
#     sigmas,
#     fitness_function_old,
#     "Gabor Fitness Function Optimized",
# )
