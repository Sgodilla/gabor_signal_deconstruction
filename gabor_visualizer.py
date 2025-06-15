import streamlit as st
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from gabor_utils import (
    compute_conv_frequency_heatmap,
    compute_conv_sigma_heatmap_fft,
    compute_full_conv_heatmap_fft,
    create_gabor_wavelet,
)
import plotly.graph_objects as go

n = 1001


def visualize_gabor_wavelet(title="Gabor Wavelet", linetype="c-"):
    """
    Visualizes a Gabor wavelet with variable parmeters

    Paramers:
        title: Title of the Gabor Wavelet
        linetype: Style of the wavelet plot line

    Sliders:
        x: Input tensor representing the domain over which to compute the wavelet.
        amplitude: Peak amplitude of the wavelet. Defaults to 1.0.
        shift: Horizontal shift of the wavelet's center. Defaults to 0.0.
        frequency: Frequency of the sinusoidal component. Defaults to 1.0.
        sigma: Standard deviation of the Gaussian envelope. Defaults to 1.0.
        phase: Phase offset of the sinusoidal component, in radians. Defaults to 0.0.
    """

    plt.style.use("dark_background")

    # Layout: two columns
    col1, col2 = st.columns([1, 2])

    with col1:
        # Create sliders for each parameter
        amplitude = st.slider(
            "Amplitude",
            min_value=0.0,
            max_value=5.0,
            value=1.0,
            step=0.1,
            key=title + " amplitude",
        )
        shift = st.slider(
            "Shift",
            min_value=-10.0,
            max_value=10.0,
            value=0.0,
            step=0.1,
            key=title + " shift",
        )
        frequency = st.slider(
            "Frequency",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key=title + " frequency",
        )
        sigma = st.slider(
            "Sigma",
            min_value=0.01,
            max_value=10.0,
            value=1.0,
            step=0.1,
            key=title + " sigma",
        )
        phase = st.slider(
            "Phase (radians)",
            min_value=0.0,
            max_value=2 * np.pi,
            value=0.0,
            step=0.1,
            key=title + " phase",
        )

    with col2:
        st.markdown("## " + title)
        # Define the range of x values
        x = torch.linspace(-10, 10, steps=n)

        # Compute the Gabor wavelet
        gabor = create_gabor_wavelet(x, amplitude, shift, frequency, sigma, phase)

        # Plot the Gabor wavelet
        fig, ax = plt.subplots()
        ax.plot(x.numpy(), gabor.numpy(), linetype, linewidth=3)
        ax.grid(True, linewidth=0.5)

        # Display the plot in Streamlit
        st.pyplot(fig)

    return gabor


def visualize_signal(
    wavelets: list[torch.Tensor], title="Resulting Signal", linetype="w-", linewidth=2
):
    st.markdown("## " + title)
    # Define the range of x values
    x = torch.linspace(-10, 10, steps=n)

    # Compute the Gabor wavelet
    signal = torch.sum(torch.stack(wavelets), dim=0)

    # Plot the Gabor wavelet
    fig, ax = plt.subplots()
    ax.plot(x.numpy(), signal.numpy(), linetype, linewidth)
    ax.grid(True, linewidth=0.5)

    # Display the plot in Streamlit
    st.pyplot(fig)

    return signal


def visualize_conv_heatmap(
    x: torch.Tensor,
    variations: torch.Tensor,
    conv_heatmap: torch.Tensor,
    title="Full Convolution Frequency",
    var_label="Frequency",
):
    # Convert tensors to NumPy arrays for plotting
    heatmap_np = conv_heatmap.numpy()[0]
    x_np = x.numpy()
    variationss_np = variations.numpy()

    # Plot the conv heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(
        heatmap_np,
        extent=(x_np[0], x_np[-1], variationss_np[0], variationss_np[-1]),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xlabel("Position")
    ax.set_ylabel(var_label)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude")

    # Display the plot in Streamlit
    st.pyplot(fig)


def visualize_frequency_conv_heatmap(
    signal: torch.Tensor, title="Frequency Convolution"
):
    # Define the range of x values
    x = torch.linspace(-10, 10, steps=n)

    # Define frequencies to sweep
    frequencies = torch.linspace(0.0, 10.0, steps=50)

    # Compute the convolution heatmap
    heatmap = compute_conv_frequency_heatmap(
        signal, x, frequencies, sigma=1.0, phase=0.0
    )

    # Convert tensors to NumPy arrays for plotting
    heatmap_np = heatmap.numpy()[0]
    x_np = x.numpy()
    frequencies_np = frequencies.numpy()

    # Plot the conv heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(
        heatmap_np,
        extent=(x_np[0], x_np[-1], frequencies_np[0], frequencies_np[-1]),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Frequency")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude")

    # Display the plot in Streamlit
    st.pyplot(fig)


def visualize_sigma_conv_heatmap(signal: torch.Tensor, title="Frequency Convolution"):
    # Define the range of x values
    x = torch.linspace(-10, 10, steps=n)

    # Define frequencies to sweep
    sigmas = torch.linspace(0.0, 10.0, steps=50)

    # Add Frequency slider
    frequency = st.slider(
        "Frequency",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.1,
        key=title + " frequency",
    )

    # Compute the convolution heatmap
    start = time.time()
    heatmap = compute_conv_sigma_heatmap_fft(
        signal, x, sigmas, frequency=frequency, phase=0.0
    )
    end = time.time()
    st.write(f"Conv execution time: {end - start:.4f} seconds")

    # Convert tensors to NumPy arrays for plotting
    heatmap_np = heatmap.numpy()[0]
    x_np = x.numpy()
    sigmas_np = sigmas.numpy()

    # Plot the conv heatmap
    fig, ax = plt.subplots()
    im = ax.imshow(
        heatmap_np,
        extent=(x_np[0], x_np[-1], sigmas_np[0], sigmas_np[-1]),
        aspect="auto",
        origin="lower",
        cmap="viridis",
    )
    ax.set_xlabel("Position")
    ax.set_ylabel("Sigma")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Amplitude")

    # Display the plot in Streamlit
    st.pyplot(fig)


def visualize_full_conv_volume(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigmas: torch.Tensor,
    phases: torch.Tensor,
    conv_heatmap: torch.Tensor,
    title="Full Convolution Volume",
):
    # Convert to NumPy
    conv_np = torch.abs(conv_heatmap).cpu().numpy()
    x_np = x.cpu().numpy()
    freq_np = frequencies.cpu().numpy()
    sig_np = sigmas.cpu().numpy()
    phase_np = phases.cpu().numpy()

    # Select phase slice
    phase_index = st.slider("Select Phase Index", 0, len(phase_np) - 1, 0)
    st.write(f"Viewing Phase: {phase_np[phase_index]:.2f} radians")

    data = conv_np[:, :, phase_index, :]  # shape: (freq, sigma, x)

    # Normalize data for visibility
    # data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)

    # Generate meshgrid for coordinates
    F, S, X = np.meshgrid(freq_np, sig_np, x_np, indexing="ij")

    # Flatten everything for volume plot
    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=F.flatten(),
            z=S.flatten(),
            value=data.flatten(),
            isomin=float(data.max() * 0.1),
            # isomin=data.min(),
            isomax=data.max(),
            opacity=0.1,  # good balance between detail and visibility
            surface_count=15,
            colorscale="turbo",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    fig.update_layout(
        title=title + " — 3D Volume View",
        scene=dict(
            xaxis_title="Position (x)",
            yaxis_title="Frequency",
            zaxis_title="Sigma",
        ),
        width=900,
        height=700,
    )

    st.plotly_chart(fig)


def verify_convolution_consistency(signal, x, frequencies, sigmas, phases):
    """
    Verifies that for a specific frequency and phase, the full convolution function
    produces results identical to the sigma convolution function.

    Parameters:
        signal (torch.Tensor): Input signal tensor
        x (torch.Tensor): Domain over which to compute the wavelets
        frequencies (torch.Tensor): Frequencies to use in the convolution
        sigmas (torch.Tensor): Sigma values to use in the convolution
        phases (torch.Tensor): Phase values to use in the convolution
    """
    st.header("Convolution Consistency Verification")

    # Create selection widgets for frequency and phase
    col1, col2 = st.columns(2)

    with col1:
        freq_idx = st.slider(
            "Select Frequency Index",
            min_value=0,
            max_value=len(frequencies) - 1,
            value=len(frequencies) // 2,
        )
        selected_freq = frequencies[freq_idx].item()
        st.write(f"Selected Frequency: {selected_freq:.3f}")

    with col2:
        phase_idx = st.slider(
            "Select Phase Index",
            min_value=0,
            max_value=len(phases) - 1,
            value=0,
        )
        selected_phase = phases[phase_idx].item()
        st.write(f"Selected Phase: {selected_phase:.3f} radians")

    if st.button("Run Verification"):
        with st.spinner("Computing and comparing results..."):
            # Compute the full convolution (if not already cached)
            st.write("Computing full convolution...")
            start_time = time.time()
            full_conv_result = compute_full_conv_heatmap_fft(
                signal, x, frequencies, sigmas, phases
            )
            full_time = time.time() - start_time

            # Extract the slice for the selected frequency and phase
            full_conv_slice = full_conv_result[freq_idx, :, phase_idx, :]

            # Compute the sigma convolution using the specific frequency and phase
            st.write("Computing sigma convolution...")
            start_time = time.time()
            sigma_conv_result = compute_conv_sigma_heatmap_fft(
                signal, x, sigmas, frequency=selected_freq, phase=selected_phase
            )
            sigma_time = time.time() - start_time

            # Reshape sigma convolution result for comparison
            sigma_conv_reshaped = sigma_conv_result.squeeze(0)

            # Compute differences
            absolute_diff = torch.abs(full_conv_slice.cpu() - sigma_conv_reshaped)

            # Calculate metrics
            mse = torch.mean((sigma_conv_reshaped - full_conv_slice.cpu()) ** 2).item()
            max_abs_diff = torch.max(absolute_diff).item()
            mean_abs_diff = torch.mean(absolute_diff).item()

            # Display performance metrics
            st.subheader("Performance Metrics")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Full Convolution Time", f"{full_time:.4f} s")
                st.metric("Mean Absolute Difference", f"{mean_abs_diff:.8e}")
            with col2:
                st.metric("Sigma Convolution Time", f"{sigma_time:.4f} s")
                st.metric("Max Absolute Difference", f"{max_abs_diff:.8e}")

            st.metric("Mean Squared Error", f"{mse:.8e}")

            # Visualization comparison
            st.subheader("Visual Comparison")

            # Create plots to visualize the results
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            # Sigma convolution heatmap
            im1 = axes[0].imshow(
                sigma_conv_reshaped.numpy(),
                aspect="auto",
                origin="lower",
                extent=(x[0].item(), x[-1].item(), sigmas[0].item(), sigmas[-1].item()),
                cmap="viridis",
            )
            axes[0].set_title(
                f"Sigma Convolution\nFreq={selected_freq:.2f}, Phase={selected_phase:.2f}"
            )
            axes[0].set_xlabel("Position")
            axes[0].set_ylabel("Sigma")
            plt.colorbar(im1, ax=axes[0], label="Amplitude")

            # Full convolution slice
            im2 = axes[1].imshow(
                full_conv_slice.cpu().numpy(),
                aspect="auto",
                origin="lower",
                extent=(x[0].item(), x[-1].item(), sigmas[0].item(), sigmas[-1].item()),
                cmap="viridis",
            )
            axes[1].set_title(
                f"Full Convolution Slice\nFreq={selected_freq:.2f}, Phase={selected_phase:.2f}"
            )
            axes[1].set_xlabel("Position")
            axes[1].set_ylabel("Sigma")
            plt.colorbar(im2, ax=axes[1], label="Amplitude")

            # Difference heatmap (use a different colormap to emphasize differences)
            max_diff_value = absolute_diff.max().item()
            if max_diff_value > 0:
                vmax = max_diff_value
            else:
                vmax = 1e-8  # Avoid division by zero if no difference

            im3 = axes[2].imshow(
                absolute_diff.numpy(),
                aspect="auto",
                origin="lower",
                extent=(x[0].item(), x[-1].item(), sigmas[0].item(), sigmas[-1].item()),
                cmap="hot",
                vmin=0,
                vmax=vmax,
            )
            axes[2].set_title("Absolute Difference")
            axes[2].set_xlabel("Position")
            axes[2].set_ylabel("Sigma")
            plt.colorbar(im3, ax=axes[2], label="Absolute Difference")

            plt.tight_layout()
            st.pyplot(fig)

            # Add cross-section plots
            st.subheader("Cross-Section Views")

            # Sigma cross-section at a specific position
            pos_idx = st.slider(
                "Position Index for Cross-Section",
                min_value=0,
                max_value=len(x) - 1,
                value=len(x) // 2,
            )
            selected_pos = x[pos_idx].item()

            fig_pos, ax_pos = plt.subplots(figsize=(12, 5))
            ax_pos.plot(
                sigmas.numpy(),
                sigma_conv_reshaped[:, pos_idx].numpy(),
                "b-",
                label="Sigma Convolution",
                linewidth=2,
            )
            ax_pos.plot(
                sigmas.numpy(),
                full_conv_slice[:, pos_idx].cpu().numpy(),
                "r--",
                label="Full Convolution Slice",
                linewidth=2,
            )
            ax_pos.grid(True)
            ax_pos.set_title(f"Sigma Cross-Section at Position = {selected_pos:.2f}")
            ax_pos.set_xlabel("Sigma")
            ax_pos.set_ylabel("Amplitude")
            ax_pos.legend()
            st.pyplot(fig_pos)

            # Position cross-section at a specific sigma
            sigma_idx = st.slider(
                "Sigma Index for Cross-Section",
                min_value=0,
                max_value=len(sigmas) - 1,
                value=len(sigmas) // 2,
            )
            selected_sigma = sigmas[sigma_idx].item()

            fig_sigma, ax_sigma = plt.subplots(figsize=(12, 5))
            ax_sigma.plot(
                x.numpy(),
                sigma_conv_reshaped[sigma_idx, :].numpy(),
                "b-",
                label="Sigma Convolution",
                linewidth=2,
            )
            ax_sigma.plot(
                x.numpy(),
                full_conv_slice[sigma_idx, :].cpu().numpy(),
                "r--",
                label="Full Convolution Slice",
                linewidth=2,
            )
            ax_sigma.grid(True)
            ax_sigma.set_title(
                f"Position Cross-Section at Sigma = {selected_sigma:.2f}"
            )
            ax_sigma.set_xlabel("Position")
            ax_sigma.set_ylabel("Amplitude")
            ax_sigma.legend()
            st.pyplot(fig_sigma)

            # Plot the difference for this cross-section
            fig_diff, ax_diff = plt.subplots(figsize=(12, 5))
            ax_diff.plot(
                x.numpy(),
                torch.abs(
                    sigma_conv_reshaped[sigma_idx, :]
                    - full_conv_slice[sigma_idx, :].cpu()
                ).numpy(),
                "g-",
                linewidth=2,
            )
            ax_diff.grid(True)
            ax_diff.set_title(f"Absolute Difference at Sigma = {selected_sigma:.2f}")
            ax_diff.set_xlabel("Position")
            ax_diff.set_ylabel("Absolute Difference")
            st.pyplot(fig_diff)


def visualize_full_conv_slice(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigmas: torch.Tensor,
    phases: torch.Tensor,
    conv_heatmap: torch.Tensor,
    title="Full Convolution Slice",
):
    """
    Visualizes a 2D slice from a 4D full convolution tensor with option to switch between
    different views (sigma vs position, frequency vs position, or phase vs position).

    Parameters:
        x (torch.Tensor): Domain tensor representing positions
        frequencies (torch.Tensor): Frequency values used in the convolution
        sigmas (torch.Tensor): Sigma values used in the convolution
        phases (torch.Tensor): Phase values used in the convolution
        conv_heatmap (torch.Tensor): 4D tensor of convolution results [freq, sigma, phase, position]
        title (str): Title for the visualization

    Returns:
        torch.Tensor: The selected 2D slice
    """
    # Ensure tensors are on CPU and convert to NumPy
    x_np = x.cpu().numpy()
    freq_np = frequencies.cpu().numpy()
    sigma_np = sigmas.cpu().numpy()
    phase_np = phases.cpu().numpy()

    # Add view selection
    view_options = ["Sigma vs Position", "Frequency vs Position", "Phase vs Position"]
    selected_view = st.radio(
        "Select View Type", options=view_options, index=1, key=f"{title}_view_type"
    )

    st.subheader("Parameter Selection")

    # Initialize indices
    freq_idx = len(freq_np) // 2
    sigma_idx = len(sigma_np) // 2
    phase_idx = 0

    # Create columns for sliders
    if selected_view == "Sigma vs Position":
        # Need to select frequency and phase
        col1, col2 = st.columns(2)
        with col1:
            freq_idx = st.slider(
                "Frequency Index",
                min_value=0,
                max_value=len(freq_np) - 1,
                value=freq_idx,
                key=f"{title}_freq_idx",
            )
            selected_freq = freq_np[freq_idx]
            st.write(f"Selected Frequency: {selected_freq:.3f}")
        with col2:
            phase_idx = st.slider(
                "Phase Index",
                min_value=0,
                max_value=len(phase_np) - 1,
                value=phase_idx,
                key=f"{title}_phase_idx",
            )
            selected_phase = phase_np[phase_idx]
            st.write(f"Selected Phase: {selected_phase:.3f} radians")

        # Extract the 2D slice for the selected frequency and phase
        conv_slice = conv_heatmap[freq_idx, :, phase_idx, :].cpu()
        subtitle = f"Freq: {selected_freq:.2f}, Phase: {selected_phase:.2f}"
        x_label = "Position"
        y_label = "Sigma"
        y_extent = (sigma_np[0], sigma_np[-1])

    elif selected_view == "Frequency vs Position":
        # Need to select sigma and phase
        col1, col2 = st.columns(2)
        with col1:
            sigma_idx = st.slider(
                "Sigma Index",
                min_value=0,
                max_value=len(sigma_np) - 1,
                value=sigma_idx,
                key=f"{title}_sigma_idx",
            )
            selected_sigma = sigma_np[sigma_idx]
            st.write(f"Selected Sigma: {selected_sigma:.3f}")
        with col2:
            phase_idx = st.slider(
                "Phase Index",
                min_value=0,
                max_value=len(phase_np) - 1,
                value=phase_idx,
                key=f"{title}_phase_idx",
            )
            selected_phase = phase_np[phase_idx]
            st.write(f"Selected Phase: {selected_phase:.3f} radians")

        # Extract the 2D slice for the selected sigma and phase
        conv_slice = conv_heatmap[:, sigma_idx, phase_idx, :].cpu()
        subtitle = f"Sigma: {selected_sigma:.2f}, Phase: {selected_phase:.2f}"
        x_label = "Position"
        y_label = "Frequency"
        y_extent = (freq_np[0], freq_np[-1])

    else:  # "Phase vs Position"
        # Need to select frequency and sigma
        col1, col2 = st.columns(2)
        with col1:
            freq_idx = st.slider(
                "Frequency Index",
                min_value=0,
                max_value=len(freq_np) - 1,
                value=freq_idx,
                key=f"{title}_freq_idx",
            )
            selected_freq = freq_np[freq_idx]
            st.write(f"Selected Frequency: {selected_freq:.3f}")
        with col2:
            sigma_idx = st.slider(
                "Sigma Index",
                min_value=0,
                max_value=len(sigma_np) - 1,
                value=sigma_idx,
                key=f"{title}_sigma_idx",
            )
            selected_sigma = sigma_np[sigma_idx]
            st.write(f"Selected Sigma: {selected_sigma:.3f}")

        # Extract the 2D slice for the selected frequency and sigma
        conv_slice = conv_heatmap[freq_idx, sigma_idx, :, :].cpu()
        subtitle = f"Freq: {selected_freq:.2f}, Sigma: {selected_sigma:.2f}"
        x_label = "Position"
        y_label = "Phase"
        y_extent = (phase_np[0], phase_np[-1])

    # Display the visualization
    st.subheader(f"{title} - {subtitle}")

    # Create a Matplotlib heatmap
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        conv_slice.numpy(),
        aspect="auto",
        origin="lower",
        extent=(x_np[0], x_np[-1], y_extent[0], y_extent[1]),
        # cmap="viridis",
        cmap="turbo",
    )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Amplitude")
    plt.tight_layout()
    st.pyplot(fig)

    # Return the selected slice in case it's needed elsewhere
    return conv_slice


def visualize_1d_conv_signal(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigmas: torch.Tensor,
    phases: torch.Tensor,
    conv_heatmap: torch.Tensor,
    title="1D Convolution Signal",
    linetype="b-",
    linewidth=2,
):
    """
    Visualizes a 1D signal from a 4D full convolution tensor by selecting specific
    frequency, sigma, and phase values.

    Parameters:
        x (torch.Tensor): Domain tensor representing positions
        frequencies (torch.Tensor): Frequency values used in the convolution
        sigmas (torch.Tensor): Sigma values used in the convolution
        phases (torch.Tensor): Phase values used in the convolution
        conv_heatmap (torch.Tensor): 4D tensor of convolution results [freq, sigma, phase, position]
        title (str): Title for the visualization

    Returns:
        torch.Tensor: The selected 1D signal
    """
    # Ensure tensors are on CPU and convert to NumPy
    x_np = x.cpu().numpy()
    freq_np = frequencies.cpu().numpy()
    sigma_np = sigmas.cpu().numpy()
    phase_np = phases.cpu().numpy()

    st.subheader("Parameter Selection")

    # Initialize default indices (middle values)
    default_freq_idx = len(freq_np) // 2
    default_sigma_idx = len(sigma_np) // 2
    default_phase_idx = 0

    # Create columns for sliders
    col1, col2, col3 = st.columns(3)

    with col1:
        freq_idx = st.slider(
            "Frequency Index",
            min_value=0,
            max_value=len(freq_np) - 1,
            value=default_freq_idx,
            key=f"{title}_freq_idx",
        )
        selected_freq = freq_np[freq_idx]
        st.write(f"Selected Frequency: {selected_freq:.3f}")

    with col2:
        sigma_idx = st.slider(
            "Sigma Index",
            min_value=0,
            max_value=len(sigma_np) - 1,
            value=default_sigma_idx,
            key=f"{title}_sigma_idx",
        )
        selected_sigma = sigma_np[sigma_idx]
        st.write(f"Selected Sigma: {selected_sigma:.3f}")

    with col3:
        phase_idx = st.slider(
            "Phase Index",
            min_value=0,
            max_value=len(phase_np) - 1,
            value=default_phase_idx,
            key=f"{title}_phase_idx",
        )
        selected_phase = phase_np[phase_idx]
        st.write(f"Selected Phase: {selected_phase:.3f} radians")

    # Extract the 1D signal for the selected parameters
    conv_signal = conv_heatmap[freq_idx, sigma_idx, phase_idx, :].cpu()

    # Create subtitle with selected parameters
    subtitle = f"Freq: {selected_freq:.3f}, Sigma: {selected_sigma:.3f}, Phase: {selected_phase:.3f}"

    # Display the visualization
    st.subheader(f"{title} - {subtitle}")

    # Create a Matplotlib line plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_np, conv_signal.numpy(), linetype, linewidth)
    ax.set_xlabel("Position")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"{title}\n{subtitle}")
    ax.grid(True, alpha=0.3)

    # Add zero line for reference
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5, linewidth=1)

    plt.tight_layout()
    st.pyplot(fig)

    # Display some statistics about the signal
    st.subheader("Signal Statistics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Max Amplitude", f"{conv_signal.max().item():.4f}")
    with col2:
        st.metric("Min Amplitude", f"{conv_signal.min().item():.4f}")
    with col3:
        st.metric("Mean", f"{conv_signal.mean().item():.4f}")
    with col4:
        st.metric("Std Dev", f"{conv_signal.std().item():.4f}")

    # Return the selected signal in case it's needed elsewhere
    return conv_signal
