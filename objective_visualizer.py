import torch
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import plotly.graph_objects as go


def visualize_gabor_objective_slice(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigmas: torch.Tensor,
    objective_heatmap: torch.Tensor,
    title="Gabor Objective Function",
):
    """
    Visualizes a 2D slice from a 3D Gabor objective tensor.
    Compatible with the optimized compute_gabor_objective function output.

    Parameters:
        x (torch.Tensor): Domain tensor representing positions
        frequencies (torch.Tensor): Frequency values used in the objective
        sigmas (torch.Tensor): Sigma values used in the objective
        objective_heatmap (torch.Tensor): 3D tensor of objective results [freq, sigma, position]
        title (str): Title for the visualization

    Returns:
        torch.Tensor: The selected 2D slice
    """
    # Ensure tensors are on CPU and convert to NumPy
    x_np = x.cpu().numpy()
    freq_np = frequencies.cpu().numpy()
    sigma_np = sigmas.cpu().numpy()

    st.subheader("Parameter Selection")

    # View selection - toggle between different 2D projections
    view_options = ["Sigma vs Position", "Frequency vs Position"]
    selected_view = st.radio(
        "📊 Select View Type",
        options=view_options,
        index=1,
        key=f"{title}_sigma_view",
        help="Choose which 2D slice to visualize from the 3D objective function",
    )

    if selected_view == "Sigma vs Position":
        # Need to select frequency only
        freq_idx = st.slider(
            "Frequency Index",
            min_value=0,
            max_value=len(freq_np) - 1,
            value=int(len(freq_np) / 2),
            key=f"{title}_freq_idx",
        )
        selected_freq = freq_np[freq_idx]
        st.write(f"Selected Frequency: {selected_freq:.3f}")

        # Extract the 2D slice for the selected frequency
        conv_slice = objective_heatmap[freq_idx, :, :].cpu()
        x_label = "Position"
        y_label = "Sigma"
        y_extent = (sigma_np[0], sigma_np[-1])

    else:  # "Frequency vs Position"
        # Need to select sigma only
        sigma_idx = st.slider(
            "Sigma Index",
            min_value=0,
            max_value=len(sigma_np) - 1,
            value=int(len(sigma_np) / 2),
            key=f"{title}_sigma_idx",
        )
        selected_sigma = sigma_np[sigma_idx]
        st.write(f"Selected Sigma: {selected_sigma:.3f}")

        # Extract the 2D slice for the selected sigma
        conv_slice = objective_heatmap[:, sigma_idx, :].cpu()
        subtitle = f"Sigma: {selected_sigma:.3f}"
        x_label = "Position"
        y_label = "Frequency"
        y_extent = (freq_np[0], freq_np[-1])

    # Create a Matplotlib heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(
        conv_slice.numpy(),
        aspect="auto",
        origin="lower",
        extent=(x_np[0], x_np[-1], y_extent[0], y_extent[1]),
        cmap="turbo",
        interpolation="bilinear",
    )
    ax.set_xlabel(f"{x_label}", fontsize=12)
    ax.set_ylabel(f"{y_label}", fontsize=12)
    ax.set_title(f"{selected_view} View", fontsize=14, fontweight="bold")

    # Enhanced colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Objective Value", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    # Add grid for better readability
    ax.grid(True, alpha=0.2, linestyle="--")

    plt.tight_layout()
    st.pyplot(fig)

    # Display slice statistics
    slice_max = torch.max(conv_slice).item()
    slice_min = torch.min(conv_slice).item()
    slice_mean = torch.mean(conv_slice).item()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Maximum", f"{slice_max:.4f}")
    with col2:
        st.metric("Minimum", f"{slice_min:.4f}")
    with col3:
        st.metric("Mean", f"{slice_mean:.4f}")

    # Return the selected slice in case it's needed elsewhere
    return conv_slice


def visualize_gabor_objective_3d_interactive(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigmas: torch.Tensor,
    objective_heatmap: torch.Tensor,
    title="3D Gabor Objective Function",
):
    """
    Creates an interactive 3D visualization of the Gabor objective function.
    Shows both frequency-sigma parameter space and position-dependent responses.

    Parameters:
        x (torch.Tensor): Domain tensor representing positions
        frequencies (torch.Tensor): Frequency values
        sigmas (torch.Tensor): Sigma values
        objective_heatmap (torch.Tensor): 3D tensor [freq, sigma, position]
        title (str): Title for the visualization
    """
    x_np = x.cpu().numpy()
    freq_np = frequencies.cpu().numpy()
    sigma_np = sigmas.cpu().numpy()
    objective_np = objective_heatmap.cpu().numpy()

    st.subheader(f"{title} - Parameter Space Analysis")

    # Position selection for parameter space view
    pos_idx = st.slider(
        "Position Index for Parameter Space View",
        min_value=0,
        max_value=len(x_np) - 1,
        value=len(x_np) // 2,
        key=f"{title}_pos_idx",
    )
    selected_pos = x_np[pos_idx]
    st.write(f"Selected Position: {selected_pos:.3f}")

    # Create parameter space heatmap
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # Left plot: Parameter space (freq vs sigma) at selected position
    param_slice = objective_np[:, :, pos_idx]
    im1 = ax1.imshow(
        param_slice,
        aspect="auto",
        origin="lower",
        extent=(sigma_np[0], sigma_np[-1], freq_np[0], freq_np[-1]),
        cmap="turbo",
    )
    ax1.set_xlabel("Sigma")
    ax1.set_ylabel("Frequency")
    ax1.set_title(f"Parameter Space at Position {selected_pos:.2f}")
    plt.colorbar(im1, ax=ax1, label="Objective Value")

    # Right plot: Position response for best parameters
    best_idx = np.unravel_index(np.argmax(param_slice), param_slice.shape)
    best_freq_idx, best_sigma_idx = best_idx
    best_freq = freq_np[best_freq_idx]
    best_sigma = sigma_np[best_sigma_idx]

    position_response = objective_np[best_freq_idx, best_sigma_idx, :]
    ax2.plot(x_np, position_response, "b-", linewidth=2)
    ax2.axvline(
        selected_pos, color="red", linestyle="--", alpha=0.7, label=f"Selected Position"
    )
    ax2.set_xlabel("Position")
    ax2.set_ylabel("Objective Value")
    ax2.set_title(
        f"Response for Best Parameters\n(f={best_freq:.3f}, σ={best_sigma:.3f})"
    )
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    st.pyplot(fig)

    # Display optimal parameters
    st.write(f"**Optimal Parameters at Position {selected_pos:.3f}:**")
    st.write(f"- Frequency: {best_freq:.4f}")
    st.write(f"- Sigma: {best_sigma:.4f}")
    st.write(f"- Objective Value: {param_slice[best_freq_idx, best_sigma_idx]:.4f}")


def visualize_gabor_objective_summary(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigmas: torch.Tensor,
    objective_heatmap: torch.Tensor,
    title="Gabor Objective Summary",
):
    """
    Provides summary statistics and global optimization results.

    Parameters:
        x (torch.Tensor): Domain tensor representing positions
        frequencies (torch.Tensor): Frequency values
        sigmas (torch.Tensor): Sigma values
        objective_heatmap (torch.Tensor): 3D tensor [freq, sigma, position]
        title (str): Title for the visualization
    """
    x_np = x.cpu().numpy()
    freq_np = frequencies.cpu().numpy()
    sigma_np = sigmas.cpu().numpy()
    objective_np = objective_heatmap.cpu().numpy()

    st.subheader(f"{title}")

    # Global maximum
    global_max_idx = np.unravel_index(np.argmax(objective_np), objective_np.shape)
    global_max_freq_idx, global_max_sigma_idx, global_max_pos_idx = global_max_idx

    global_max_freq = freq_np[global_max_freq_idx]
    global_max_sigma = sigma_np[global_max_sigma_idx]
    global_max_pos = x_np[global_max_pos_idx]
    global_max_value = objective_np[global_max_idx]

    # Statistics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Global Maximum", f"{global_max_value:.4f}")
        st.metric("Mean Objective", f"{np.mean(objective_np):.4f}")

    with col2:
        st.metric("Optimal Frequency", f"{global_max_freq:.4f}")
        st.metric("Optimal Sigma", f"{global_max_sigma:.4f}")

    with col3:
        st.metric("Optimal Position", f"{global_max_pos:.4f}")
        st.metric("Std Deviation", f"{np.std(objective_np):.4f}")

    # Position-wise optimization
    st.subheader("Position-wise Optimal Parameters")

    # Find best parameters for each position
    best_params_per_pos = []
    for pos_idx in range(len(x_np)):
        pos_slice = objective_np[:, :, pos_idx]
        best_idx = np.unravel_index(np.argmax(pos_slice), pos_slice.shape)
        best_freq_idx, best_sigma_idx = best_idx
        best_params_per_pos.append(
            {
                "position": x_np[pos_idx],
                "frequency": freq_np[best_freq_idx],
                "sigma": sigma_np[best_sigma_idx],
                "objective": pos_slice[best_idx],
            }
        )

    # Plot optimal parameters vs position
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    positions = [p["position"] for p in best_params_per_pos]
    optimal_frequencies = [p["frequency"] for p in best_params_per_pos]
    optimal_sigmas = [p["sigma"] for p in best_params_per_pos]
    objectives = [p["objective"] for p in best_params_per_pos]

    ax1.plot(positions, optimal_frequencies, "b-", linewidth=2)
    ax1.set_ylabel("Optimal Frequency")
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Position-wise Optimal Parameters")

    ax2.plot(positions, optimal_sigmas, "g-", linewidth=2)
    ax2.set_ylabel("Optimal Sigma")
    ax2.grid(True, alpha=0.3)

    ax3.plot(positions, objectives, "r-", linewidth=2)
    ax3.set_xlabel("Position")
    ax3.set_ylabel("Objective Value")
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(fig)


def visualize_gabor_objective_volume(
    x: torch.Tensor,
    frequencies: torch.Tensor,
    sigmas: torch.Tensor,
    objective_heatmap: torch.Tensor,
    title="3D Gabor Objective Volume",
):
    """
    Creates a 3D volume visualization of the Gabor objective function.

    Parameters:
        x (torch.Tensor): Domain tensor representing positions
        frequencies (torch.Tensor): Frequency values
        sigmas (torch.Tensor): Sigma values
        objective_heatmap (torch.Tensor): 3D tensor [freq, sigma, position]
        title (str): Title for the visualization
    """
    # Convert to NumPy
    objective_np = torch.abs(objective_heatmap).cpu().numpy()
    x_np = x.cpu().numpy()
    freq_np = frequencies.cpu().numpy()
    sigma_np = sigmas.cpu().numpy()

    # Generate meshgrid for coordinates
    F, S, X = np.meshgrid(freq_np, sigma_np, x_np, indexing="ij")

    # Create volume plot
    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=F.flatten(),
            z=S.flatten(),
            value=objective_np.flatten(),
            isomin=float(objective_np.max() * 0.1),
            isomax=objective_np.max(),
            opacity=0.1,
            surface_count=15,
            colorscale="turbo",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )

    fig.update_layout(
        title=title + " — 3D Volume View",
        scene=dict(
            xaxis_title="Position",
            yaxis_title="Frequency",
            zaxis_title="Sigma",
        ),
        width=900,
        height=700,
    )

    st.plotly_chart(fig)
