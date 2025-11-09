import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def plot_summary_figure(
    I_input,
    I_guided,
    I_guided_prop,
    I_propagated,
    P_input_core,
    P_guided_core,
    P_input,
    P_guided,
    eta,
    df_coeff_input,
    df_coeff_output,
    prop_axis_ext,
    axis_ext,
    radius,
    CMAP,
    DIST_FROM_FIBER,
    dist_from_waist,
    w0_x,
    w0_y,
    x0,
    y0,
    roll_angle,
    pitch_angle,
    yaw_angle,
    polarization_angle,
    fiber_length,
    fiber_v,
    normalize_palette=True,
):
    """
    Generates a 2x3 summary figure with intensity plots, power summary,
    and coefficient tables.
    """

    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)

    # --- Setup Axes ---
    ax_in = axes[0, 0]
    ax_guided = axes[0, 1]
    ax_summary_in = axes[0, 2]

    ax_out = axes[1, 0]
    ax_propagated = axes[1, 1]
    ax_summary_out = axes[1, 2]

    # --- Share axes for in, guided and out intensity plots ---
    ax_guided.sharex(ax_in)
    ax_out.sharex(ax_in)

    ax_guided.sharey(ax_in)
    ax_out.sharey(ax_in)

    # --- Base Image Arguments ---
    im_args = {
        "extent": [-axis_ext, axis_ext, -axis_ext, axis_ext],
        "origin": "lower",
        "cmap": CMAP,
        "aspect": "equal",
    }

    if normalize_palette:
        # Vmax is calculated ONLY on the first three plots
        vmax = np.max(
            [
                np.max(I_input),
                np.max(I_guided),
                np.max(I_guided_prop),
                # I_propagated is excluded
            ]
        )
        if vmax == 0:
            vmax = 1.0

        im_args["vmin"] = 0
        im_args["vmax"] = vmax

    # --- Row 1 ---

    # Input Intensity
    im_in = ax_in.imshow(I_input, **im_args)
    ax_in.set_title("Input Intensity (z=0)")
    ax_in.set_ylabel("y (radius units)")

    # Guided Intensity (Input)
    im_guided = ax_guided.imshow(I_guided, **im_args)
    ax_guided.set_title("Guided Intensity (z=0)")

    # --- START: Summary (Top Right) ---
    ax_summary_in.axis("off")

    # --- Column 1: Simulation Parameters ---
    ax_summary_in.text(
        0.05,
        0.96,
        "Simulation Param.",
        transform=ax_summary_in.transAxes,
        fontsize=12,
        va="top",
        weight="bold",
    )
    params_text = (
        f"V-Number:    {fiber_v:.2f}\n"
        f"Fiber Len:   {fiber_length:.1e}\n"
        f"Dist. waist:  {dist_from_waist:.2f}\n"
        f"Waist:   ({w0_x:.2f}, {w0_y:.2f})\n"
        f"Incid.:  ({x0:.2f}, {y0:.2f})\n"
        f"Roll Angle:    {roll_angle * 180 / np.pi:.2f}°\n"
        f"Pitch Angle: {pitch_angle * 180 / np.pi:.2f}°\n"
        f"Yaw Angle:   {yaw_angle * 180 / np.pi:.2f}°\n"
        f"Pol. Angle:   {polarization_angle * 180 / np.pi:.2f}"
    )
    ax_summary_in.text(
        0.05,
        0.9,  # Position data below title
        params_text,
        transform=ax_summary_in.transAxes,
        fontsize=9,
        va="top",
        fontfamily="monospace",
    )

    # --- Column 2: Power Summary ---
    summary_text = (
        f"P_input_core  = {P_input_core:.3f}\n"
        f"P_guided_core = {P_guided_core:.3f}\n"
        f"P_input_total = {P_input:.2f}\n"
        f"P_guided_total= {P_guided:.2f}\n"
        f"Coupling (eta)= {eta:.3f}"
    )
    ax_summary_in.text(
        0.55,
        0.96,
        "Power Summary",
        transform=ax_summary_in.transAxes,
        fontsize=12,
        va="top",
        weight="bold",
    )
    ax_summary_in.text(
        0.55,
        0.9,
        summary_text,
        transform=ax_summary_in.transAxes,
        fontsize=10,
        va="top",
        fontfamily="monospace",
    )

    # --- Input coefficients table ---
    ax_summary_in.text(
        0.05,
        0.50, # Positioned below the text blocks
        r"Input Coefficients ($|A|^2$ %)",
        transform=ax_summary_in.transAxes,
        fontsize=12,
        va="top",
        weight="bold",
    )
    # --- END: Summary ---

    try:
        # Calculate squared modulus in %
        df_plot_in = np.abs(df_coeff_input.iloc[:, 1:]) ** 2 * 100

        table_data_in = df_plot_in.reset_index().values
        formatted_data_in = []
        for row in table_data_in:
            formatted_row = [f"{row[0]:.0f}", f"{row[1]:.0f}"] + [
                f"{x:.1f}" for x in row[2:]
            ]
            formatted_data_in.append(formatted_row)

        column_labels_in = ["l", "m"] + df_plot_in.columns.tolist()

        table_in = ax_summary_in.table(
            cellText=formatted_data_in,
            colLabels=column_labels_in,
            loc="center",
            cellLoc="center",
            bbox=[0.05, -0.05, 0.90, 0.45], # Stays at the bottom
        )
        table_in.auto_set_font_size(False)
        table_in.set_fontsize(9)
        table_in.scale(1.0, 1.2)

    except Exception as e:
        ax_summary_in.text(
            0.0,
            0.3,
            f"Error creating input table: {e}",
            transform=ax_summary_in.transAxes,
            color="red",
        )

    # --- Row 2 ---

    # Output Intensity
    im_out = ax_out.imshow(I_guided_prop, **im_args)
    ax_out.set_title("Output Intensity (z=L)")
    ax_out.set_xlabel("x (radius units)")
    ax_out.set_ylabel("y (radius units)")

    # This plot ALWAYS auto-scales, ignoring im_args['vmin']/im_args['vmax']
    im_propagated = ax_propagated.imshow(
        I_propagated,
        origin="lower",
        cmap=CMAP,
        aspect="equal",
        extent=[-prop_axis_ext, prop_axis_ext, -prop_axis_ext, prop_axis_ext],
    )
    
    
    ax_propagated.set_title(f"Propagated Intensity (z=L+{DIST_FROM_FIBER})")
    ax_propagated.set_xlabel("x (radius units)")

    # --- Output Coefficients ---
    ax_summary_out.axis("off")
    ax_summary_out.text(
        0.05,
        0.85, # Position for title
        r"Output Coefficients  ($|A|^2$ %)",
        transform=ax_summary_out.transAxes,
        fontsize=12,
        va="top",
        weight="bold",
    )

    try:
        # Calculate squared modulus in %
        df_plot_out = np.abs(df_coeff_output.iloc[:, 1:]) ** 2 * 100

        table_data_out = df_plot_out.reset_index().values
        formatted_data_out = []
        for row in table_data_out:
            formatted_row = [f"{row[0]:.0f}", f"{row[1]:.0f}"] + [
                f"{x:.1f}" for x in row[2:]
            ]
            formatted_data_out.append(formatted_row)

        column_labels_out = ["l", "m"] + df_plot_out.columns.tolist()

        table_out = ax_summary_out.table(
            cellText=formatted_data_out,
            colLabels=column_labels_out,
            loc="center",
            cellLoc="center",
            bbox=[0.05, 0.1, 0.90, 0.65], 
        )
        table_out.auto_set_font_size(False)
        table_out.set_fontsize(9)
        table_out.scale(1.0, 1.2)

    except Exception as e:
        ax_summary_out.text(
            0.0,
            0.5,
            f"Error creating output table: {e}",
            transform=ax_summary_out.transAxes,
            color="red",
        )

    # --- Add core circle overlay ---
    for ax in [ax_in, ax_guided, ax_out]:
        core_circle = Circle(
            (0, 0),
            radius,
            facecolor="none",
            edgecolor="white",
            linewidth=1.0,
            linestyle="--",
            zorder=5,
        )
        ax.add_patch(core_circle)
        
    core_circle_prop = Circle(
            (0, 0),
            radius,
            facecolor="none",
            edgecolor="white",
            linewidth=1.0,
            linestyle="--",
            zorder=5,
        )
    ax_propagated.add_patch(core_circle_prop)


    # --- Add Colorbars ---
    
    fig.colorbar(im_in, ax=ax_in, shrink=0.8, aspect=20, pad=0.02)
    fig.colorbar(im_guided, ax=ax_guided, shrink=0.8, aspect=20, pad=0.02)
    fig.colorbar(im_out, ax=ax_out, shrink=0.8, aspect=20, pad=0.02)
    fig.colorbar(im_propagated, ax=ax_propagated, shrink=0.8, aspect=20, pad=0.02)

    return fig, axes