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
    axis_ext,
    radius,
    CMAP,
    DIST_FROM_FIBER,
    normalize_palette=True,
):
    """
    Generates a 2x3 summary figure with intensity plots, power summary,
    and coefficient tables.
    """

    fig, axes = plt.subplots(2, 3, figsize=(21, 10), constrained_layout=True)

    # --- Setup Axes ---
    ax_in = axes[0, 0]
    ax_guided = axes[0, 1]
    ax_summary_in = axes[0, 2]

    ax_out = axes[1, 0]
    ax_propagated = axes[1, 1]
    ax_summary_out = axes[1, 2]

    # --- Share axes for intensity plots ---
    ax_guided.sharex(ax_in)
    ax_out.sharex(ax_in)
    ax_propagated.sharex(ax_in)

    ax_guided.sharey(ax_in)
    ax_out.sharey(ax_in)
    ax_propagated.sharey(ax_in)

    # --- Base Image Arguments ---
    im_args = {
        "extent": [-axis_ext, axis_ext, -axis_ext, axis_ext],
        "origin": "lower",
        "cmap": CMAP,
        "aspect": "equal",
    }

    # --- Conditional Normalization ---
    if normalize_palette:
        vmax = np.max(
            [
                np.max(I_input),
                np.max(I_guided),
                np.max(I_guided_prop),
                np.max(I_propagated),
            ]
        )
        if vmax == 0:
            vmax = 1.0

        im_args["vmin"] = 0
        im_args["vmax"] = vmax

    # --- Row 1 ---

    # Input Intensity
    ax_in.imshow(I_input, **im_args)
    ax_in.set_title("Input Intensity (z=0)")
    ax_in.set_ylabel("y (radius units)")

    # Guided Intensity (Input)
    ax_guided.imshow(I_guided, **im_args)
    ax_guided.set_title("Guided Intensity (z=0)")

    # Power Summary & Input Coefficients
    ax_summary_in.axis("off")

    # Power summary text
    summary_text = (
        f"P_input_core  = {P_input_core:.3f}\n"
        f"P_guided_core = {P_guided_core:.3f}\n"
        f"P_input_total = {P_input:.2f}\n"
        f"P_guided_total= {P_guided:.2f}\n"
        f"Coupling (eta)= {eta:.3f}"
    )
    ax_summary_in.text(
        0.05,
        0.96,
        "Power Summary",
        transform=ax_summary_in.transAxes,
        fontsize=12,
        va="top",
        weight="bold",
    )
    ax_summary_in.text(
        0.05,
        0.9,
        summary_text,
        transform=ax_summary_in.transAxes,
        fontsize=10,
        va="top",
        fontfamily="monospace",
    )

    # Input coefficients table
    ax_summary_in.text(
        0.05,
        0.55,
        r"Input Coefficients ($|A|^2$ %)",
        transform=ax_summary_in.transAxes,
        fontsize=12,
        va="top",
        weight="bold",
    )

    try:
        # Calculate squared modulus in %
        df_plot_in = np.abs(df_coeff_input.iloc[:, 1:]) ** 2 * 100

        table_data_in = df_plot_in.reset_index().values
        # Format data to 1 decimal place
        formatted_data_in = []
        for row in table_data_in:
            # Format l/m as int (0 decimal places), coefficients as 1 decimal place
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
            bbox=[0.05, 0.0, 0.90, 0.45],
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

    # Propagated Intensity
    ax_propagated.imshow(I_propagated, **im_args)
    ax_propagated.set_title(f"Propagated Intensity (z=L+{DIST_FROM_FIBER})")
    ax_propagated.set_xlabel("x (radius units)")

    # Output Coefficients
    ax_summary_out.axis("off")
    ax_summary_out.text(
        0.05,
        0.85,
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
        # Format data to 1 decimal place
        formatted_data_out = []
        for row in table_data_out:
            # Format l/m as int (0 decimal places), coefficients as 1 decimal place
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
            bbox=[0.05, 0.3, 0.90, 0.45],
        )  # <-- Reduced height from 0.85 to 0.45
        table_out.auto_set_font_size(False)
        table_out.set_fontsize(9)
        table_out.scale(1.0, 1.2)
        # Position table
        # table_out.set_bbox([0.0, 0.0, 1.0, 0.85]) # <-- REMOVED: This line caused the error

    except Exception as e:
        ax_summary_out.text(
            0.0,
            0.5,
            f"Error creating output table: {e}",
            transform=ax_summary_out.transAxes,
            color="red",
        )

    # --- Add core circle overlay ---
    for ax in [ax_in, ax_guided, ax_out, ax_propagated]:
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

    # --- Add shared colorbar ---
    # Place colorbar to the right of the intensity plots

    if normalize_palette:
        cbar = fig.colorbar(
            im_out, ax=[ax_guided, ax_propagated], shrink=0.8, aspect=30, pad=0.02
        )
        cbar.set_label(None)  # Remove label as requested
    else:
        cbar_in = fig.colorbar(
            ax_in.images[0], ax=ax_in, shrink=0.8, aspect=30, pad=0.02
        )
        cbar_in.set_label(None)  # Remove label as requested

        cbar_guided = fig.colorbar(
            ax_guided.images[0], ax=ax_guided, shrink=0.8, aspect=30, pad=0.02
        )
        cbar_guided.set_label(None)  # Remove label as requested

        cbar_out = fig.colorbar(
            ax_out.images[0], ax=ax_out, shrink=0.8, aspect=30, pad=0.02
        )
        cbar_out.set_label(None)  # Remove label as requested

        cbar_propagated = fig.colorbar(
            ax_propagated.images[0], ax=ax_propagated, shrink=0.8, aspect=30, pad=0.02
        )
        cbar_propagated.set_label(None)  # Remove label as requested

    return fig, axes
