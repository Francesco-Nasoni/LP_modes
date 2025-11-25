import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.patches import Circle
from scipy.special import jv, kn, jn_zeros
from scipy.optimize import root_scalar
import pandas as pd

from source.LP_projection_functions import (
    get_guided_modes,
    get_LP_modes_projection_coefficients,
    get_complete_guided_field,
    get_tilted_beam_from_incidence,
)

from source.propagation import (
    fiber_propagation,
    free_propagation_asm_hankel
)

from source.graph import plot_summary_figure
import time

# --------------------------------------- PARAMETERS ----------------------------------------------
# -------------------------------------------------------------------------------------------------
# NOTE: all the length are measured in units of fiber radius

# --- Various Parameters ---
FIBER_V = 5.8
MODES_TO_TEST = [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)]
FIBER_N1 = 1
FIBER_LENGTH = 1.e4
DIST_FROM_FIBER = 800
RZ_FACTOR = 1.25

# --- Injected field parameters ---
LAMBDA = 0.0426                 # Wavelength of the injected beam
DIST_TO_WAIST = 0               # Distance from the beam waist to the fiber input plane
W0_X = 0.6                      # Beam waist size along the x-axis
W0_Y = 0.7                       # Beam waist size along the y-axis
X0 = -0.2                        # x-coordinate of the beam's incidence point on the fiber input plane
Y0 = -0.1                        # y-coordinate of the beam's incidence point on the fiber input plane
ROLL_ANGLE = -0 * np.pi / 180    # Roll angle of the beam (rotation about the z-axis, in radians)
PITCH_ANGLE = 0 * np.pi / 180   # Pitch angle of the beam (tilt in the x-z plane, in radians)
YAW_ANGLE = 0 * np.pi / 180     # Yaw angle of the beam (tilt in the y-z plane, in radians)
POLARIZATION_ANGLE = 0    # Polarization angle of the beam (angle of the electric field vector, in radians)

# --- Grid stuff ---
AXIS_SIZE = 1.3
GRID_SIZE = 1000
MIN_GRID_SIZE_PROPAGATED_FIELD = GRID_SIZE

# --- Visualization stuff ---
# Colormap name passed to matplotlib for the power density plots
# First parameter is the color map name ("gnuplot2" recommended),
# second parameter is the number of colors
CMAP = plt.get_cmap('gnuplot2', 20)

# If True, use a common color scale (same vmax) for input field and guided field plots
# to allow direct visual comparison. If False, each plot scales independently.
NORMALIZE_COLOR_PALETTE = False

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


# --- Fiber radius ---
radius = 1.0

# --- Some stuff on angles ---
NA = LAMBDA * FIBER_V / (2 * np.pi * radius)
total_tilt = np.arccos(np.cos(PITCH_ANGLE) * np.cos(YAW_ANGLE))

print("\n", "ANGLE STUFF", "\n" + "*" * 50)
print(f"Fiber NA = {NA:.2f}")
print(f"Fiber acceptance angle = {np.arcsin(NA) * (180/np.pi):.2f}°")
print(f"Total tilt setted = {total_tilt * (180/np.pi):.2f}°")
print("*" * 50 + "\n")

# --- Grid ---
axis_ext = AXIS_SIZE * radius
x = np.linspace(-axis_ext, axis_ext, GRID_SIZE)
y = np.linspace(-axis_ext, axis_ext, GRID_SIZE)
X, Y = np.meshgrid(x, y)

# --- Ploar coordinates ---
R = np.sqrt(X**2 + Y**2)
PHI = np.arctan2(Y, X)

# --- Differential Area Element ---
dA = (axis_ext * 2 / GRID_SIZE) ** 2

# --- DEFINE THE INPUT ELECTRIC FIELD AS A TILTED GAUSSIAN BEAM ---
E_input = get_tilted_beam_from_incidence(
    X,
    Y,
    z_plane=0,
    x_incidence=X0,
    y_incidence=Y0,
    dist_to_waist=DIST_TO_WAIST,
    euler_alpha=ROLL_ANGLE,
    euler_beta=PITCH_ANGLE,
    euler_gamma=YAW_ANGLE,
    dA=dA,
    w0_x=W0_X,
    w0_y=W0_Y,
    wavelength=LAMBDA,
    polarization_angle=POLARIZATION_ANGLE,
)

# --- COMPUTE THE GUIDED MODES AND THEIR PROJECTION COEFFICIENTS ON THE INPUT FIELD ---
guided_modes = []
coefficients = []
for l, m in MODES_TO_TEST:

    mode = get_guided_modes(l, m, FIBER_V, radius, R, PHI, dA)

    if mode is not None:
        guided_modes.append(mode)
        coefficients_res = get_LP_modes_projection_coefficients(E_input, mode, dA)
        coefficients.append(coefficients_res)


# create a pd dataframe, and make (l,m) the index for easier lookup
df_coeff = pd.DataFrame(coefficients)
df_coeff.set_index(["l", "m"], inplace=True)

# --- RECONSTRUCT THE GUIDED ELECTRIC FIELD USING LP MODES AND THEIR PROJECTION COEFFICIENTS ---
E_guided_x, E_guided_y, sum_squared_coeff = get_complete_guided_field(
    guided_modes, df_coeff, X, Y
)


# Get the power of the guided field and the coupling coefficient
E_input_x = E_input[0]
E_input_y = E_input[1]
I_guided = np.abs(E_guided_x) ** 2 + np.abs(E_guided_y) ** 2
I_input = np.abs(E_input_x) ** 2 + np.abs(E_input_y) ** 2

P_input_core = np.sum(I_input[R <= radius]) * dA
P_guided_core = np.sum(I_guided[R <= radius]) * dA
P_input = np.sum(I_input) * dA
P_guided = np.sum(I_guided) * dA

eta = P_guided / P_input if P_input != 0 else 0.0

coeff_power_transport = (np.abs(df_coeff.iloc[:, 1:]) ** 2) / P_guided


# --- TERMINAL OUTPUT ---
print("\n", "SQUARED MODULUS OF COEFFICIENTS", "\n" + "*" * 50)
print(
    (coeff_power_transport * 100).to_string(
        float_format=lambda x: f"{x:.1f}", justify="center", col_space=6
    )
)
print("*" * 50)
print("\n\n", "SUMMARY", "\n" + "*" * 50)
print(f"Sum of squared A coeff = {sum_squared_coeff:.2f}")
print(f"P_input by the core = {P_input_core:.3f}")
print(f"P_guided by the core = {P_guided_core:.3f}")
print(f"P_input = {P_input:.2f}")
print(f"P_guided = {P_guided:.2f}")
print(f"Coupling efficiency = {eta:.3f}")
print("*" * 50 + "\n")


# --- CALCULATE THE COEFFICIENTS AFTER THE PROPAGATION INSIDE THE FIBER ---
df_coeff_fib_prop = fiber_propagation(
    df_coeff,
    n1=FIBER_N1,
    a=radius,
    lam=LAMBDA,
    z_fiber=FIBER_LENGTH,
)

# --- RECONSTRUCT THE GUIDED ELECTRIC FIELD AFTER FIBER PROPAGATION ---
E_guided_x_prop, E_guided_y_prop, _ = get_complete_guided_field(
    guided_modes, df_coeff_fib_prop, X, Y
)

I_guided_prop = np.abs(E_guided_x_prop) ** 2 + np.abs(E_guided_y_prop) ** 2


# --- PROPAGATE THE FIELD USING ASM TO z=DIST_FROM_FIBER ---
start_time = time.time()

E_propagated_x, E_propagated_y, prop_axis_ext = free_propagation_asm_hankel(
    guided_modes,
    df_coeff_fib_prop,
    DIST_FROM_FIBER,
    NA,
    RZ_FACTOR,
    MIN_GRID_SIZE_PROPAGATED_FIELD,
    FIBER_V,
    axis_ext,
    min_point_per_period=10,
    radius=radius,
    lambda_0=LAMBDA
)

end_time = time.time()
print(f"Time taken for free_propagation: {end_time - start_time:.4f} seconds")

if E_propagated_x is not None:
    I_propagated = np.abs(E_propagated_x) ** 2 + np.abs(E_propagated_y) ** 2
    dA_prop = (2*prop_axis_ext/E_propagated_x.shape[0])**2

    print("\n", "PROPAGATED FIELD POWER", "\n" + "*" * 50)
    print(f"Power of the propagated field = {np.sum(I_propagated) * dA_prop:.3f}")
    print("*" * 50 + "\n")
else:
    I_propagated = np.zeros_like(np.real(E_guided_x_prop))
    prop_axis_ext = axis_ext

# --- VISUALIZATION ---
fig, _ = plot_summary_figure(
    I_input,
    I_guided,
    I_guided_prop,
    I_propagated,
    P_input_core,
    P_guided_core,
    P_input,
    P_guided,
    eta,
    df_coeff,
    df_coeff_fib_prop,
    prop_axis_ext,
    axis_ext,
    radius,
    CMAP,
    DIST_FROM_FIBER,
    dist_from_waist=DIST_TO_WAIST,
    w0_x=W0_X,
    w0_y=W0_Y,
    x0=X0,
    y0=Y0,
    roll_angle=ROLL_ANGLE,
    pitch_angle=PITCH_ANGLE,
    yaw_angle=YAW_ANGLE,
    polarization_angle=POLARIZATION_ANGLE,
    fiber_length=FIBER_LENGTH,
    fiber_v=FIBER_V,
    normalize_palette=NORMALIZE_COLOR_PALETTE,
)

# --- Manage output directory and file naming ---
output_dir = Path("output")
output_dir.mkdir(parents=True, exist_ok=True)
file_name = "summary_figure.png"
file_path = output_dir / file_name
counter = 1

while file_path.exists():
    file_name = f"summary_figure_{counter}.png"
    file_path = output_dir / file_name
    counter += 1

fig.savefig(file_path, dpi=300)

plt.show()
