import numpy as np
import pandas as pd

def fiber_propagation(df_coeff, n1, a, lam, z_fiber):
    """
    Computes the fiber propagation by applying phase factors to the coefficients.

    Parameters:
        df_coeff (pd.DataFrame): DataFrame containing mode coefficients and 'u' values.
        n1 (float): Refractive index of the core.
        a (float): Core radius of the fiber.
        lam (float): Wavelength of the light.
        z_fiber (float): Propagation distance along the fiber.

    Returns:
        pd.DataFrame: Updated DataFrame with modified coefficients.
    """
    df = df_coeff.copy()
    k0 = 2 * np.pi / lam

    df["beta_lm"] = (np.sqrt((n1 * k0) ** 2 - (df["u"] / a) ** 2)).astype(np.complex128)
    df["phase_fact"] = np.exp(-1j * df["beta_lm"] * z_fiber)

    columns = ["x_p_phi", "y_p_phi", "x_m_phi", "y_m_phi"]
    df.loc[:, columns] = df.loc[:, columns].multiply(df["phase_fact"], axis=0)

    return df.drop(columns=["beta_lm", "phase_fact"])


def free_propagate_asm_scalar(E_component_in, z, L, lambda_0):
    """
    Propagate a single scalar component, over a distance z using
    the Angular Spectrum Method (ASM)

    Args:
        E_component_in (np.ndarray): 2D array (N x N) of the complex field at z=0.
        z (float): Propagation distance.
        L (float): Physical size of the grid.
        lambda_0 (float): Wavelength in vacuum.

    Returns:
        np.ndarray: 2D array (N x N) of the propagated complex field at z.
    """
    N = E_component_in.shape[0]
    k0 = 2 * np.pi / lambda_0

    # --- Setup spatial frequency grid (kx, ky) ---
    dx = L / N
    dk = 2 * np.pi / L
    kx_v = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky_v = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx_v, ky_v)

    # --- Compute the FFT of the input field ---
    # (No need for shift/ifftshift if using fftfreq)
    A_in = np.fft.fft2(E_component_in)

    # --- Compute the propagator in k-space ---
    k_transverse_sq = KX**2 + KY**2
    kz_sq = k0**2 - k_transverse_sq
    kz = np.sqrt(kz_sq.astype(complex))
    # NOTE: here it is better to specify the type since it should never return
    #       'nan' for negative numbers but always the complex result

    H = np.exp(1j * kz * z)

    # --- Apply propagator and inverse FFT ---
    A_out = A_in * H
    E_out = np.fft.ifft2(A_out)

    return E_out


def free_propagate_asm_scalar_aliasing_robust(E_component_in, z, L_orig, lambda_0, NA, Rz_factor = 1):
    """
    Perform aliasing-robust scalar angular spectrum method (ASM) free-space 
    propagation of a 2D complex field. This method pads the input field to 
    reduce aliasing effects during propagation.

    Parameters:
        E_component_in (np.ndarray): Input 2D complex field.
        z (float): Propagation distance.
        L_orig (float): Original physical size of the input field.
        lambda_0 (float): Wavelength of the light.
        NA (float): Numerical aperture.

    Returns:
        tuple: 
            - E_out_cropped (np.ndarray): Cropped output 2D complex field.
            - float: Half of the physical size of the output field.
    """

    N_orig = E_component_in.shape[0]
    dx = L_orig / N_orig

    R_z = NA * z * Rz_factor
    L_pad = max(L_orig, 2*R_z)

    # maintain approximately same dx if possible
    # here the maximum 1e4x1e4 require peak 9.6GB of ram
    N_pad = int(L_pad/dx)
    if N_pad > 1.1e4:
        print(f"Propagation distance z={z} requires too many resources.")
        return None, None

    # Find the center index
    start_idx = (N_pad - N_orig) // 2
    end_idx = start_idx + N_orig

    # --- Create the padded field ---
    E_in_padded = np.zeros((N_pad, N_pad), dtype=complex)
    E_in_padded[start_idx:end_idx, start_idx:end_idx] = E_component_in

    E_out_padded = free_propagate_asm_scalar(
        E_in_padded, 
        z, 
        L_pad,
        lambda_0
    )

    # Crop the output field to the region of interest
    crop_start_idx = max(0, (N_pad // 2) - int(L_pad/2 / dx))
    crop_end_idx = min(N_pad, (N_pad // 2) + int(L_pad/2 / dx))

    E_out_cropped = E_out_padded[crop_start_idx:crop_end_idx, crop_start_idx:crop_end_idx]
    L_out = E_out_cropped.shape[0] * dx

    return E_out_cropped, L_out/2