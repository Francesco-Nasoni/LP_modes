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
    if N_pad > 1.2e4:
        print(f"Propagation distance z={z} requires too many resources (N_pad = {N_pad})")
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



def free_propagation_swag(guided_modes, df_coeff, z, NA, Rz_factor, N_x, N_k, fiber_V, radius=1.0, lambda_0=1.0):
    from scipy.special import jv, kn
    from scipy.integrate import simpson

    def analytical_hankel_core(l, u, a, k_grid):
        """
        Analytical Hankel Transform of the Core field (J_l) from 0 to a.
        Uses the finite Lommel Integral.
        """
        U = u / a
        
        # Denominator: u^2 - k^2
        # Handle singularity at k = u/a with a small epsilon
        denom = U**2 - k_grid**2
        denom[np.abs(denom) < 1e-12] = 1e-12
        
        # Formula: (a / (u^2-k^2)) * [ u*J_{l+1}(ua)*J_l(ka) - k*J_l(ua)*J_{l+1}(ka) ]
        term = (a / denom) * (
            U * jv(l + 1, U * a) * jv(l, k_grid * a) 
            - k_grid * jv(l, U * a) * jv(l + 1, k_grid * a)
        )
        return term

    def analytical_hankel_cladding(l, w, a, k_grid):
        """
        Analytical Hankel Transform of the Cladding field (K_l) from a to infinity.
        Uses the Lommel Integral adapted for K functions.
        
        """
        gamma = w / a
        
        # Denominator: w^2 + k^2
        denom = gamma**2 + k_grid**2
        
        # Formula: (a / (w^2+k^2)) * [ w/a * J_l(ka) * K_{l+1}(w) - k * J_{l+1}(ka) * K_l(w) ]
        term = (a / denom) * (
            gamma * jv(l, k_grid * a) * kn(l + 1, w) 
            - k_grid * jv(l + 1, k_grid * a) * kn(l, w)
        )
        return term

    def get_normalization_factor(l, u, w, a):
        """
        Computes the normalization constant N such that Integral(|E|^2 dA) = 1
        for the spatial mode profile R(r).
        """
        # Core Integral: Int(J_l^2(ur/a) r dr) from 0 to a
        int_core = (a**2 / 2) * (jv(l, u)**2 - jv(l-1, u) * jv(l+1, u))
        
        # Cladding Integral: Int(K_l^2(wr/a) r dr) from a to inf
        int_clad = (a**2 / 2) * (kn(l-1, w) * kn(l+1, w) - kn(l, w)**2)
        
        # Continuity factor B at interface: J_l(u) / K_l(w)
        B = jv(l, u) / kn(l, w)
        
        # Total Power = 2*pi * (Core_Int + B^2 * Clad_Int)
        total_norm_sq = 2 * np.pi * (int_core + B**2 * int_clad)
        
        return np.sqrt(total_norm_sq)
    
    # Coordinates in position space
    R_z = NA * z * Rz_factor
    x = np.linspace(-R_z, R_z, N_x)
    y = np.linspace(-R_z, R_z, N_x)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    # Coordinate in k_space
    # Since we are not using fft we can use as many point as we want
    k0 = 2 * np.pi / lambda_0
    k_max = k0 * 2 + 10/radius  # Go slightly beyond k0 to capture evanescent tails
    k_grid = np.linspace(1e-5, k_max, N_k)

    # --- 3. Pre-cooked ropagator (1D) ---
    # H(k) = exp(i * z * sqrt(k0^2 - k^2))
    kz_sq = (k0**2 - k_grid**2).astype(complex)
    kz = np.sqrt(kz_sq)
    propagator = np.exp(1j * kz * z)
    
    # --- 4. Accumulate Fields ---
    E_final_x = np.zeros_like(R, dtype=complex)
    E_final_y = np.zeros_like(R, dtype=complex)
    
    # Pre-compute a 1D radial axis for interpolation (speed optimization)
    r_1d = np.linspace(0, R_z * np.sqrt(2), N_x)

    for mode in guided_modes:
        if mode is None:
            continue

        l = mode["l"]
        m = mode["m"]
        u = mode['u']
        w = np.sqrt(fiber_V**2 - u**2)

        coeffs = df_coeff.loc[l, m]

        B = jv(l, u) / kn(l, w)
        norm_factor = get_normalization_factor(l, u, w, radius)

        # Analitically computed Hankel transform in the core and in the clad
        F_core = analytical_hankel_core(l, u, radius, k_grid)
        F_clad = analytical_hankel_cladding(l, w, radius, k_grid)

        # Total Hankel transform F_k
        F_k = F_core + B * F_clad

        F_k /= norm_factor

        # Application of the propagator
        F_k_prop = F_k * propagator

        # Numerical inverse Hankel function
        # f(r, z) = Integral [ F(k) * J_l(kr) * k dk ]

        # Compute the integrand and integrate through simpson
        bessel_term = jv(l, k_grid[None, :] * r_1d[:, None])
        integrand = F_k_prop[None, :] * bessel_term * k_grid[None, :]
        f_r_prop = simpson(integrand, x=k_grid, axis=1)

        # Interpolation
        field_envelope = np.interp(R, r_1d, f_r_prop)

        # --- Reconstruct Angular Dependence & Polarization ---
        ang_p = np.exp(1j * l * PHI)
        ang_m = np.exp(-1j * l * PHI)
        
        # Add contributions to X and Y fields
        E_final_x += field_envelope * (coeffs["x_p_phi"] * ang_p + coeffs["x_m_phi"] * ang_m)
        E_final_y += field_envelope * (coeffs["y_p_phi"] * ang_p + coeffs["y_m_phi"] * ang_m)
    
    return E_final_x, E_final_y, R_z









       



        



