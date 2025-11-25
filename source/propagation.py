import numpy as np
import pandas as pd
from scipy.special import jv, kn
from scipy.integrate import simpson, trapezoid


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


def free_propagation_asm_hankel(
    guided_modes,
    df_coeff,
    z,
    NA,
    Rz_factor,
    N_x,
    fiber_V,
    R_origin,
    min_point_per_period=10,
    radius=1.0,
    lambda_0=1.0,
):

    def analytical_hankel_core(l, u, a, k):
        """
        Analytical Hankel Transform of the Core field (J_l) from 0 to a.
        Uses the finite Lommel Integral.
        """
        h = u / a

        # Denominator: u^2 - k^2
        # Handle singularity at k = u/a with a small epsilon
        denom = h**2 - k**2
        denom[np.abs(denom) < 1e-12] = 1e-12

        # Formula: (a / (h^2-k^2)) * [ u*J_{l+1}(ha)*J_l(ka) - k*J_l(ha)*J_{l+1}(ka) ]
        term = (a / denom) * (
            h * jv(l + 1, u) * jv(l, k * a)
            - k * jv(l, u) * jv(l + 1, k * a)
        )
        return term

    def analytical_hankel_cladding(l, w, a, k):
        """
        Analytical Hankel Transform of the Cladding field (K_l) from a to infinity.
        Uses the Lommel Integral adapted for K functions.

        """
        q = w / a

        # Denominator: q^2 + k^2
        denom = q**2 + k**2

        # Formula: (a / (w^2+k^2)) * [ w/a * J_l(ka) * K_{l+1}(w) - k * J_{l+1}(ka) * K_l(w) ]
        term = (a / denom) * (
            q * jv(l, k * a) * kn(l + 1, w)
            - k * jv(l + 1, k * a) * kn(l, w)
        )
        return term

    def get_normalization_factor(l, u, w, a):
        """
        Computes the normalization constant N such that Integral(|E|^2 dA) = 1
        for the spatial mode profile R(r).
        """
        # Core Integral: Int(J_l^2(ur/a) r dr) from 0 to a
        int_core = (a**2 / 2) * (jv(l, u) ** 2 - jv(l - 1, u) * jv(l + 1, u))

        # Cladding Integral: Int(K_l^2(wr/a) r dr) from a to inf
        int_clad = (a**2 / 2) * (kn(l - 1, w) * kn(l + 1, w) - kn(l, w) ** 2)

        # Continuity factor B at interface: J_l(u) / K_l(w)
        B = jv(l, u) / kn(l, w)

        # Total Power = 2*pi * (Core_Int + B^2 * Clad_Int)
        total_norm_sq = 2 * np.pi * (int_core + B**2 * int_clad)

        return np.sqrt(total_norm_sq)

    # Coordinates in position space
    R_z = NA * z * Rz_factor
    R_z = max(R_origin, R_z)
    x = np.linspace(-R_z, R_z, N_x)
    y = np.linspace(-R_z, R_z, N_x)
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)
    PHI = np.arctan2(Y, X)

    # Coordinate in k_space
    # Since we are not using fft we can use as many point as we want
    k0 = 2 * np.pi / lambda_0
    k_max = max(
        k0 * 4 * NA, 10 / radius
    )

    # * Calculating N_k to have {min_point_per_period} point per period for k_max
    # Asymptotic J_l(ax) ~ cos (ax + φ) -> λ=2π/a
    # Max period when calculating Hankel transform
    #   R_max = R_z * √2  (accounting for the corners)
    #   λ_max = 2π/R_max
    #   Δx =  λ_max/min_point_per_period = 2π/R_max/min_point_per_period
    #   N_k = (k_max / Δx)

    N_k = int(np.ceil(k_max / (2 * np.pi / R_z / np.sqrt(2) / min_point_per_period)))
    print(f"Hankel transforms will be applaied to {N_k} k-points")
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
        u = mode["u"]
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
        E_final_x += field_envelope * (
            coeffs["x_p_phi"] * ang_p + coeffs["x_m_phi"] * ang_m
        )
        E_final_y += field_envelope * (
            coeffs["y_p_phi"] * ang_p + coeffs["y_m_phi"] * ang_m
        )

    return E_final_x, E_final_y, R_z
