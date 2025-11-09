import numpy as np
import pandas as pd
import warnings

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


# ==============================================================================
# 2. FREE-SPACE PROPAGATION KERNELS
# (All kernels are unchanged)
# ==============================================================================

def free_propagate_asm_scalar(E_component_in, z, L, lambda_0):
    """
    (Kernel 1: ASM) Propagate a single scalar component, over a distance z.
    (Unchanged)
    """
    N = E_component_in.shape[0]
    k0 = 2 * np.pi / lambda_0
    dx = L / N
    kx_v = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky_v = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx_v, ky_v)
    A_in = np.fft.fft2(E_component_in)
    k_transverse_sq = KX**2 + KY**2
    kz_sq = k0**2 - k_transverse_sq
    kz = np.sqrt(kz_sq.astype(complex))
    H = np.exp(1j * kz * z)
    A_out = A_in * H
    E_out = np.fft.ifft2(A_out)
    return E_out


def free_propagate_fresnel_scalar(E_component_in, z, L, lambda_0):
    """
    (Kernel 2: Fresnel) Propagate a component using the Fresnel 
    (paraxial) approximation via the Transfer Function method.
    (Unchanged)
    """
    N = E_component_in.shape[0]
    k0 = 2 * np.pi / lambda_0
    dx = L / N
    kx_v = np.fft.fftfreq(N, dx) * 2 * np.pi
    ky_v = np.fft.fftfreq(N, dx) * 2 * np.pi
    KX, KY = np.meshgrid(kx_v, ky_v)
    A_in = np.fft.fft2(E_component_in)
    k_transverse_sq = KX**2 + KY**2
    H = np.exp(1j * k0 * z) * np.exp(-1j * z * k_transverse_sq / (2 * k0))
    A_out = A_in * H
    E_out = np.fft.ifft2(A_out)
    return E_out


def free_propagate_fraunhofer_scalar(E_component_in, z, L_in, lambda_0):
    """
    (Kernel 3: Fraunhofer) Propagate a component using the
    Fraunhofer (far-field) approximation.
    (Unchanged)
    """
    N = E_component_in.shape[0]
    k0 = 2 * np.pi / lambda_0
    dx_in = L_in / N
    # L_out is determined by physics (FFT properties)
    L_out = (lambda_0 * z) / dx_in 
    dx_out = L_out / N 
    x_v_out = np.linspace(-L_out / 2, L_out / 2 - dx_out, N)
    X_out, Y_out = np.meshgrid(x_v_out, x_v_out)
    R_sq_out = X_out**2 + Y_out**2
    prefactor = (np.exp(1j * k0 * z) / (1j * lambda_0 * z))
    quadratic_phase = np.exp(1j * k0 * R_sq_out / (2 * z))
    A_in = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(E_component_in))) * (dx_in**2)
    E_out = prefactor * quadratic_phase * A_in
    return E_out, L_out


# ==============================================================================
# 3. HYBRID PROPAGATION MANAGER (3-STAGE, ALIASING-FIXED)
# ==============================================================================

def free_propagate_asm_scalar_hybrid(
    E_component_in, 
    z, 
    L_orig, 
    lambda_0, 
    NA, 
    Rz_factor=1,
    asm_fresnel_threshold=0.528, # z_th_1 = 400
    fraunhofer_threshold=0.1,    # z_th_2 = 2112
    N_pad_limit=11000,           # Max grid size for padding
    verbose=True
):
    """
    Perform scalar free-space propagation using a 3-stage hybrid method.
    
    *** MODIFIED LOGIC (Fraunhofer) ***
    1. Padding logic uses 'dx_match' (prevents undersampling),
       capped at N_pad_limit (default 11000).
    2. Output grid is *then* cropped to a physical size of 2*Rz.
    """
    
    # --- Handle z=0 case (no propagation) ---
    if z == 0:
        if verbose:
            print("--- Using Hybrid Propagator (z=0, no propagation) ---")
        return E_component_in, L_orig / 2

    # --- Calculate Fresnel number & common params ---
    fresnel_number = (L_orig**2) / (lambda_0 * z)
    N_orig = E_component_in.shape[0]
    dx_orig = L_orig / N_orig
    
    # --- Calculate R_z (needed for crop in Stage 3 and padding in Stage 1/2) ---
    R_z = NA * z * Rz_factor
    

    # --- STAGE 3: Far-Field (Fraunhofer) ---
    if fresnel_number < fraunhofer_threshold:
        if verbose:
            print(f"--- Using Fraunhofer propagation (F={fresnel_number:.2e} < {fraunhofer_threshold}) ---")
            
        # --- PADDING LOGIC (dx_match) ---
        # 1. Calculate N required to match dx_out = dx_orig
        N_pad_ideal_for_dx_match = int(np.ceil((lambda_0 * z) / (dx_orig**2)))
        
        # ++++++++++
        #  FIX: Removed N_max_fraun, using N_pad_limit directly
        # ++++++++++
        if N_pad_ideal_for_dx_match <= N_pad_limit:
            N_pad_fraun = N_pad_ideal_for_dx_match
        else:
            N_pad_fraun = N_pad_limit
        
        N_pad_fraun = int(max(N_pad_fraun, N_orig))
        
        if verbose:
            print(f"--- (dx_match logic: N_ideal={N_pad_ideal_for_dx_match}, "
                  f"Capped to N_pad={N_pad_fraun} by N_pad_limit) ---")
        # --- END PADDING ---
        
        if N_pad_fraun <= N_orig:
            E_in_fraun = E_component_in
            L_in_fraun = L_orig
        else:
            # Perform standard zero-padding
            start_idx = (N_pad_fraun - N_orig) // 2
            end_idx = start_idx + N_orig
            E_in_fraun = np.zeros((N_pad_fraun, N_pad_fraun), dtype=complex)
            E_in_fraun[start_idx:end_idx, start_idx:end_idx] = E_component_in
            L_in_fraun = N_pad_fraun * dx_orig
            
        # Call kernel with (dynamically) padded input
        E_out_padded, L_out = free_propagate_fraunhofer_scalar(
            E_in_fraun, 
            z,
            L_in_fraun,
            lambda_0
        )
        
        # --- CROPPING LOGIC (Per user request: Rz x Rz) ---
        # We interpret "Rz x Rz" as a window of total width 2*Rz
        
        # 1. Find output sample spacing
        dx_out = L_out / N_pad_fraun
        
        # 2. Define target crop size (physical)
        L_crop_target = 2 * R_z 
        
        # 3. Find target crop size (pixels)
        N_crop = int(np.ceil(L_crop_target / dx_out))
        if N_crop % 2 != 0: N_crop += 1 # Ensure even for centering
            
        # 4. Check if crop is necessary (if target is smaller than grid)
        if N_crop >= N_pad_fraun:
            if verbose:
                print(f"--- (Crop target N={N_crop} >= grid N={N_pad_fraun}. "
                      "Returning full grid.) ---")
            return E_out_padded, L_out / 2
        else:
            # 5. Perform central crop
            center_idx = N_pad_fraun // 2
            half_crop = N_crop // 2
            start = center_idx - half_crop
            end = center_idx + half_crop
            
            E_out_cropped = E_out_padded[start:end, start:end]
            L_cropped_final = N_crop * dx_out
            
            if verbose:
                print(f"--- (Cropped output to {N_crop}x{N_crop} grid, "
                      f"L_final={L_cropped_final:.2f}) ---")
                
            return E_out_cropped, L_cropped_final / 2
        

    # --- STAGE 1 & 2: Near/Mid-Field (ASM or Fresnel) ---
    else:
        # --- Start Padding Logic (for aliasing) ---
        L_pad_ideal = max(L_orig, 2 * R_z)
        dx_pad = dx_orig
        N_pad_ideal = max(N_orig, int(np.ceil(L_pad_ideal / dx_pad)))
        # ++++++++++
        #  NOTE: This block now also uses N_pad_limit (default 11000)
        # ++++++++++
        N_pad = int(min(N_pad_ideal, N_pad_limit))
        L_pad = N_pad * dx_pad 

        if N_pad == N_pad_limit and N_pad_ideal > N_pad_limit:
            warnings.warn(
                f"Propagation at z={z:.1f} requires N={N_pad_ideal} samples. "
                f"Clamped to N_pad={N_pad} by N_pad_limit. "
                "Result may suffer from aliasing."
            )
        
        # --- No-padding case ---
        if N_pad == N_orig:
            if verbose:
                print(f"--- (N_pad={N_pad}, L_pad={L_pad:.2f}, No Padding) ---")
            
            if fresnel_number >= asm_fresnel_threshold:
                # STAGE 1 (ASM)
                if verbose:
                    print(f"--- Using ASM propagation (F={fresnel_number:.2e} >= {asm_fresnel_threshold}) ---")
                E_out = free_propagate_asm_scalar(E_component_in, z, L_orig, lambda_0)
            else:
                # STAGE 2 (Fresnel)
                if verbose:
                    print(f"--- Using Fresnel propagation (F={fresnel_number:.2e} < {asm_fresnel_threshold}) ---")
                E_out = free_propagate_fresnel_scalar(E_component_in, z, L_orig, lambda_0)
            
            return E_out, L_orig / 2

        # --- Padded case ---
        if verbose:
            print(f"--- (N_pad={N_pad}, L_pad={L_pad:.2f}, Padded) ---")
            
        start_idx = (N_pad - N_orig) // 2
        end_idx = start_idx + N_orig
        E_in_padded = np.zeros((N_pad, N_pad), dtype=complex)
        E_in_padded[start_idx:end_idx, start_idx:end_idx] = E_component_in

        # --- Propagate the PADDED field using the chosen kernel ---
        if fresnel_number >= asm_fresnel_threshold:
            # STAGE 1 (ASM)
            if verbose:
                print(f"--- Using Padded ASM propagation (F={fresnel_number:.2e} >= {asm_fresnel_threshold}) ---")
            E_out_padded = free_propagate_asm_scalar(
                E_in_padded, z, L_pad, lambda_0
            )
        else:
            # STAGE 2 (Fresnel)
            if verbose:
                print(f"--- Using Padded Fresnel propagation (F={fresnel_number:.2e} < {asm_fresnel_threshold}) ---")
            E_out_padded = free_propagate_fresnel_scalar(
                E_in_padded, z, L_pad, lambda_0
            )
        
        # L_out for ASM/Fresnel is just L_pad
        return E_out_padded, L_pad / 2