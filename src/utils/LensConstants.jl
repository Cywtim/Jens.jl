module LensConstants

    export ARCSEC_PER_RAD, ARCSEC2_TO_RAD2, MAS_TO_RAD
    export C_LIGHT_MPC_S, MPC_TO_CM
    export DEFAULT_H0, DEFAULT_OMEGA_M
    export YR_TO_SEC, DAY_TO_SEC
    export G_OVER_C2, M_SUN_G, H0_OVER_C_MPC

    # ═══════════════════════════════════════════════════════════════
    #  Angular unit conversions
    # ═══════════════════════════════════════════════════════════════

    """
        ARCSEC_PER_RAD

    Number of arcseconds per radian: 180 × 3600 / π ≈ 206265.
    """
    const ARCSEC_PER_RAD = 180 * 3600 / π

    """
        ARCSEC2_TO_RAD2

    Conversion factor from arcsec² to radian²: (π/180/3600)².
    """
    const ARCSEC2_TO_RAD2 = (π / 180 / 3600)^2

    """
        MAS_TO_RAD

    Conversion factor from milliarcseconds to radians.
    """
    const MAS_TO_RAD = π / 180 / 3600 / 1000

    # ═══════════════════════════════════════════════════════════════
    #  Physical constants (lensing-convenient units)
    # ═══════════════════════════════════════════════════════════════

    """
        C_LIGHT_MPC_S

    Speed of light in Mpc/s ≈ 9.716×10⁻¹⁵.
    Used in time-delay: Δt = D_Δt · τ · (arcsec→rad)² / c.
    """
    const C_LIGHT_MPC_S = 9.71561189025635e-15

    """
        MPC_TO_CM

    1 Mpc in cm: 3.0857×10²⁴.
    """
    const MPC_TO_CM = 3.085677581e24

    """
        G_OVER_C2

    Gravitational constant over c² in cgs: G/c² = 7.426×10⁻²⁹ cm g⁻¹.
    Appears in the Einstein radius formula:
        θ_E² = (4G/c²) · M · D_ls / (D_l · D_s).
    """
    const G_OVER_C2 = 6.67430e-8 / (2.99792458e10)^2

    """
        M_SUN_G

    Solar mass in grams: 1.9885×10³³ g.
    """
    const M_SUN_G = 1.98847e33

    # ═══════════════════════════════════════════════════════════════
    #  Time units
    # ═══════════════════════════════════════════════════════════════

    """
        YR_TO_SEC

    One Julian year in seconds: 365.25 × 86400.
    """
    const YR_TO_SEC = 365.25 * 86400

    """
        DAY_TO_SEC

    One day in seconds: 86400.
    """
    const DAY_TO_SEC = 86400

    # ═══════════════════════════════════════════════════════════════
    #  Cosmological scaling
    # ═══════════════════════════════════════════════════════════════

    """
        H0_OVER_C_MPC

    H₀ / c in Mpc⁻¹, with H₀ = 100 h km/s/Mpc.
    Useful for converting between Hubble units and physical scales:
        H₀=70 km/s/Mpc → h=0.7 → H₀/c = h · H0_OVER_C_MPC.
    """
    const H0_OVER_C_MPC = 100 / 2.99792458e5

    # ═══════════════════════════════════════════════════════════════
    #  Cosmological defaults
    # ═══════════════════════════════════════════════════════════════

    """
        DEFAULT_H0

    Default Hubble constant (km/s/Mpc).
    """
    const DEFAULT_H0 = 70.0

    """
        DEFAULT_OMEGA_M

    Default matter density parameter.
    """
    const DEFAULT_OMEGA_M = 0.3

end