"""
    PseudoJaffe — Tidally Truncated Subhalo

3D density:  ρ(r) ∝ 1 / (r² (r² + r_t²))

Convergence:  κ(R) = (θ_E/2)·[1/R − 1/√(R² + r_t²)]

A physically motivated subhalo model with finite total mass.
Used extensively for dark-matter substructure detection in
strong lensing.  Reduces to SIS as r_t → ∞.

# Parameters
- `theta_E`: Einstein radius (SIS limit as r_t → ∞) [arcsec]
- `r_t`: tidal / truncation radius [arcsec]
- `xcentre`, `ycentre`: centre [arcsec]

# References
- Muñoz+2001 (ApJ 546, 769) — pseudo-Jaffe derivation
- Minor+2016, arXiv:1612.05250 — subhalo perturbation scale
- Vegetti+2023, arXiv:2306.11781 — subhalo lensing review

# Example
    lens = SingleModel(PseudoJaffe; theta_E=0.1, r_t=0.5)
"""
module PseudoJaffe

    export LensCheck, LensPotential, LensDerivative, LensHessian

    # ═══════════════════════════════════════════════════════════════
    #  1. PARAMETER VALIDATION
    # ═══════════════════════════════════════════════════════════════

    function LensCheck(; theta_E::Real, r_t::Real,
                        xcentre::Real=0., ycentre::Real=0.)
        @assert theta_E > 0  "theta_E must be positive, got $theta_E"
        @assert r_t > 0      "r_t must be positive, got $r_t"
    end

    # ═══════════════════════════════════════════════════════════════
    #  2. RADIAL DERIVATIVES — pure functions, GPU-safe
    #
    #  All radial functions operate on shifted + safe-clamped R.
    #  Broadcasting via @. ensures CuArray compatibility.
    #
    #  Derivation:
    #    κ(R) = (θ_E/2)·[1/R − 1/√(R² + r_t²)]
    #    M(<R) ∝ ∫₀ᴿ κ(R′)R′ dR′ = (θ_E/2)·[R + r_t − √(R² + r_t²)]
    #    α(R) = 2·M(<R)/(π·R) = θ_E·[R + r_t − √(R² + r_t²)]/R
    #    dα/dR = θ_E·[r_t²/(R²·√) − r_t/R²]
    # ═══════════════════════════════════════════════════════════════

    # ── convergence κ(R) ──────────────────────────────────────
    @inline function _kappa(R, theta_E, r_t)
        T = eltype(R)
        # κ(R) = (θ_E/2) · [1/R − 1/√(R²+r_t²)]
        return @. (theta_E / T(2)) * (T(1)/R - T(1)/sqrt(R^2 + r_t^2))
    end

    # ── radial deflection α(R) ────────────────────────────────
    @inline function _alpha_r(R, theta_E, r_t)
        # α(R) = θ_E·[R + r_t − √(R²+r_t²)] / R
        term = @. sqrt(R^2 + r_t^2)
        return @. theta_E * (R + r_t - term) / R
    end

    # ── dα/dR ─────────────────────────────────────────────────
    @inline function _dalpha_dR(R, theta_E, r_t)
        # dα/dR = θ_E·[r_t²/(R²·√(R²+r_t²)) − r_t/R²]
        R2 = @. R^2
        term = @. sqrt(R2 + r_t^2)
        return @. theta_E * (r_t^2 / (R2 * term) - r_t / R2)
    end

    # ═══════════════════════════════════════════════════════════════
    #  3. LENS POTENTIAL — ψ(R)
    #
    #    ψ(R) = θ_E·[R − √(R²+r_t²) + r_t·log((√(R²+r_t²)+r_t) / (2r_t))]
    #
    #  Verified: dψ/dR = α(R) ✓
    #  Limits:  ψ(0) → 0,  ψ(∞) → θ_E·r_t·log(R/r_t) (log-divergent, like SIS)
    # ═══════════════════════════════════════════════════════════════

    function LensPotential(x, y; theta_E::Real, r_t::Real,
                           xcentre::Real=0., ycentre::Real=0.)
        xsh = @. x - xcentre
        ysh = @. y - ycentre
        T = promote_type(eltype(x), typeof(theta_E), typeof(r_t))
        R = @. max(sqrt(xsh^2 + ysh^2), eps(T))

        term = @. sqrt(R^2 + r_t^2)
        psi = @. theta_E * (R - term + r_t * log((term + r_t) / (T(2) * r_t)))
        return psi
    end

    # ═══════════════════════════════════════════════════════════════
    #  4. DEFLECTION ANGLE
    #
    #    α_x = α(R)·x/R,  α_y = α(R)·y/R
    # ═══════════════════════════════════════════════════════════════

    function LensDerivative(x, y; theta_E::Real, r_t::Real,
                            xcentre::Real=0., ycentre::Real=0.)
        xsh = @. x - xcentre
        ysh = @. y - ycentre
        T = promote_type(eltype(x), typeof(theta_E), typeof(r_t))
        R = @. max(sqrt(xsh^2 + ysh^2), eps(T))

        aR_over_R = @. _alpha_r(R, theta_E, r_t) / R
        f_x = @. aR_over_R * xsh
        f_y = @. aR_over_R * ysh

        return f_x, f_y
    end

    # ═══════════════════════════════════════════════════════════════
    #  5. HESSIAN
    #
    #  Cartesian from radial derivatives (more stable than κ+γ form):
    #    f_xx = (dα/dR)·cos²φ + (α/R)·sin²φ
    #    f_xy = (dα/dR − α/R)·sinφ·cosφ
    #    f_yy = (dα/dR)·sin²φ + (α/R)·cos²φ
    #
    #  At R→0 (center): α/R and dα/dR both diverge in opposite
    #  directions but the Cartesian components must be finite.
    #  Per-direction, we clamp and accept the ~1eV precision
    #  at grid points accidentally hitting the exact center.
    # ═══════════════════════════════════════════════════════════════

    function LensHessian(x, y; theta_E::Real, r_t::Real,
                         xcentre::Real=0., ycentre::Real=0.)
        xsh = @. x - xcentre
        ysh = @. y - ycentre
        T = promote_type(eltype(x), typeof(theta_E), typeof(r_t))
        R = @. max(sqrt(xsh^2 + ysh^2), eps(T))

        # Radial quantities
        alpha_over_R = @. _alpha_r(R, theta_E, r_t) / R
        dalpha_dR    = @. _dalpha_dR(R, theta_E, r_t)

        # Angular factors
        cos_phi = @. xsh / R
        sin_phi = @. ysh / R
        cos2 = @. cos_phi^2
        sin2 = @. sin_phi^2
        sincos = @. sin_phi * cos_phi

        # Cartesian Hessian
        f_xx = @. dalpha_dR * cos2 + alpha_over_R * sin2
        f_xy = @. (dalpha_dR - alpha_over_R) * sincos
        f_yy = @. dalpha_dR * sin2 + alpha_over_R * cos2

        return f_xx, f_xy, f_yy
    end

end # module PseudoJaffe