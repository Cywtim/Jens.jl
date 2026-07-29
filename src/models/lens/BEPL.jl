"""
    BEPL — Broken Elliptical Power Law

3D density:  ρ(r) ∝ r^(-γ_in) · [1 + (r/r_t)^α]^((γ_in-γ_out)/α)

Power-law elliptical lens with a smooth break in the 3D density slope
at transition radius r_t.  Reduces to EPL when γ_in = γ_out.

Asymptotic behaviour:
- r ≪ r_t :  ρ ∝ r^(-γ_in)     (inner power-law)
- r ≫ r_t :  ρ ∝ r^(-γ_out)    (outer power-law)

The transition sharpness is controlled by α (default 4, following the
Baltz+2009 smooth truncation convention).  α → ∞ gives a sharp break.

Radial deflection and convergence are precomputed on a log-spaced grid
via numerical line-of-sight integration, then interpolated at render
time (GPU-safe, zero scalar indexing).  Ellipticity is applied via
coordinate distortion, consistent with NFWE.

# Parameters
- `theta_E`: Einstein radius [arcsec] — sets the overall mass scale
- `gamma_in`: inner 3D density slope (γ_in=2 → isothermal inner)
- `gamma_out`: outer 3D density slope (γ_out=2 → isothermal outer)
- `r_t`: transition radius [arcsec]
- `alpha_trans`: transition sharpness (default 4, range 2–10)
- `e1`, `e2`: ellipticity components
- `xcentre`, `ycentre`: lens centre [arcsec]
- `n_radial`: number of precomputed radial samples (default 400)

# Reference
O'Riordan et al. (2021), MNRAS 504, 3335 — doi:10.1093/mnras/staa3747

# Example
    # Core steeper than outskirts (cuspy centre, shallow halo)
    lens = BEPLLens(; theta_E=1.2, gamma_in=2.3, gamma_out=1.8,
                     r_t=0.5, e1=0.1, e2=0.0)

    # Core shallower than outskirts (core + steep halo)
    lens = BEPLLens(; theta_E=1.2, gamma_in=1.5, gamma_out=2.5,
                     r_t=0.8, e1=0.15, e2=-0.05)

    # With SingleModel / CombinedLens
    sm = SingleModel(BEPL; theta_E=1.2, gamma_in=2.1, gamma_out=1.9,
                     r_t=1.0, e1=0.1, e2=0.0)
"""
# ═══════════════════════════════════════════════════════════════
#  BEPL — Broken Elliptical Power Law
#
#  3D density:
#    ρ(r) = ρ₀ · r^(-γ_in) · [1 + (r/r_t)^α]^((γ_in-γ_out)/α)
#
#  Asymptotic:
#    r ≪ r_t :  ρ ∝ r^(-γ_in)
#    r ≫ r_t :  ρ ∝ r^(-γ_out)
#
#  Surface mass density Σ(R) computed via:
#    Σ(R) = 2 ∫₀^∞ ρ(√(R²+z²)) dz     (trapezoidal, n_z samples)
#
#  Deflection α(R) from cumulative:
#    α(R) = (2/R) ∫₀ᴿ κ(R') R' dR'
#
#  Normalisation:  α(θ_E) = θ_E  (Einstein radius definition)
#
#  Ellipticity: applied via LensUtils.EllipticalDistortion,
#  consistent with NFWE.
#
#  Refs:
#    O'Riordan+2021 (MNRAS 504, 3335) — broken power-law lensing
#    Baltz+2009 (arXiv:0705.3336)     — smooth truncation convention
# ═══════════════════════════════════════════════════════════════

module BEPL

    using Jens.LensUtils: EllipticalDistortion

    import Jens.LensBase: AbstractLens, lens_derivative, lens_hessian, lens_potential, lens_check

    export BEPLLens

    # ═══════════════════════════════════════════════════════════
    #  Struct
    # ═══════════════════════════════════════════════════════════

    """
        BEPLLens(; theta_E, gamma_in, gamma_out, r_t,
                  alpha_trans=4.0, e1=0.0, e2=0.0,
                  xcentre=0.0, ycentre=0.0, n_radial=400)

    Broken Elliptical Power Law lens model.

    Precomputes radial deflection and convergence on a log-spaced grid
    at construction time.  Parameters are stored as Float64 internally
    for precision during precomputation; rendering broadcasts to JFloat.
    """
    struct BEPLLens{V<:AbstractVector{Float64}} <: AbstractLens
        theta_E::Float64
        gamma_in::Float64
        gamma_out::Float64
        r_t::Float64
        alpha_trans::Float64
        e1::Float64
        e2::Float64
        xcentre::Float64
        ycentre::Float64
        # Precomputed radial tables [log-spaced]
        R_grid::V
        alpha_grid::V
        kappa_grid::V
        psi_grid::V
    end

    function BEPLLens(; theta_E::Real, gamma_in::Real, gamma_out::Real,
                       r_t::Real, alpha_trans::Real=4.0,
                       e1::Real=0.0, e2::Real=0.0,
                       xcentre::Real=0.0, ycentre::Real=0.0,
                       n_radial::Int=400)
        theta_E_f  = Float64(theta_E)
        gamma_in_f  = Float64(gamma_in)
        gamma_out_f = Float64(gamma_out)
        r_t_f       = Float64(r_t)
        alpha_t_f   = Float64(alpha_trans)

        R_grid, alpha_grid, kappa_grid, psi_grid =
            _precompute(theta_E_f, gamma_in_f, gamma_out_f, r_t_f, alpha_t_f, n_radial)

        return BEPLLens(theta_E_f, gamma_in_f, gamma_out_f, r_t_f, alpha_t_f,
                        Float64(e1), Float64(e2),
                        Float64(xcentre), Float64(ycentre),
                        R_grid, alpha_grid, kappa_grid, psi_grid)
    end

    # ═══════════════════════════════════════════════════════════
    #  3D density
    # ═══════════════════════════════════════════════════════════

    @inline function _rho_3d(r::Float64, gamma_in::Float64, gamma_out::Float64,
                              r_t::Float64, alpha_trans::Float64)
        # ρ(r) ∝ r^(-γ_in) · [1 + (r/r_t)^α]^((γ_in-γ_out)/α)
        # Returns unnormalised density (ρ₀ = 1)
        core = r^(-gamma_in)
        x    = (r / r_t)^alpha_trans
        if x > 1e-8
            envelope = (1.0 + x)^((gamma_in - gamma_out) / alpha_trans)
        else
            envelope = 1.0  # r ≪ r_t limit
        end
        return core * envelope
    end

    # ═══════════════════════════════════════════════════════════
    #  Surface mass density Σ(R) via LOS integration
    # ═══════════════════════════════════════════════════════════

    function _sigma_at_R(R::Float64, gamma_in::Float64, gamma_out::Float64,
                          r_t::Float64, alpha_trans::Float64;
                          n_t::Int=600, t_max::Float64=10.0)
        # Σ(R) = 2 ∫_R^∞ ρ(r) r / √(r²-R²) dr     (Abel transform)
        #
        # Substitution: r = R·cosh(t),  dr = R·sinh(t) dt
        #   √(r²-R²) = R·sinh(t)
        #   r dr / √(r²-R²) = R·cosh(t)·sinh(t) / sinh(t) · R dt = R·cosh(t) dt
        #
        # So: Σ(R) = 2R ∫_0^∞ ρ(R·cosh(t))·cosh(t) dt
        #
        # The cosh substitution is far more efficient than z-integration
        # because the integrand decays as e^{-(γ_out-1)·t} for power-law
        # profiles (vs ~1/z² for the z-form).  For γ_out=2, the decay is
        # e^{-t} — 1% of peak at t=4.6, compared to z=tan(π/2-0.01)≈100R.
        dz_t = t_max / (n_t - 1)
        s = 0.0
        for j in 1:n_t
            t = (j - 1) * dz_t
            cosh_t = cosh(t)
            r = R * cosh_t
            r = max(r, 1e-10)
            rho_val = _rho_3d(r, gamma_in, gamma_out, r_t, alpha_trans)
            w = ifelse(j == 1 || j == n_t, 0.5, 1.0)
            s += w * rho_val * cosh_t
        end
        return 2.0 * R * s * dz_t
    end

    # ═══════════════════════════════════════════════════════════
    #  Precomputation
    # ═══════════════════════════════════════════════════════════

    function _precompute(theta_E::Float64, gamma_in::Float64, gamma_out::Float64,
                          r_t::Float64, alpha_trans::Float64, n::Int)
        # ── Radial grid: log-spaced ──
        R_min = max(1e-5 * min(theta_E, r_t), 1e-8)
        R_max = 8.0 * max(theta_E, r_t)
        R_grid = exp10.(range(log10(R_min), log10(R_max); length=n))

        # ── Numerical Σ(R) for each projected radius ──
        Sigma = Vector{Float64}(undef, n)
        for i in 1:n
            Sigma[i] = _sigma_at_R(R_grid[i], gamma_in, gamma_out, r_t, alpha_trans)
        end

        # ── Unnormalised κ(R) ∝ Σ(R) ──
        kappa_unnorm = copy(Sigma)

        # ── Unnormalised α(R) = (2/R) ∫₀ᴿ κ(R') R' dR' ──
        alpha_unnorm = Vector{Float64}(undef, n)
        cum = 0.0
        alpha_unnorm[1] = 0.0
        for i in 2:n
            dR = R_grid[i] - R_grid[i-1]
            cum += 0.5 * (kappa_unnorm[i] * R_grid[i] + kappa_unnorm[i-1] * R_grid[i-1]) * dR
            alpha_unnorm[i] = 2.0 * cum / R_grid[i]
        end

        # ── Normalisation: enforce α(θ_E) = θ_E ──
        alpha_at_thetaE = _interp_scalar(theta_E, R_grid, alpha_unnorm)
        if alpha_at_thetaE < 1e-15
            error("BEPL precomputation: α(θ_E) ≈ 0.  " *
                  "Check that theta_E=$theta_E is within the precomputed radial range " *
                  "[$R_min, $R_max].")
        end
        scale = theta_E / alpha_at_thetaE

        alpha_grid = alpha_unnorm .* scale
        kappa_grid = kappa_unnorm .* scale

        # ── Potential ψ(R) = ∫ α(R') dR' ──
        psi_grid = Vector{Float64}(undef, n)
        psi_grid[1] = 0.0
        for i in 2:n
            dR = R_grid[i] - R_grid[i-1]
            psi_grid[i] = psi_grid[i-1] +
                0.5 * (alpha_grid[i] + alpha_grid[i-1]) * dR
        end

        return R_grid, alpha_grid, kappa_grid, psi_grid
    end

    function _interp_scalar(R::Float64, R_grid::AbstractVector{Float64},
                             vals::AbstractVector{Float64})
        # Linear interpolation for a single scalar R
        if R <= R_grid[1]
            return vals[1]
        elseif R >= R_grid[end]
            return vals[end]
        end
        idx = searchsortedlast(R_grid, R)
        lo, hi = R_grid[idx], R_grid[idx+1]
        t = (R - lo) / (hi - lo)
        return vals[idx] + t * (vals[idx+1] - vals[idx])
    end

    # ═══════════════════════════════════════════════════════════
    #  GPU-safe linear interpolation (broadcast)
    # ═══════════════════════════════════════════════════════════

    function _interp1_vec(R::AbstractArray, R_grid::AbstractVector, vals::AbstractVector)
        T = eltype(R)
        result = fill(convert(T, NaN), size(R))
        n_bins = length(R_grid) - 1

        for i in 1:n_bins
            lo  = convert(T, R_grid[i])
            hi  = convert(T, R_grid[i+1])
            vlo = convert(T, vals[i])
            vhi = convert(T, vals[i+1])
            in_bin = @. (R >= lo) & (R < hi)
            t = @. (R - lo) / (hi - lo)
            result = @. ifelse(in_bin, vlo + t * (vhi - vlo), result)
        end

        vfirst = convert(T, vals[1])
        vlast  = convert(T, vals[end])
        result = @. ifelse(R < R_grid[1],   ifelse(R .>= 0, vfirst, zero(T)), result)
        result = @. ifelse(R >= R_grid[end], zero(T), result)

        return result
    end

    # ═══════════════════════════════════════════════════════════
    #  LensBase interface
    # ═══════════════════════════════════════════════════════════

    function lens_check(lens::BEPLLens; kwargs...)
        @assert lens.theta_E > 0   "theta_E must be positive, got $(lens.theta_E)"
        @assert lens.gamma_in > 0  "gamma_in must be positive, got $(lens.gamma_in)"
        @assert lens.gamma_out > 0 "gamma_out must be positive, got $(lens.gamma_out)"
        @assert lens.r_t > 0       "r_t must be positive, got $(lens.r_t)"
        @assert lens.alpha_trans >= 1.0 "alpha_trans must be ≥ 1, got $(lens.alpha_trans)"
        @assert abs(lens.e1) < 1.0 "|e1| must be < 1, got $(lens.e1)"
        @assert abs(lens.e2) < 1.0 "|e2| must be < 1, got $(lens.e2)"
    end

    function lens_derivative(lens::BEPLLens, x, y; kwargs...)
        # Apply ellipticity: map to circular equivalent radius
        xsh, ysh = EllipticalDistortion(x, y;
                      e1=lens.e1, e2=lens.e2,
                      xcentre=lens.xcentre, ycentre=lens.ycentre)
        R = @. max(sqrt(xsh^2 + ysh^2), eps(Float64))

        a = _interp1_vec(R, lens.R_grid, lens.alpha_grid)
        a_over_R = @. a / R

        f_x = @. a_over_R * xsh
        f_y = @. a_over_R * ysh
        return f_x, f_y
    end

    function lens_hessian(lens::BEPLLens, x, y; kwargs...)
        xsh, ysh = EllipticalDistortion(x, y;
                      e1=lens.e1, e2=lens.e2,
                      xcentre=lens.xcentre, ycentre=lens.ycentre)
        R = @. max(sqrt(xsh^2 + ysh^2), eps(Float64))

        kappa = _interp1_vec(R, lens.R_grid, lens.kappa_grid)
        a     = _interp1_vec(R, lens.R_grid, lens.alpha_grid)
        a_over_R = @. a / R

        # For axisymmetric lens: γ(R) = κ̄(<R) − κ(R) = α(R)/R − κ(R)
        gamma = @. a_over_R - kappa

        cos_phi = @. xsh / R
        sin_phi = @. ysh / R
        cos2phi = @. cos_phi^2 - sin_phi^2
        sin2phi = @. 2 * sin_phi * cos_phi

        f_xx = @. kappa + gamma * cos2phi
        f_yy = @. kappa - gamma * cos2phi
        f_xy = @. gamma * sin2phi

        return f_xx, f_xy, f_yy
    end

    function lens_potential(lens::BEPLLens, x, y; kwargs...)
        xsh, ysh = EllipticalDistortion(x, y;
                      e1=lens.e1, e2=lens.e2,
                      xcentre=lens.xcentre, ycentre=lens.ycentre)
        R = @. max(sqrt(xsh^2 + ysh^2), eps(Float64))
        return _interp1_vec(R, lens.R_grid, lens.psi_grid)
    end

    # ═══════════════════════════════════════════════════════════
    #  Module-level functions (for LensBase.Module dispatch)
    # ═══════════════════════════════════════════════════════════

    """
        BEPL.LensCheck(; theta_E, gamma_in, gamma_out, r_t, kwargs...)

    Validate BEPL parameters.  Constructs a temporary BEPLLens and delegates
    to its `lens_check`.  Supported by LensBase.Module dispatch.
    """
    function LensCheck(; theta_E::Real, gamma_in::Real, gamma_out::Real,
                        r_t::Real, alpha_trans::Real=4.0,
                        e1::Real=0.0, e2::Real=0.0,
                        xcentre::Real=0.0, ycentre::Real=0.0)
        lens = BEPLLens(; theta_E=theta_E, gamma_in=gamma_in, gamma_out=gamma_out,
                         r_t=r_t, alpha_trans=alpha_trans,
                         e1=e1, e2=e2, xcentre=xcentre, ycentre=ycentre)
        return lens_check(lens)
    end

    function LensPotential(xg::AbstractArray, yg::AbstractArray;
                            theta_E::Real, gamma_in::Real, gamma_out::Real,
                            r_t::Real, alpha_trans::Real=4.0,
                            e1::Real=0.0, e2::Real=0.0,
                            xcentre::Real=0.0, ycentre::Real=0.0)
        lens = BEPLLens(; theta_E=theta_E, gamma_in=gamma_in, gamma_out=gamma_out,
                         r_t=r_t, alpha_trans=alpha_trans,
                         e1=e1, e2=e2, xcentre=xcentre, ycentre=ycentre)
        return lens_potential(lens, xg, yg)
    end

    function LensDerivative(xg::AbstractArray, yg::AbstractArray;
                             theta_E::Real, gamma_in::Real, gamma_out::Real,
                             r_t::Real, alpha_trans::Real=4.0,
                             e1::Real=0.0, e2::Real=0.0,
                             xcentre::Real=0.0, ycentre::Real=0.0)
        lens = BEPLLens(; theta_E=theta_E, gamma_in=gamma_in, gamma_out=gamma_out,
                         r_t=r_t, alpha_trans=alpha_trans,
                         e1=e1, e2=e2, xcentre=xcentre, ycentre=ycentre)
        return lens_derivative(lens, xg, yg)
    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
                           theta_E::Real, gamma_in::Real, gamma_out::Real,
                           r_t::Real, alpha_trans::Real=4.0,
                           e1::Real=0.0, e2::Real=0.0,
                           xcentre::Real=0.0, ycentre::Real=0.0)
        lens = BEPLLens(; theta_E=theta_E, gamma_in=gamma_in, gamma_out=gamma_out,
                         r_t=r_t, alpha_trans=alpha_trans,
                         e1=e1, e2=e2, xcentre=xcentre, ycentre=ycentre)
        return lens_hessian(lens, xg, yg)
    end

end # module BEPL