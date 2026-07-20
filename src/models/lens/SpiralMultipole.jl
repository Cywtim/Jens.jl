"""
    SpiralMultipole — Density-Wave Spiral Perturbation

Convergence:  κ(R,φ) = κ₀(R) + Σₘ Aₘ(R) cos(mφ − f(R))

Density-wave theory applied to lens galaxies.  The m=0 term is
provided by a smooth MGE lens; the m≥2 terms are localized in a
Gaussian annulus around the corotation radius R_cr.  The radial
Green function is precomputed for GPU-safe interpolation.

# Parameters
- `m0_lens::MGECombinedLens`: axisymmetric (m=0) smooth component
- `m::Int`: angular mode number (2 = bisymmetric spiral)
- `amplitude`: peak amplitude A₀ of the density-wave perturbation
- `R_cr`: corotation radius [arcsec]
- `delta_R`: Gaussian width of the radial annulus [arcsec]
- `pitch_angle`: pitch angle i_p [rad]
- `R0`: reference radius for logarithmic spiral phase [arcsec]

# References
- Lin & Shu (1964) — density wave theory
- Shajib (2019), MNRAS 488, 1387 — MGE for lensing

# Example
    m0 = MGECombinedLens(...)
    lens = SpiralMultipoleLens(m0, 2, 0.05, 1.0, 0.3, 0.2, 1.0)
"""
module SpiralMultipole

    using SpecialFunctions
    using LinearAlgebra

    import Jens.LensBase: lens_derivative, lens_hessian
    using Jens.LensBase: AbstractLens
    using Jens.MGE: MGECombinedLens

    export SpiralMultipoleLens

    # ═══════════════════════════════════════════════════════════════
    #  Struct definition
    # ═══════════════════════════════════════════════════════════════

    """
        SpiralMultipoleLens(m0_lens, m, amplitude, R_cr, delta_R, pitch_angle, R0; n_radial=200)

    Density-wave multipole lens model.

    # Fields
    - `m0_lens::MGECombinedLens`: axisymmetric (m=0) smooth component
    - `m::Int`: angular mode number (2 = bisymmetric spiral)
    - `amplitude::Real`: peak amplitude A₀ of the density-wave perturbation
    - `R_cr::Real`: corotation radius where the perturbation is strongest
    - `delta_R::Real`: Gaussian width of the radial annulus
    - `pitch_angle::Real`: pitch angle i_p in radians
    - `R0::Real`: reference radius for logarithmic spiral phase

    # Precomputed (via constructor)
    - `R_grid`, `psi_m_grid`, `dpsim_dR_grid`: radial interpolation tables
      for the m-th Green function and its derivative.
    """
    struct SpiralMultipoleLens{T<:Real, V<:AbstractVector{<:AbstractFloat}} <: AbstractLens
        m0_lens::MGECombinedLens
        m::Int
        amplitude::T
        R_cr::T
        delta_R::T
        pitch_angle::T
        R0::T
        # Precomputed radial tables
        R_grid::V
        psi_m_grid::V
        dpsim_dR_grid::V

        function SpiralMultipoleLens(
            m0_lens::MGECombinedLens,
            m::Int, amplitude::Real,
            R_cr::Real, delta_R::Real,
            pitch_angle::Real, R0::Real;
            n_radial::Int=200,
        )
            @assert m >= 1 "m must be ≥ 1 (m=0 is handled by m0_lens)"
            @assert amplitude >= 0 "amplitude must be non-negative"
            @assert R_cr > 0 "R_cr must be positive"
            @assert delta_R > 0 "delta_R must be positive"

            R_grid = _build_radial_grid(Float64(R_cr), Float64(delta_R); n=n_radial)
            psi_m_grid, dpsim_dR_grid = _precompute_green(
                m, Float64(amplitude), Float64(R_cr), Float64(delta_R), R_grid)

            new{Float64, typeof(R_grid)}(
                m0_lens, m,
                Float64(amplitude), Float64(R_cr), Float64(delta_R),
                Float64(pitch_angle), Float64(R0),
                R_grid, psi_m_grid, dpsim_dR_grid)
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  Radial Green function for the m-th angular mode
    #
    #  ODE:   [1/R · d/dR(R · d/dR) − m²/R²] ψₘ(R) = 2 κₘ(R)
    #
    #  Green function solution (m ≥ 1):
    #
    #    ψₘ(R) = (1/m) [ R⁻ᵐ ∫₀ᴿ sᵐ⁺¹ κₘ(s) ds
    #                   + Rᵐ  ∫_R^∞ s¹⁻ᵐ κₘ(s) ds ]
    #
    #    dψₘ/dR = −R⁻ᵐ⁻¹ ∫₀ᴿ sᵐ⁺¹ κₘ(s) ds
    #             + Rᵐ⁻¹  ∫_R^∞ s¹⁻ᵐ κₘ(s) ds
    #
    #    κₘ(R) = A₀ · exp(−(R−R_cr)² / (2·ΔR²))
    # ═══════════════════════════════════════════════════════════════

    function _kappa_density_wave(R::Float64, A0::Float64, R_cr::Float64, delta_R::Float64)
        return A0 * exp(-(R - R_cr)^2 / (2.0 * delta_R^2))
    end

    function _build_radial_grid(R_cr::Float64, delta_R::Float64; n::Int=200)
        R_min = max(1e-4 * R_cr, 0.01 * R_cr)
        R_max = R_cr + 6.0 * delta_R
        return exp10.(range(log10(R_min), log10(R_max); length=n))
    end

    function _precompute_green(m::Int, A0::Float64, R_cr::Float64, delta_R::Float64,
                               R_grid::Vector{Float64})
        n_r = length(R_grid)
        psi = zeros(n_r)
        dpsi = zeros(n_r)

        n_fine = 2000
        s_fine = exp10.(range(log10(R_grid[1]), log10(R_grid[end]); length=n_fine))
        kappa_fine = [_kappa_density_wave(s, A0, R_cr, delta_R) for s in s_fine]

        integrand_1 = [s^(m+1) * _kappa_density_wave(s, A0, R_cr, delta_R) for s in s_fine]
        integrand_2 = [s^(1-m) * _kappa_density_wave(s, A0, R_cr, delta_R) for s in s_fine]

        I1_cum = zeros(n_fine)
        I2_cum = zeros(n_fine)
        ds = diff(s_fine)

        for i in 2:n_fine
            I1_cum[i] = I1_cum[i-1] + 0.5 * (integrand_1[i-1] + integrand_1[i]) * ds[i-1]
        end
        for i in (n_fine-1):-1:1
            I2_cum[i] = I2_cum[i+1] + 0.5 * (integrand_2[i] + integrand_2[i+1]) * ds[i]
        end

        for (j, R) in enumerate(R_grid)
            idx = searchsortedfirst(s_fine, R)
            idx = clamp(idx, 1, n_fine - 1)
            I1 = I1_cum[idx]
            I2 = I2_cum[idx]

            psi[j] = (1.0 / m) * (R^(-m) * I1 + R^m * I2)
            dpsi[j] = -R^(-m-1) * I1 + R^(m-1) * I2
        end

        psi[1] = 0.0
        dpsi[1] = 0.0

        return psi, dpsi
    end

    # ═══════════════════════════════════════════════════════════════
    #  Spiral phase function: f(R) for logarithmic spiral
    #
    #    f(R) = m / tan(i_p) · ln(R / R₀)
    #    f'(R) = m / (R · tan(i_p))
    # ═══════════════════════════════════════════════════════════════

    @inline _spiral_phase(R::Real, m::Int, pitch_angle::Real, R0::Real) =
        m / tan(pitch_angle) * log(R / R0)

    @inline _spiral_phase_deriv(R::Real, m::Int, pitch_angle::Real) =
        m / (R * tan(pitch_angle))

    # ═══════════════════════════════════════════════════════════════
    #  Vectorized linear interpolation — GPU-safe via broadcast bins
    # ═══════════════════════════════════════════════════════════════

    function _interp1_vec(R::AbstractArray, R_grid::AbstractVector, vals::AbstractVector)
        T = eltype(R)
        result = similar(R, T)
        result .= convert(T, NaN)
        n_bins = length(R_grid) - 1

        for i in 1:n_bins
            lo = convert(T, R_grid[i])
            hi = convert(T, R_grid[i+1])
            vlo = convert(T, vals[i])
            vhi = convert(T, vals[i+1])
            in_bin = @. (R >= lo) & (R < hi)
            t = @. (R - lo) / (hi - lo)
            result = @. ifelse(in_bin, vlo + t * (vhi - vlo), result)
        end

        vfirst = convert(T, vals[1])
        vlast  = convert(T, vals[end])
        result = @. ifelse(R < R_grid[1],   vfirst, result)
        result = @. ifelse(R >= R_grid[end], vlast,  result)

        return result
    end

    # ═══════════════════════════════════════════════════════════════
    #  Deflection angle from the m-th perturbation
    #
    #    ψ(R,φ) = ψ₀(R) + ψₘ(R) · cos(mφ − f(R))
    #
    #    α_R = dψ₀/dR + dψₘ/dR · cos(mφ−f)
    #                   + ψₘ · sin(mφ−f) · f'(R)
    #
    #    α_φ = (1/R) · ∂ψ/∂φ = −(m/R) · ψₘ · sin(mφ−f)
    #
    #    α_x = cos φ · α_R − sin φ · α_φ
    #    α_y = sin φ · α_R + cos φ · α_φ
    # ═══════════════════════════════════════════════════════════════

    function _multipole_deflection!(
        ax, ay, x, y,
        lens::SpiralMultipoleLens,
    )
        m = lens.m
        T_scalar = promote_type(eltype(x), Float32)
        i_p = convert(T_scalar, lens.pitch_angle)
        R0  = convert(T_scalar, lens.R0)

        R = @. sqrt(x^2 + y^2)
        epsR = eps(eltype(R))
        oneR = one(eltype(R))
        R_safe = @. ifelse(R < epsR, oneR, R)

        psi_m  = _interp1_vec(R, lens.R_grid, lens.psi_m_grid)
        dpsi_m = _interp1_vec(R, lens.R_grid, lens.dpsim_dR_grid)

        f_R    = @. m / tan(i_p) * log(R_safe / R0)
        fprime = @. m / (R_safe * tan(i_p))

        cos_arg = @. cos(m * atan(y, x) - f_R)
        sin_arg = @. sin(m * atan(y, x) - f_R)

        alpha_R   = @. dpsi_m * cos_arg + psi_m * sin_arg * fprime
        alpha_phi = @. -(m / R_safe) * psi_m * sin_arg

        cos_phi = @. x / R_safe
        sin_phi = @. y / R_safe

        T_mask = promote_type(eltype(ax), eltype(alpha_R))
        mask = R .> epsR
        dax = @. ifelse(mask, alpha_R * cos_phi - alpha_phi * sin_phi, zero(T_mask))
        day = @. ifelse(mask, alpha_R * sin_phi + alpha_phi * cos_phi, zero(T_mask))

        ax .+= dax
        ay .+= day
        return ax, ay
    end

    # ═══════════════════════════════════════════════════════════════
    #  LensBase dispatch
    # ═══════════════════════════════════════════════════════════════

    function lens_derivative(lens::SpiralMultipoleLens, x, y; kwargs...)
        ax, ay = lens_derivative(lens.m0_lens, x, y)
        _multipole_deflection!(ax, ay, x, y, lens)
        return ax, ay
    end

    function lens_hessian(lens::SpiralMultipoleLens, x, y; kwargs...)
        fxx0, fxy0, fyy0 = lens_hessian(lens.m0_lens, x, y)

        m = lens.m
        T_scalar = promote_type(eltype(x), Float32)
        i_p = convert(T_scalar, lens.pitch_angle)
        R0  = convert(T_scalar, lens.R0)

        d2psi_grid = _second_deriv(lens.R_grid, lens.dpsim_dR_grid)

        R = @. sqrt(x^2 + y^2)
        epsR = eps(eltype(R))
        oneR = one(eltype(R))
        R_safe = @. ifelse(R < epsR, oneR, R)
        mask = R .> epsR
        phi     = atan.(y, x)
        cos_phi = @. x / R_safe
        sin_phi = @. y / R_safe

        psi_m   = _interp1_vec(R, lens.R_grid, lens.psi_m_grid)
        dpsi_m  = _interp1_vec(R, lens.R_grid, lens.dpsim_dR_grid)
        d2psi_m = _interp1_vec(R, lens.R_grid, d2psi_grid)

        f_R     = @. m / tan(i_p) * log(R_safe / R0)
        fprime  = @. m / (R_safe * tan(i_p))
        fdoubleprime = @. -m / (R_safe^2 * tan(i_p))

        cos_arg = @. cos(m * phi - f_R)
        sin_arg = @. sin(m * phi - f_R)

        psi_RR = @. (d2psi_m * cos_arg
                      + 2 * dpsi_m * sin_arg * fprime
                      + psi_m * cos_arg * fprime^2
                      + psi_m * sin_arg * fdoubleprime)

        psi_Rphi = @. (-m * dpsi_m * sin_arg
                        + m * psi_m * cos_arg * fprime)

        psi_phiphi = @. -m^2 * psi_m * cos_arg

        alpha_R_term = @. dpsi_m * cos_arg + psi_m * sin_arg * fprime

        f_xx_pert = @. (psi_RR * cos_phi^2
                         - 2 * psi_Rphi * sin_phi * cos_phi / R_safe
                         + psi_phiphi * sin_phi^2 / R_safe^2
                         + alpha_R_term * sin_phi^2 / R_safe)

        f_yy_pert = @. (psi_RR * sin_phi^2
                         + 2 * psi_Rphi * sin_phi * cos_phi / R_safe
                         + psi_phiphi * cos_phi^2 / R_safe^2
                         + alpha_R_term * cos_phi^2 / R_safe)

        f_xy_pert = @. ((psi_RR - alpha_R_term / R_safe) * sin_phi * cos_phi
                         + psi_Rphi * (cos_phi^2 - sin_phi^2) / R_safe
                         - psi_phiphi * sin_phi * cos_phi / R_safe^2)

        fxx = @. ifelse(mask, fxx0 + f_xx_pert, fxx0)
        fxy = @. ifelse(mask, fxy0 + f_xy_pert, fxy0)
        fyy = @. ifelse(mask, fyy0 + f_yy_pert, fyy0)

        return fxx, fxy, fyy
    end

    function _second_deriv(R_grid::AbstractVector, dpsi_grid::AbstractVector)
        d2 = similar(dpsi_grid)
        d2[1] = zero(eltype(d2))
        for i in 2:(length(R_grid)-1)
            d2[i] = (dpsi_grid[i+1] - dpsi_grid[i-1]) / (R_grid[i+1] - R_grid[i-1])
        end
        d2[end] = d2[end-1]
        return d2
    end

end # module SpiralMultipole