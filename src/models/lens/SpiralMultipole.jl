module SpiralMultipole

    # ═══════════════════════════════════════════════════════════════
    #  SpiralMultipoleLens — density-wave multipole expansion lens
    #
    #  κ(R,φ) = κ₀(R) + Σₘ Aₘ(R) cos(mφ - f(R))
    #
    #  The m=0 term is provided by MGECombinedLens (circular Gaussians).
    #  The m>0 terms are density-wave perturbations, localized in a
    #  Gaussian annulus around the corotation radius R_cr.
    #
    #  The radial part ψₘ(R) is computed via the 2D Poisson Green
    #  function and precomputed on a radial grid for interpolation
    #  during ray tracing.
    #
    #  Refs:
    #    Lin & Shu (1964) — density wave theory
    #    Shajib (2019) MNRAS 488, 1387 — MGE for lensing
    # ═══════════════════════════════════════════════════════════════

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
    - `amplitude::Float64`: peak amplitude A₀ of the density-wave perturbation
    - `R_cr::Float64`: corotation radius where the perturbation is strongest
    - `delta_R::Float64`: Gaussian width of the radial annulus
    - `pitch_angle::Float64`: pitch angle i_p in radians
    - `R0::Float64`: reference radius for logarithmic spiral phase

    # Precomputed (via constructor)
    - `R_grid`, `psi_m_grid`, `dpsim_dR_grid`: radial interpolation tables
      for the m-th Green function and its derivative.
    """
    struct SpiralMultipoleLens <: AbstractLens
        m0_lens::MGECombinedLens
        m::Int
        amplitude::Float64
        R_cr::Float64
        delta_R::Float64
        pitch_angle::Float64
        R0::Float64
        # Precomputed radial tables
        R_grid::Vector{Float64}
        psi_m_grid::Vector{Float64}
        dpsim_dR_grid::Vector{Float64}

        function SpiralMultipoleLens(
            m0_lens::MGECombinedLens,
            m::Int, amplitude::Float64,
            R_cr::Float64, delta_R::Float64,
            pitch_angle::Float64, R0::Float64;
            n_radial::Int=200,
        )
            @assert m >= 1 "m must be ≥ 1 (m=0 is handled by m0_lens)"
            @assert amplitude >= 0 "amplitude must be non-negative"
            @assert R_cr > 0 "R_cr must be positive"
            @assert delta_R > 0 "delta_R must be positive"

            # -- build radial grid for Green function precomputation --
            R_grid = _build_radial_grid(R_cr, delta_R; n=n_radial)
            psi_m_grid, dpsim_dR_grid = _precompute_green(m, amplitude, R_cr, delta_R, R_grid)

            new(m0_lens, m, amplitude, R_cr, delta_R, pitch_angle, R0,
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
        # Cover [0.01·R_cr, R_cr + 5·delta_R] with dense sampling near R_cr
        R_min = max(1e-4 * R_cr, 0.01 * R_cr)
        R_max = R_cr + 6.0 * delta_R
        return exp10.(range(log10(R_min), log10(R_max); length=n))
    end

    function _precompute_green(m::Int, A0::Float64, R_cr::Float64, delta_R::Float64,
                               R_grid::Vector{Float64})
        n_r = length(R_grid)
        psi = zeros(n_r)
        dpsi = zeros(n_r)

        # Use quadrature on a finer grid for the two integrals
        n_fine = 2000
        s_fine = exp10.(range(log10(R_grid[1]), log10(R_grid[end]); length=n_fine))
        kappa_fine = [_kappa_density_wave(s, A0, R_cr, delta_R) for s in s_fine]

        # Precompute cumulative integrals via trapezoidal rule
        # I₁(R) = ∫₀ᴿ sᵐ⁺¹ κ(s) ds
        # I₂(R) = ∫_R^∞ s¹⁻ᵐ κ(s) ds
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
            # Find nearest index in fine grid
            idx = searchsortedfirst(s_fine, R)
            idx = clamp(idx, 1, n_fine - 1)
            I1 = I1_cum[idx]
            I2 = I2_cum[idx]

            psi[j] = (1.0 / m) * (R^(-m) * I1 + R^m * I2)
            dpsi[j] = -R^(-m-1) * I1 + R^(m-1) * I2
        end

        # Handle R → 0 limit
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

    @inline _spiral_phase(R::Float64, m::Int, pitch_angle::Float64, R0::Float64) =
        m / tan(pitch_angle) * log(R / R0)

    @inline _spiral_phase_deriv(R::Float64, m::Int, pitch_angle::Float64) =
        m / (R * tan(pitch_angle))

    # ═══════════════════════════════════════════════════════════════
    #  Inline linear interpolation (avoid Interpolations.jl dep)
    # ═══════════════════════════════════════════════════════════════

    @inline function _interp1(R::Float64, R_grid::Vector{Float64}, vals::Vector{Float64})
        if R <= R_grid[1];  return vals[1];   end
        if R >= R_grid[end]; return vals[end]; end
        i = searchsortedfirst(R_grid, R)
        t = (R - R_grid[i-1]) / (R_grid[i] - R_grid[i-1])
        return vals[i-1] + t * (vals[i] - vals[i-1])
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
        A0 = lens.amplitude
        i_p = lens.pitch_angle
        R0 = lens.R0

        for idx in eachindex(x)
            R = sqrt(x[idx]^2 + y[idx]^2)
            if R < 1e-20
                continue
            end
            phi = atan(y[idx], x[idx])

            psi_m   = _interp1(R, lens.R_grid, lens.psi_m_grid)
            dpsi_m  = _interp1(R, lens.R_grid, lens.dpsim_dR_grid)
            f_R     = _spiral_phase(R, m, i_p, R0)
            fprime  = _spiral_phase_deriv(R, m, i_p)

            cos_arg = cos(m * phi - f_R)
            sin_arg = sin(m * phi - f_R)

            alpha_R = dpsi_m * cos_arg + psi_m * sin_arg * fprime
            alpha_phi = -(m / R) * psi_m * sin_arg

            cos_phi = x[idx] / R
            sin_phi = y[idx] / R

            ax[idx] += alpha_R * cos_phi - alpha_phi * sin_phi
            ay[idx] += alpha_R * sin_phi + alpha_phi * cos_phi
        end
        return ax, ay
    end

    # ═══════════════════════════════════════════════════════════════
    #  LensBase dispatch
    # ═══════════════════════════════════════════════════════════════

    function lens_derivative(lens::SpiralMultipoleLens, x, y; kwargs...)
        # 1. m=0 component from MGE
        ax, ay = lens_derivative(lens.m0_lens, x, y)

        # 2. Add multipole perturbation
        _multipole_deflection!(ax, ay, x, y, lens)

        return ax, ay
    end

    function lens_hessian(lens::SpiralMultipoleLens, x, y; kwargs...)
        # 1. m=0 component from MGE
        fxx0, fxy0, fyy0 = lens_hessian(lens.m0_lens, x, y)

        # 2. Analytic multipole Hessian
        m = lens.m
        A0 = lens.amplitude
        i_p = lens.pitch_angle
        R0 = lens.R0

        # Precompute d²ψₘ/dR² grid (once, not in the hot loop)
        d2psi_grid = _second_deriv(lens.R_grid, lens.dpsim_dR_grid)

        n = length(x)
        fxx = zeros(n)
        fxy = zeros(n)
        fyy = zeros(n)

        for idx in eachindex(x)
            R = sqrt(x[idx]^2 + y[idx]^2)
            if R < 1e-20
                continue
            end
            phi = atan(y[idx], x[idx])
            cos_phi = x[idx] / R
            sin_phi = y[idx] / R

            psi_m   = _interp1(R, lens.R_grid, lens.psi_m_grid)
            dpsi_m  = _interp1(R, lens.R_grid, lens.dpsim_dR_grid)
            d2psi_m = _interp1(R, lens.R_grid, d2psi_grid)
            f_R     = _spiral_phase(R, m, i_p, R0)
            fprime  = _spiral_phase_deriv(R, m, i_p)
            # f''(R) for logarithmic spiral:
            fdoubleprime = -m / (R^2 * tan(i_p))

            cos_arg = cos(m * phi - f_R)
            sin_arg = sin(m * phi - f_R)

            # ∂²ψ/∂R², ∂²ψ/∂R∂φ, ∂²ψ/∂φ² for the m-th mode
            psi_RR = (d2psi_m * cos_arg
                      + 2.0 * dpsi_m * sin_arg * fprime
                      + psi_m * cos_arg * fprime^2
                      + psi_m * sin_arg * fdoubleprime)

            psi_Rphi = (-m * dpsi_m * sin_arg
                        + m * psi_m * cos_arg * fprime)

            psi_phiphi = -m^2 * psi_m * cos_arg

            # Convert to Cartesian Hessian
            f_xx_pert = (psi_RR * cos_phi^2
                         - 2.0 * psi_Rphi * sin_phi * cos_phi / R
                         + psi_phiphi * sin_phi^2 / R^2
                         + (dpsi_m * cos_arg + psi_m * sin_arg * fprime) * sin_phi^2 / R)

            f_yy_pert = (psi_RR * sin_phi^2
                         + 2.0 * psi_Rphi * sin_phi * cos_phi / R
                         + psi_phiphi * cos_phi^2 / R^2
                         + (dpsi_m * cos_arg + psi_m * sin_arg * fprime) * cos_phi^2 / R)

            f_xy_pert = ((psi_RR - (dpsi_m * cos_arg + psi_m * sin_arg * fprime) / R) * sin_phi * cos_phi
                         + psi_Rphi * (cos_phi^2 - sin_phi^2) / R
                         - psi_phiphi * sin_phi * cos_phi / R^2)

            fxx[idx] = fxx0[idx] + f_xx_pert
            fxy[idx] = fxy0[idx] + f_xy_pert
            fyy[idx] = fyy0[idx] + f_yy_pert
        end

        return fxx, fxy, fyy
    end

    # Helper: second derivative via finite differences on the precomputed grid
    function _second_deriv(R_grid::Vector{Float64}, dpsi_grid::Vector{Float64})
        d2 = similar(dpsi_grid)
        d2[1] = 0.0
        for i in 2:(length(R_grid)-1)
            d2[i] = (dpsi_grid[i+1] - dpsi_grid[i-1]) / (R_grid[i+1] - R_grid[i-1])
        end
        d2[end] = d2[end-1]
        return d2
    end

end # module SpiralMultipole