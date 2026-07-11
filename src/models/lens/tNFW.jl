# ═══════════════════════════════════════════════════════════════
#  tNFW — truncated NFW lens model with smooth tidal cutoff
#
#  3D density:
#    ρ(r) = ρ_NFW(r) / (1 + (r/r_t)⁴)
#
#  where  ρ_NFW(r) = ρ₀ / [(r/r_s) (1 + r/r_s)²]
#
#  Smooth truncation: ρ ~ r⁻¹ inside r_s, ρ ~ r⁻³ between, ρ ~ r⁻⁷ outside r_t.
#  Finite total mass, physically motivated tidal cutoff.
#
#  Parameters:
#    Rs       — NFW scale radius          [arcsec]
#    alpha_Rs — deflection scale at Rs    [arcsec]
#    r_t      — truncation / tidal radius [arcsec]
#    xcentre, ycentre — centre            [arcsec]
#
#  Radial quantities (deflection & convergence) are precomputed
#  on a log-spaced grid at construction.  Rendering uses GPU-safe
#  linear interpolation (broadcast, zero scalar indexing).
#
#  Refs:
#    Baltz+2009 (arXiv:0705.3336) — smooth NFW truncation
#    Minor+2020 (2011.10629)       — subhalo concentration effects
# ═══════════════════════════════════════════════════════════════

module tNFW

    import Jens.LensBase: AbstractLens, lens_derivative, lens_hessian, lens_potential, lens_check

    export tNFWLens

    # ═══════════════════════════════════════════════════════════
    #  Struct
    # ═══════════════════════════════════════════════════════════

    """
        tNFWLens(; Rs, alpha_Rs, r_t, xcentre=0., ycentre=0., n_radial=300)

    Truncated NFW lens model.  Precomputes radial deflection and
    convergence on a log-spaced grid at construction time.

    # Parameters
    - `Rs`: NFW scale radius [arcsec]
    - `alpha_Rs`: deflection scale at Rs (same convention as NFW.jl) [arcsec]
    - `r_t`: truncation radius [arcsec] — ρ → 0 for r ≫ r_t
    - `xcentre`, `ycentre`: centre position [arcsec]
    - `n_radial`: number of precomputed radial samples (default 300)
    """
    struct tNFWLens{V<:AbstractVector{Float64}} <: AbstractLens
        Rs::Float64
        alpha_Rs::Float64
        r_t::Float64
        xcentre::Float64
        ycentre::Float64
        # Precomputed radial tables  [log-spaced, from ~1e-4·Rs to 5·r_t]
        R_grid::V
        alpha_grid::V
        kappa_grid::V
    end

    function tNFWLens(; Rs::Real, alpha_Rs::Real, r_t::Real,
                       xcentre::Real=0.0, ycentre::Real=0.0,
                       n_radial::Int=300)
        R_grid, alpha_grid, kappa_grid =
            _precompute(Float64(Rs), Float64(alpha_Rs), Float64(r_t), n_radial)
        return tNFWLens(Float64(Rs), Float64(alpha_Rs), Float64(r_t),
                        Float64(xcentre), Float64(ycentre),
                        R_grid, alpha_grid, kappa_grid)
    end

    function lens_check(lens::tNFWLens; kwargs...)
        @assert lens.Rs > 0
        @assert lens.alpha_Rs > 0
        @assert lens.r_t > 0
    end

    # ═══════════════════════════════════════════════════════════
    #  Precomputation
    # ═══════════════════════════════════════════════════════════

    function _gamma_NFW(r, Rs, rho0)
        # ρ_NFW(r) = rho0 / [(r/Rs)(1 + r/Rs)²]
        x = r / Rs
        return rho0 / (x * (1 + x)^2)
    end

    function _gamma_tNFW(r, Rs, rho0, r_t)
        # ρ(r) = ρ_NFW(r) / (1 + (r/r_t)⁴)
        x4 = (r / r_t)^4
        return _gamma_NFW(r, Rs, rho0) / (1 + x4)
    end

    function _precompute(Rs::Float64, alpha_Rs::Float64, r_t::Float64, n::Int)
        # ── NFW density normalisation (same as NFW.jl) ──
        rho0 = alpha_Rs / (4 * Rs^2 * (1 + log(0.5)))

        # ── Radial grid: log-spaced ──
        R_min = max(1e-4 * Rs, 1e-6)
        R_max = 5.0 * max(r_t, Rs)
        R_grid = exp10.(range(log10(R_min), log10(R_max); length=n))

        # ── Numerical Σ_tNFW(R) = 2 ∫₀^∞ ρ_tNFW(r) dz ──
        n_z  = 2000
        Sigma = Vector{Float64}(undef, n)
        for i in 1:n
            R = R_grid[i]
            z_max = 10.0 * r_t
            dz = z_max / (n_z - 1)
            s = 0.0
            for j in 1:n_z
                z = (j - 1) * dz
                r = sqrt(R^2 + z^2)
                rho_val = _gamma_tNFW(r, Rs, rho0, r_t)
                w = ifelse(j == 1 || j == n_z, 0.5, 1.0)
                s += w * rho_val
            end
            Sigma[i] = 2.0 * s * dz
        end

        # ── Calibration: Σ_NFW(Rs) / κ_NFW(Rs) = Σ_crit_eff ──
        #     κ_NFW(Rs) = 2·rho0·Rs·f(1)  where f is the NFW convergence kernel
        #     We compute Σ_NFW(Rs) numerically for r_t→∞
        #     Then κ_tNFW(R) = Σ_tNFW(R) / Σ_NFW(Rs) * κ_NFW(Rs)
        #
        #     Use large r_t (100·Rs) as effective "untruncated"
        r_t_eff = 100.0 * Rs
        Sigma_nfw_at_Rs = _compute_Sigma_at(Rs, Rs, rho0, r_t_eff)
        kappa_nfw_at_Rs = _nfw_kappa_at_Rs(Rs, rho0)
        Sigma_crit_eff = Sigma_nfw_at_Rs / kappa_nfw_at_Rs

        kappa_grid = Sigma ./ Sigma_crit_eff

        # ── α(R) = (2/R) ∫₀ᴿ κ(R') R' dR' ──
        alpha_grid = Vector{Float64}(undef, n)
        cum = 0.0
        alpha_grid[1] = 0.0
        for i in 2:n
            dR = R_grid[i] - R_grid[i-1]
            cum += 0.5 * (kappa_grid[i] * R_grid[i] + kappa_grid[i-1] * R_grid[i-1]) * dR
            alpha_grid[i] = 2.0 * cum / R_grid[i]
        end

        return R_grid, alpha_grid, kappa_grid
    end

    function _compute_Sigma_at(R::Float64, Rs::Float64, rho0::Float64, r_t_eff::Float64)
        n_z = 2000
        z_max = 10.0 * r_t_eff
        dz = z_max / (n_z - 1)
        s = 0.0
        for j in 1:n_z
            z = (j - 1) * dz
            r = sqrt(R^2 + z^2)
            rho_val = _gamma_tNFW(r, Rs, rho0, r_t_eff)
            w = ifelse(j == 1 || j == n_z, 0.5, 1.0)
            s += w * rho_val
        end
        return 2.0 * s * dz
    end

    function _nfw_kappa_at_Rs(Rs::Float64, rho0::Float64)
        # κ_NFW(Rs) = 2·rho0·Rs·f(1) where f(x) is the NFW convergence kernel
        # f(1) from analytic formula (B03, W97)
        x = 1.0  # R/Rs = 1
        if abs(x - 1) < 1e-6
            return 2 * rho0 * Rs * (1.0 / 3.0)
        elseif x < 1
            sq = sqrt(1 - x^2)
            f = 1/(x^2-1) * (1 - 2/sq * atanh(sqrt((1-x)/(1+x))))
        else
            sq = sqrt(x^2 - 1)
            f = 1/(x^2-1) * (1 - 2/sq * atan(sqrt((x-1)/(1+x))))
        end
        return 2 * rho0 * Rs * f
    end

    # ═══════════════════════════════════════════════════════════
    #  GPU-safe linear interpolation
    # ═══════════════════════════════════════════════════════════

    function _interp1_vec(R::AbstractArray, R_grid::AbstractVector, vals::AbstractVector)
        T = eltype(R)
        result = fill(convert(T, NaN), size(R))
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
        result = @. ifelse(R < R_grid[1],   ifelse(R .>= 0, vfirst, zero(T)), result)
        result = @. ifelse(R >= R_grid[end], zero(T), result)

        return result
    end

    # ═══════════════════════════════════════════════════════════
    #  LensBase interface
    # ═══════════════════════════════════════════════════════════

    function lens_derivative(lens::tNFWLens, x, y; kwargs...)
        xsh = x .- lens.xcentre
        ysh = y .- lens.ycentre
        T = promote_type(eltype(x), Float32)
        R = @. max(sqrt(xsh^2 + ysh^2), eps(T))

        a = _interp1_vec(R, lens.R_grid, lens.alpha_grid)
        a_over_R = @. a / R

        f_x = @. a_over_R * xsh
        f_y = @. a_over_R * ysh
        return f_x, f_y
    end

    function lens_hessian(lens::tNFWLens, x, y; kwargs...)
        xsh = x .- lens.xcentre
        ysh = y .- lens.ycentre
        T = promote_type(eltype(x), Float32)
        R = @. max(sqrt(xsh^2 + ysh^2), eps(T))

        # κ from interpolation
        kappa = _interp1_vec(R, lens.R_grid, lens.kappa_grid)

        # α/R from interpolation
        a = _interp1_vec(R, lens.R_grid, lens.alpha_grid)
        alpha_over_R = @. a / R

        cos_phi = @. xsh / R
        sin_phi = @. ysh / R
        cos2 = @. cos_phi^2
        sin2 = @. sin_phi^2
        sincos = @. sin_phi * cos_phi

        # Cartesian Hessian from κ, γ
        # γ = κ̄ − κ,  κ̄ ≈ α/R  (exact: κ̄ = α/(2R)... no, κ̄ = M/(πR²Σcrit))
        # For axisymmetric lens: f_xx = κ + γ cos2φ, etc.
        # Using: γ = alpha_over_R - kappa  (since κ̄(<R) = α/R for axisymmetric)
        gamma = @. alpha_over_R - kappa

        f_xx = @. kappa + gamma * (cos2 - sin2 + cos2)  # simplify: κ + γ cos(2φ)
        # Actually: f_xx = κ + γ·cos(2φ),  f_yy = κ − γ·cos(2φ),  f_xy = γ·sin(2φ)
        cos2phi = @. cos2 - sin2
        sin2phi = @. 2 * sincos

        f_xx = @. kappa + gamma * cos2phi
        f_yy = @. kappa - gamma * cos2phi
        f_xy = @. gamma * sin2phi

        return f_xx, f_xy, f_yy
    end

    function lens_potential(lens::tNFWLens, x, y; kwargs...)
        xsh = x .- lens.xcentre
        ysh = y .- lens.ycentre
        T = promote_type(eltype(x), Float32)
        R = @. max(sqrt(xsh^2 + ysh^2), eps(T))

        # ψ(R) = ∫ α(R') dR' — cumulative trapezoidal
        # We precompute this from alpha_grid
        # Simple: use the precomputed alpha to integrate
        psi_vals = similar(lens.alpha_grid)
        psi_vals[1] = 0.0
        for i in 2:length(lens.R_grid)
            dR = lens.R_grid[i] - lens.R_grid[i-1]
            psi_vals[i] = psi_vals[i-1] +
                0.5 * (lens.alpha_grid[i] + lens.alpha_grid[i-1]) * dR
        end
        return _interp1_vec(R, lens.R_grid, psi_vals)
    end

end # module tNFW