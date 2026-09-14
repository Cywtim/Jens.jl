"""
    dPIE — Dual Pseudo-Isothermal Elliptical mass distribution

Cluster-scale lens model with both a core radius and a truncation
radius.  Physically motivated for cluster member galaxies: the core
softens the central singularity and the truncation reflects tidal
stripping by the cluster potential.

The 3D density profile is:

    ρ(r) ∝ 1 / [(1 + r²/r_core²) × (1 + r²/r_trunc²)]

which yields the 2D convergence (in the SIE limit r_core → 0, r_trunc → ∞):

    κ(R) = (θ_E / 2) × [1/√(R² + r_core²) − 1/√(R² + r_trunc²)]

Implemented as the difference of two NIE profiles:

    α_dPIE = α_NIE(r_core) − α_NIE(r_trunc)

This guarantees correct limiting behaviour:
    r_core → 0, r_trunc → ∞  →  SIE
    r_core → 0, r_trunc → ∞, q → 1  →  SIS

# Parameters
- `theta_E`: normalization [arcsec] — Einstein radius of the equivalent SIS
- `r_core`: core radius [arcsec] (≥ 0)
- `r_trunc`: truncation radius [arcsec] (> r_core)
- `e1`, `e2`: ellipticity components (∈ [−0.5, 0.5])
- `xcentre`, `ycentre`: lens centre [arcsec]

# Reference
- Jullo et al. (2007), arXiv:0706.0048 — Bayesian cluster lens modelling
- Elíasdóttir et al. (2007), arXiv:0710.5636 — Abell 2218

# Example
    lens = SingleModel(dPIE; theta_E=0.8, r_core=0.01, r_trunc=5.0,
                       e1=0.1, e2=0.05)
"""
module dPIE

    using Jens.LensUtils

    # ═══════════════════════════════════════════════════════════════
    #  LensCheck — parameter validation
    # ═══════════════════════════════════════════════════════════════

    function LensCheck(;
            theta_E::Real,
            r_core::Real,
            r_trunc::Real,
            e1::Real,
            e2::Real,
            xcentre::Real = 0.0,
            ycentre::Real = 0.0,
        )

        theta_E > 0 || error("dPIE: theta_E must be > 0, got $theta_E")
        r_core >= 0 || error("dPIE: r_core must be >= 0, got $r_core")
        r_trunc > r_core || error("dPIE: r_trunc ($r_trunc) must be > r_core ($r_core)")

        for (name, val) in [("e1", e1), ("e2", e2)]
            -0.5 < val < 0.5 || error("dPIE: $name = $val out of range (−0.5, 0.5)")
        end

        return nothing
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: NIE deflection/potential/hessian (inline)
    #
    #  We don't import NIE/NIEkappa to avoid circular dependency
    #  issues inside LensModel.  Instead we inline the core formulas
    #  from NIEkappa.jl (Keeton & Kochanek 1998).
    # ═══════════════════════════════════════════════════════════════

    # ── Transform e1/e2 → major-axis (b, s, q, varphi) ──

    function _to_major(theta_E::Real, s_scale::Real, e1::Real, e2::Real)
        q, varphi = LensUtils.e2phiq(e1, e2)
        # circularised → major-axis Einstein radius
        # b = theta_E * sqrt((1+q^2)/2)  — same convention as SIE._to_major_axes
        # (Keeton & Kochanek 1998).  Previous code had b = theta_E * sqrt(q),
        # which underestimated mass by ~1-q for q < 1.
        b = theta_E * sqrt((1.0 + q^2) / 2.0)
        # core/trunc radius in major-axis frame
        s = s_scale * sqrt((1.0 + q^2) / (2.0 * q^2))
        q = min(q, 0.9999)
        return (; b, s, q, varphi)
    end

    # ── NIE deflection in major-axis coordinates ──

    function _nie_deflection_major(x::AbstractArray, y::AbstractArray,
                                    b::Real, s::Real, q::Real)
        psi = sqrt.(q^2 .* (s^2 .+ x.^2) .+ y.^2)
        pref = b / sqrt(1.0 - q^2)
        fx = pref .* atan.(sqrt(1.0 - q^2) .* x ./ (psi .+ s))
        fy = pref .* atanh.(sqrt(1.0 - q^2) .* y ./ (psi .+ q^2 .* s))
        return fx, fy
    end

    # ── NIE potential in major-axis coordinates ──

    function _nie_potential_major(x::AbstractArray, y::AbstractArray,
                                   b::Real, s::Real, q::Real)
        psi = sqrt.(q^2 .* (s^2 .+ x.^2) .+ y.^2)
        fx, fy = _nie_deflection_major(x, y, b, s, q)
        f = x .* fx .+ y .* fy .- b .* s ./ 2.0 .* log.((psi .+ s).^2 .+ (1.0 .- q^2) .* x.^2)
        return f
    end

    # ── NIE hessian in major-axis coordinates (finite difference) ──

    function _nie_hessian_major(x::AbstractArray, y::AbstractArray,
                                 b::Real, s::Real, q::Real;
                                 diff::Real = 1e-10)
        fx, fy = _nie_deflection_major(x, y, b, s, q)
        fx_dx, _ = _nie_deflection_major(x .+ diff, y, b, s, q)
        fx_dy, fy_dy = _nie_deflection_major(x, y .+ diff, b, s, q)

        f_xx = (fx_dx .- fx) ./ diff
        f_xy = (fx_dy .- fx) ./ diff
        f_yy = (fy_dy .- fy) ./ diff
        return f_xx, f_xy, f_yy
    end

    # ═══════════════════════════════════════════════════════════════
    #  dPIE = NIE(core) − NIE(trunc)
    #
    #  All quantities (deflection, potential, hessian) are linear
    #  in the surface density κ, so subtraction is exact.
    # ═══════════════════════════════════════════════════════════════

    # ── Helper: compute major-axis dPIE as diff of two NIEs ──

    function _dpie_major(fn::Function, x::AbstractArray, y::AbstractArray,
                          theta_E::Real, r_core::Real, r_trunc::Real,
                          e1::Real, e2::Real; kwargs...)

        m_core  = _to_major(theta_E, r_core, e1, e2)
        m_trunc = _to_major(theta_E, r_trunc, e1, e2)

        # Both must use the same q and varphi (from the same e1,e2)
        result_core  = fn(x, y, m_core.b, m_core.s, m_core.q; kwargs...)
        result_trunc = fn(x, y, m_trunc.b, m_trunc.s, m_trunc.q; kwargs...)

        diff = _subtract_results(result_core, result_trunc)
        return diff, m_core.varphi
    end

    # ── Subtract: works on scalars, tuples, and plain arrays ──

    _subtract_results(a::AbstractArray, b::AbstractArray) = a .- b

    function _subtract_results(a::Tuple, b::Tuple)
        return ntuple(i -> _subtract_results(a[i], b[i]), length(a))
    end

    # ═══════════════════════════════════════════════════════════════
    #  Public API
    # ═══════════════════════════════════════════════════════════════

    function LensPotential(xg::AbstractArray, yg::AbstractArray;
            theta_E::Real,
            r_core::Real,
            r_trunc::Real,
            e1::Real,
            e2::Real,
            xcentre::Real = 0.0,
            ycentre::Real = 0.0,
        )

        # Shift and rotate to major axis
        xsh = xg .- xcentre
        ysh = yg .- ycentre
        _, _, _, varphi = _to_major(theta_E, r_core, e1, e2)
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        f_diff, _ = _dpie_major(_nie_potential_major, xsh, ysh,
                                 theta_E, r_core, r_trunc, e1, e2)
        return f_diff   # potential is scalar → no rotation needed
    end

    function LensDerivative(xg::AbstractArray, yg::AbstractArray;
            theta_E::Real,
            r_core::Real,
            r_trunc::Real,
            e1::Real,
            e2::Real,
            xcentre::Real = 0.0,
            ycentre::Real = 0.0,
        )

        # Shift and rotate to major axis
        xsh = xg .- xcentre
        ysh = yg .- ycentre
        _, _, _, varphi = _to_major(theta_E, r_core, e1, e2)
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        f_diff, _ = _dpie_major(_nie_deflection_major, xsh, ysh,
                                 theta_E, r_core, r_trunc, e1, e2)

        # Rotate back to image coordinates
        fx, fy = LensUtils.LensRotation(f_diff[1], f_diff[2], varphi)
        return fx, fy
    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
            theta_E::Real,
            r_core::Real,
            r_trunc::Real,
            e1::Real,
            e2::Real,
            xcentre::Real = 0.0,
            ycentre::Real = 0.0,
            diff::Real = 1e-10,
        )

        # Shift and rotate to major axis
        xsh = xg .- xcentre
        ysh = yg .- ycentre
        _, _, _, varphi = _to_major(theta_E, r_core, e1, e2)
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        (f_xx, f_xy, f_yy), _ = _dpie_major(_nie_hessian_major, xsh, ysh,
                                              theta_E, r_core, r_trunc, e1, e2;
                                              diff=diff)

        # Rotate hessian back: κ invariant, shear rotates by 2φ
        kappa   = @. 0.5 * (f_xx + f_yy)
        gamma1_ = @. 0.5 * (f_xx - f_yy)
        gamma2_ = f_xy
        gamma1  = @. cos(2.0 * varphi) * gamma1_ + sin(2.0 * varphi) * gamma2_
        gamma2  = @. -sin(2.0 * varphi) * gamma1_ + cos(2.0 * varphi) * gamma2_

        f_xx_rot = @. kappa + gamma1
        f_yy_rot = @. kappa - gamma1
        f_xy_rot = gamma2

        return f_xx_rot, f_xy_rot, f_yy_rot
    end

end # module dPIE