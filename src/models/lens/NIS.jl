"""
    NIS — Non-singular Isothermal Sphere

Surface mass density:  κ(θ) = θ_E / (2√(s² + θ²))

Same as SIS but with a finite core radius `s` that removes the
central singularity.  Reduces to SIS as s → 0.

# Derivation
For κ(R) = θ_E/(2R) with R = √(s² + r²):

    α(r) = θ_E·(R − s) / r      (r > 0)
    α(0) = 0

    ψ(r) = θ_E·[R − s − s·arsinh(r/s)]

    dα/dr  = θ_E·s·(R−s) / (r²·R)
    α/r    = θ_E·(R−s) / r²

    f_xx = (dα/dr)·cos²φ + (α/r)·sin²φ
    f_yy = (dα/dr)·sin²φ + (α/r)·cos²φ
    f_xy = (dα/dr − α/r)·cosφ·sinφ

Limits:
- s → 0:  recovers SIS exactly
- r → 0:  dα/dr → θ_E/(2s),  α/r → θ_E/(2s),  κ = (f_xx+f_yy)/2 → θ_E/(2s)

# Parameters
- `theta_E`: Einstein radius [arcsec]
- `s`: core radius [arcsec]
- `xcentre`, `ycentre`: lens centre [arcsec]

# Example
    lens = SingleModel(NIS; theta_E=1.2, s=0.1)
"""
module NIS

    export LensCheck, LensPotential, LensDerivative, LensHessian

    # ═══════════════════════════════════════════════════════════
    #  1. Parameter validation
    # ═══════════════════════════════════════════════════════════

    function LensCheck(; theta_E::Real, s::Real, xcentre::Real=0., ycentre::Real=0.)
        @assert theta_E > 0  "theta_E must be positive, got $theta_E"
        @assert s > 0        "core radius s must be positive, got $s"
    end

    # ═══════════════════════════════════════════════════════════
    #  2. Radial helpers
    # ═══════════════════════════════════════════════════════════

    @inline _r2(x, y) = @. x^2 + y^2

    # ═══════════════════════════════════════════════════════════
    #  3. Lens potential  ψ(r)
    #     ψ(r) = θ_E·[R − s − s·arsinh(r/s)]
    # ═══════════════════════════════════════════════════════════

    function LensPotential(xg, yg; theta_E::Real, s::Real, xcentre::Real=0., ycentre::Real=0.)
        xsh = xg .- xcentre
        ysh = yg .- ycentre
        r2  = _r2(xsh, ysh)
        r   = @. sqrt(r2)
        R   = @. sqrt(s^2 + r2)
        # arsinh(r/s) = log(r/s + √(1+r²/s²)) = log((r+R)/s)
        arsinh_v = @. log((r + R) / s)
        return @. theta_E * (R - s - s * arsinh_v)
    end

    # ═══════════════════════════════════════════════════════════
    #  4. Lens derivative  α(r)
    #     α(r) = θ_E·(R − s) / r   (r > 0)
    #     α(0) = 0
    # ═══════════════════════════════════════════════════════════

    function LensDerivative(xg, yg; theta_E::Real, s::Real, xcentre::Real=0., ycentre::Real=0.)
        xsh = xg .- xcentre
        ysh = yg .- ycentre
        r2  = _r2(xsh, ysh)
        r   = @. sqrt(r2)
        R   = @. sqrt(s^2 + r2)

        # α_r(r) = θ_E·(R−s)/r  —  radial deflection magnitude
        alpha_r = similar(r)
        m = r .> 0
        alpha_r[m] .= theta_E .* (R[m] .- s) ./ r[m]
        alpha_r[.!m] .= 0.0

        # α_vector = α_r(r) · r̂ = α_r · (xsh/r, ysh/r)
        f_x = @. alpha_r * xsh / r
        f_y = @. alpha_r * ysh / r

        # r=0: both components → 0 (alpha_r is already 0)
        f_x[.!m] .= 0.0
        f_y[.!m] .= 0.0

        return f_x, f_y
    end

    # ═══════════════════════════════════════════════════════════
    #  5. Lens Hessian  —  radial → cartesian
    #
    #     For axisymmetric lens:  α = α_r(r) · r̂
    #       f_xx = (dα/dr)·cos²φ + (α/r)·sin²φ
    #       f_yy = (dα/dr)·sin²φ + (α/r)·cos²φ
    #       f_xy = (dα/dr − α/r)·cosφ·sinφ
    #
    #     NIS radial quantities:
    #       α/r    = θ_E·(R−s)/r²
    #       dα/dr  = θ_E·s·(R−s)/(r²·R)
    # ═══════════════════════════════════════════════════════════

    function LensHessian(xg, yg; theta_E::Real, s::Real, xcentre::Real=0., ycentre::Real=0.)
        xsh = xg .- xcentre
        ysh = yg .- ycentre
        r2  = _r2(xsh, ysh)
        r   = @. sqrt(r2)
        R   = @. sqrt(s^2 + r2)

        m    = r .> 0

        # ── α/r  and  dα/dr ──
        R_minus_s = @. R - s

        alpha_over_r = similar(r)
        alpha_over_r[m] .= theta_E .* R_minus_s[m] ./ r2[m]

        dalpha_dr = similar(r)
        dalpha_dr[m] .= theta_E .* s .* R_minus_s[m] ./ (r2[m] .* R[m])

        # ── r → 0 limits ──
        #     lim α/r     = θ_E/(2s)
        #     lim dα/dr   = θ_E/(2s)
        #     → f_xx = f_yy = θ_E/(2s),  f_xy = 0
        alpha_over_r[.!m] .= theta_E / (2 * s)
        dalpha_dr[.!m]    .= theta_E / (2 * s)

        # ── cosφ, sinφ ──
        cos_phi = similar(r)
        sin_phi = similar(r)
        cos_phi[m] .= xsh[m] ./ r[m]
        sin_phi[m] .= ysh[m] ./ r[m]
        cos_phi[.!m] .= 1.0   # placeholder; r=0 → y²·coeff=0 regardless
        sin_phi[.!m] .= 0.0

        cos2 = @. cos_phi^2
        sin2 = @. sin_phi^2
        sincos = @. sin_phi * cos_phi

        f_xx = @. dalpha_dr * cos2  +  alpha_over_r * sin2
        f_yy = @. dalpha_dr * sin2  +  alpha_over_r * cos2
        f_xy = @. (dalpha_dr - alpha_over_r) * sincos

        return f_xx, f_xy, f_yy
    end

end