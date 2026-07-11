# ═══════════════════════════════════════════════════════════════
#  MultipoleEPL — m=3,4 angular multipole perturbation
#
#  Adds azimuthal deviation to an EPL lens following the
#  Oh+2024 (2404.17124) circular multipole convention.
#
#  Designed to be paired with EPL via CombinedLens:
#
#    lens = CombinedLens(
#        EPL => (theta_E=1.2, gamma=2.0, e1=0.1, e2=0.05, ...),
#        MultipoleEPL => (m=3, amp=0.01, phi=0.2, theta_E=1.2, ...),
#        MultipoleEPL => (m=4, amp=0.02, phi=0.0, theta_E=1.2, ...),
#    )
#
#  Convention (Oh+2024, circular multipole):
#    δr = a_m cos(m(φ − φ_m))   at the κ=1/2 isodensity contour
#    amp = a_m / a              (fraction of semi-major axis)
#
#  The perturbation is localised near the Einstein radius
#  (where lensing is most sensitive) with a Gaussian envelope.
#
#  Refs:
#    Oh+2024 (2404.17124) — multipole prior from galaxy isophotes
#    Van de Vyvere+2022 (A&A 659, A127) — boxy/disky detectability
# ═══════════════════════════════════════════════════════════════

module MultipoleEPL

    using SpecialFunctions
    import Jens.LensBase: lens_derivative, lens_hessian, lens_potential, lens_check

    export LensCheck, LensPotential, LensDerivative, LensHessian

    # ═══════════════════════════════════════════════════════════
    #  Parameters:
    #    m       — angular order (3 = triangle, 4 = boxy/disky)
    #    amp     — a_m/a  (relative amplitude, typically ±0.005–0.05)
    #    phi_m   — multipole orientation relative to EPL major axis [rad]
    #    theta_E — Einstein radius of the associated EPL [arcsec]
    #    xcentre, ycentre — centre (should match EPL)
    #
    #  NOTE: m=4 with amp>0 (phi_m≈phi_EPL) → disky
    #        m=4 with amp<0 (phi_m≈phi_EPL) → boxy
    # ═══════════════════════════════════════════════════════════

    function LensCheck(; m::Int, amp::Real, phi_m::Real,
                        theta_E::Real,
                        xcentre::Real=0., ycentre::Real=0.)
        @assert m == 3 || m == 4  "m must be 3 or 4, got $m"
        @assert theta_E > 0
    end

    # ═══════════════════════════════════════════════════════════
    #  Multipole potential perturbation
    #
    #    ψ_m(R,φ) = amp · θ_E · G(R) · cos(m(φ − φ_m))
    #
    #  where G(R) = exp(−(R−θ_E)² / (2σ²))  (Gaussian at θ_E)
    #  and   σ   = θ_E / 3  (width ~ Einstein radius)
    #
    #  Derivatives:
    #    ∂ψ/∂R = amp·θ_E·[G'(R)·cos + G(R)·0]
    #    ∂ψ/∂φ = −amp·θ_E·m·G(R)·sin(m(φ−φ_m))
    #
    #    α_R = ∂ψ/∂R,  α_φ = (1/R)·∂ψ/∂φ
    #    α_x = α_R·cos(φ) − α_φ·sin(φ)
    #    α_y = α_R·sin(φ) + α_φ·cos(φ)
    # ═══════════════════════════════════════════════════════════

    @inline function _gaussian(R, theta_E, sigma)
        return @. exp(-(R - theta_E)^2 / (2 * sigma^2))
    end

    @inline function _gaussian_deriv(R, theta_E, sigma)
        G = _gaussian(R, theta_E, sigma)
        return @. -G * (R - theta_E) / sigma^2
    end

    # ── Multipole deflection — in-place addition ──
    function _multipole_deflection!(ax, ay, x, y, m, amp, phi_m,
                                     theta_E, xc, yc)
        T = eltype(x)
        sigma = theta_E / T(3)

        dx = x .- xc
        dy = y .- yc
        R  = @. max(sqrt(dx^2 + dy^2), eps(T))

        G   = _gaussian(R, theta_E, sigma)
        Gp  = _gaussian_deriv(R, theta_E, sigma)

        phi = @. atan(dy, dx)
        arg = @. m * (phi - phi_m)
        cos_m = @. cos(arg)
        sin_m = @. sin(arg)

        # α_R = amp · θ_E · G'(R) · cos(m(φ−φ_m))
        # α_φ = −amp · θ_E · m · G(R) · sin(m(φ−φ_m)) / R
        a_R  = @. amp * theta_E * Gp * cos_m
        a_phi = @. -amp * theta_E * T(m) * G * sin_m / R

        cos_phi = @. dx / R
        sin_phi = @. dy / R

        ax .+= @. a_R * cos_phi - a_phi * sin_phi
        ay .+= @. a_R * sin_phi + a_phi * cos_phi
        return nothing
    end

    # ═══════════════════════════════════════════════════════════
    #  Module interface (compatible with CombinedLens)
    # ═══════════════════════════════════════════════════════════

    function LensPotential(x, y; m::Real, amp::Real, phi_m::Real,
                           theta_E::Real,
                           xcentre::Real=0., ycentre::Real=0.)
        m_int = Int(m)
        T = eltype(x)
        sigma = theta_E / T(3)
        dx = x .- xcentre
        dy = y .- ycentre
        R  = @. max(sqrt(dx^2 + dy^2), eps(T))
        phi = @. atan(dy, dx)
        G   = _gaussian(R, theta_E, sigma)
        return @. amp * theta_E * G * cos(T(m_int) * (phi - phi_m))
    end

    function LensDerivative(x, y; m::Real, amp::Real, phi_m::Real,
                             theta_E::Real,
                             xcentre::Real=0., ycentre::Real=0.)
        m_int = Int(m)
        T = eltype(x)
        ax = zeros(T, size(x))
        ay = zeros(T, size(y))
        _multipole_deflection!(ax, ay, x, y, m_int, amp, phi_m,
                               theta_E, xcentre, ycentre)
        return ax, ay
    end

    function LensHessian(x, y; m::Real, amp::Real, phi_m::Real,
                          theta_E::Real,
                          xcentre::Real=0., ycentre::Real=0.)
        m_int = Int(m)
        # Use finite differences of LensDerivative for simplicity.
        # Multipole amplitudes are small (~1%) so 1e-8 FD is accurate.
        diff = 1e-8
        f_x, f_y = LensDerivative(x, y; m=m_int, amp, phi_m, theta_E,
                                   xcentre, ycentre)
        f_x_dx, _      = LensDerivative(x .+ diff, y; m=m_int, amp, phi_m,
                                         theta_E, xcentre, ycentre)
        f_x_dy, f_y_dy = LensDerivative(x, y .+ diff; m=m_int, amp, phi_m,
                                         theta_E, xcentre, ycentre)
        f_xx = (f_x_dx .- f_x) ./ diff
        f_xy = (f_x_dy .- f_x) ./ diff
        f_yy = (f_y_dy .- f_y) ./ diff
        return f_xx, f_xy, f_yy
    end

end # module MultipoleEPL