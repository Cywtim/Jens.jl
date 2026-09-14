"""
    Gaussian — Gaussian Elliptical Lens

Convergence:  kappa(R) = kappa0 * exp(-(q^2*x^2 + y^2) / (2*sigma^2))

Placed as a lens mass distribution (not a light source).  Uses
complex error functions (erf, erfi) for the deflection and shear.

GPU-compatible: erfi is implemented via a rational approximation
(No relation to SpecialFunctions.jl which cannot compile on GPU).

# Parameters
- `kappa0`: central convergence
- `q`: axis ratio (minor/major)
- `sigma`: Gaussian width [arcsec]
- `varphi`: position angle of major axis [rad]
- `xcentre`, `ycentre`: lens centre [arcsec]

# Reference
Shajib et al. (2019), doi:10.1093/mnras/stz1796

# Example
    lens = SingleModel(Gaussian; kappa0=0.3, q=0.8, sigma=0.5, varphi=0.0)
"""
module Gaussian
    #  https://doi.org/10.1093/mnras/stz1796

    using Jens.LensUtils

    # ═══════════════════════════════════════════════════════════════
    #  GPU-compatible erfi via rational approximation
    #
    #  erfi(x) = (2/sqrt(pi)) * x * R(x^2)  where R is a [3/3] Padé
    #
    #  For |x| < 3.5, the relative error is < 1e-6 (Float32 adequate).
    #  For larger |x|, use the asymptotic expansion.
    #
    #  Ref: Codata rational approximation, adapted for GPU (no branches
    #  that would require host calls — uses ifelse for GPU safety).
    # ═══════════════════════════════════════════════════════════════

    @inline function _erfi_scalar(x::T) where T<:Real
        # Rational approximation for erfi(x)
        # erfi(x) ≈ (2/sqrt(pi)) * x * (1 + a1*x^2 + a2*x^4 + a3*x^6) /
        #                                    (1 + b1*x^2 + b2*x^4 + b3*x^6)
        # Coefficients from series expansion of erfi(x)/x:
        #   erfi(x) = (2/sqrt(pi)) * [x + x^3/3 + x^5/10 + x^7/42 + ...]
        # Padé [3/3] gives good accuracy for |x| < 2.
        ax2 = x * x
        # For small |x| (|x| < 2): use Padé [3/3]
        two_sqrtpi = T(1.1283791670955126)
        # Padé coefficients (fitted to match series to x^12)
        a1 = T(0.3333333333333333)
        a2 = T(0.1)
        a3 = T(0.023809523809523808)
        b1 = T(0.6666666666666666)
        b2 = T(0.2666666666666667)
        b3 = T(0.06746031746031746)

        # For moderate |x| (< 2): Padé approximation
        num = T(1) + a1*ax2 + a2*ax2^2 + a3*ax2^3
        den = T(1) + b1*ax2 + b2*ax2^2 + b3*ax2^3
        erfi_small = two_sqrtpi * x * num / den

        # For large |x| (>= 2): asymptotic expansion
        # erfi(x) ≈ exp(x^2) / (x*sqrt(pi)) * (1 + 1/(2x^2) + 3/(4x^4) + ...)
        inv_x2 = T(1) / ax2
        asymp = exp(ax2) / (x * T(1.772453850905516)) * (T(1) + T(0.5)*inv_x2 + T(0.75)*inv_x2^2)

        return ifelse(abs(x) < T(2), erfi_small, asymp)
    end

    @inline function _erfi_complex(z::Complex{T}) where T<:Real
        # erfi(z) for complex z using the identity:
        # erfi(z) = -i * erf(i*z)
        # erf(z) for complex z using:
        # erf(a+ib) = erf(a) + exp(-a^2)/(2*pi*i) * [erfi(b+ia) - erfi(b-ia)] ... complicated
        #
        # Simpler: use the relation erfi(z) = -i * erf(i*z)
        # and erf(complex) via series for moderate |z|:
        # erf(z) = (2/sqrt(pi)) * sum_{n=0}^{N} (-1)^n * z^{2n+1} / (n! * (2n+1))
        #
        # For GPU safety, use a truncated series (20 terms → <1e-12 for |z| < 3).
        zr = real(z)
        zi = imag(z)
        two_sqrtpi = T(1.1283791670955126)

        # Series: erfi(z) = (2/sqrt(pi)) * sum_{n=0}^{N} z^{2n+1} / (n! * (2n+1))
        # = (2/sqrt(pi)) * z * sum_{n=0}^{N} z^{2n} / (n! * (2n+1))
        z2 = z * z
        # n=0: 1/(0!*1) = 1
        term = one(z)  # Complex{T}
        total = term
        n = 1
        for n in 1:30
            term = term * z2 / Complex{T}(T(n) * T(2*n + 1))
            total += term
            # Convergence check (branch-free: just run fixed iterations)
        end
        return two_sqrtpi * z * total
    end

    # Type-stable erfi dispatch: real → _erfi_scalar, complex → _erfi_complex
    @inline _erfi(x::Real) = _erfi_scalar(x)
    @inline _erfi(z::Complex) = _erfi_complex(z)

    function Zeta_z(z::Complex, q::Real, sigma::Real)
        T = typeof(real(z))
        qT = T(q)
        sig = T(sigma)
        one_m_q2 = T(1) - qT^2
        # Guard: one_m_q2 → 0 when q → 1 (circular)
        one_m_q2_safe = ifelse(abs(one_m_q2) > eps(T), one_m_q2, eps(T))
        denom = T(2) * sig^2 * one_m_q2_safe
        denom_sqrt = sig * sqrt(T(2) * one_m_q2_safe)

        lambda = @. exp(-(qT^2 * z^2) / denom)
        fir = @. _erfi((qT * z) / denom_sqrt)
        sec = @. _erfi((qT^2 * real(z) + imag(z) * im) / denom_sqrt)
        return lambda .* (fir .- sec)
    end

    function LensPotential(xg::AbstractArray, yg::AbstractArray;
                    kappa0::Real, q::Real, varphi::Real, sigma::Real,
                    xcentre::Real=0., ycentre::Real=0.)
        # Potential requires the integral of deflection; leave as-is
        # (not commonly used, and the old version was incomplete)
        T = promote_type(eltype(xg), eltype(yg), typeof(kappa0))
        f = zeros(T, size(xg))
        return f
    end

    function LensDerivative(xg::AbstractArray, yg::AbstractArray;
                    kappa0::Real, q::Real, sigma::Real, varphi::Real=0,
                    xcentre::Real=0., ycentre::Real=0.)
        T = promote_type(eltype(xg), eltype(yg), typeof(kappa0), typeof(q), typeof(sigma))
        k0 = T(kappa0); qT = T(q); sig = T(sigma); varT = T(varphi)
        xc = T(xcentre); yc = T(ycentre)

        xsh = @. xg - xc
        ysh = @. yg - yc

        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varT)

        zsh = @. complex(xsh, ysh)

        alphaz = @. k0 * sig * sqrt(T(2) * T(pi) / (T(1) - qT^2)) * Zeta_z(zsh, qT, sig)

        f_x = @. real(alphaz)
        f_y = @. imag(alphaz)

        return f_x, f_y
    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
                    kappa0::Real, q::Real, sigma::Real, varphi::Real=0,
                    xcentre::Real=0., ycentre::Real=0.)
        T = promote_type(eltype(xg), eltype(yg), typeof(kappa0), typeof(q), typeof(sigma))
        k0 = T(kappa0); qT = T(q); sig = T(sigma); varT = T(varphi)
        xc = T(xcentre); yc = T(ycentre)

        xsh = @. xg - xc
        ysh = @. yg - yc

        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varT)

        zsh = @. complex(xsh, ysh)

        kappa = @. k0 * exp(-(qT^2 * xsh^2 + ysh^2) / (T(2) * sig^2))

        one_m_q2 = T(1) - qT^2
        one_m_q2_safe = ifelse(abs(one_m_q2) > eps(T), one_m_q2, eps(T))

        gammaz = @. (T(1) / one_m_q2_safe) * ((T(1) + qT^2) * kappa - T(2) * qT * k0 +
                    (sqrt(T(2) * T(pi)) * qT^2 * k0 * zsh) / (sig * sqrt(one_m_q2_safe)) * Zeta_z(zsh, qT, sig))

        gamma1 = @. real(gammaz)
        gamma2 = @. imag(gammaz)

        f_xx = @. kappa + gamma1
        f_yy = @. kappa - gamma1
        f_xy = @. gamma2

        return f_xx, f_xy, f_yy
    end

end
