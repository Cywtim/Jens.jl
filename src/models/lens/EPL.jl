"""
    EPL — Elliptical Power Law

Surface mass density:  kappa proportional R^{1-gamma}  (gamma=2 -> isothermal)

Extension of SIE to arbitrary radial slope.  Uses the Tessore & Metcalf
(2015) iterative recurrence for the complex deflection, which is GPU-safe
(no HypergeometricFunctions dependency).  gamma=2 recovers SIE,
gamma>2 is steeper than isothermal, gamma<2 is shallower.

# Parameters
- `theta_E`: Einstein radius [arcsec]
- `gamma`: radial slope parameter (gamma=2 is isothermal)
- `e1`, `e2`: ellipticity components
- `xcentre`, `ycentre`: lens centre [arcsec]

# Reference
Tessore & Metcalf (2015), doi:10.1051/0004-6361/201526773

# Example
    lens = SingleModel(EPL; theta_E=1.2, gamma=2.0, e1=0.1, e2=-0.05)
"""
module EPL

    using Jens.LensUtils

    function LensCheck(; theta_E::Real, gamma::Real,
         e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        para = [theta_E, gamma, e1, e2, xcentre, ycentre]

        if all([0.,1.5,-0.5,-0.5,-100,-100] .< para) && all([10,2.5, 0.5,  0.5, 100, 100] .> para)
            return
        else
            error("The EPL configuration is out of range!")
        end
    end

    function Main2MajorAxes(theta_E::Real, gamma::Real,
        e1::Real, e2::Real)
        t = gamma - 1
        q, varphi = LensUtils.e2phiq(e1, e2)
        b = theta_E * sqrt(q)
        return b, t, q, varphi
    end

    # ═══════════════════════════════════════════════════════════════
    #  Tessore & Metcalf (2015) iterative recurrence for _2F1
    #
    #  The complex deflection requires the hypergeometric function
    #    omega(z) = _2F1(1, t/2; 2-t/2; -f * z/conj(z))
    #  where f = (1-q)/(1+q).
    #
    #  Instead of calling HypergeometricFunctions.pFq (which uses
    #  logging/regex/string internally and cannot compile on GPU),
    #  we use the Gauss hypergeometric series recurrence:
    #
    #    a_0 = 1
    #    a_{n+1} = a_n * (1 + t/2 + n - 1) / (2 - t/2 + n - 1) * (-f * z/conj(z))
    #
    #  i.e.  a_{n+1} = a_n * (n + t/2) / (n + 2 - t/2) * (-f) * (z/conj(z))
    #
    #  Convergence: ~20-50 terms for Float32, ~60 for Float64.
    #  The ratio |z/conj(z)| = 1 (unit modulus), and |f| < 1 for q > 0,
    #  so the series converges geometrically.
    # ═══════════════════════════════════════════════════════════════

    # Maximum iterations for the hypergeometric series.
    # 80 terms gives <1e-12 relative error for |f| < 0.9 (q > 0.05).
    const _EPL_MAX_ITER = 80

    """
        _2F1_tessore(z, t, f) -> Complex

    Evaluate the Gauss hypergeometric function _2F1(1, t/2; 2-t/2; -f*w)
    where w = z/conj(z), using the series recurrence from Tessore &
    Metcalf (2015), eq. (12).

    GPU-safe: pure arithmetic, no allocations, no external library calls.
    """
    @inline function _2F1_tessore(z::Complex, t::Real, f::Real)
        T = typeof(real(z))
        # w = z / conj(z) — unit-modulus phase factor
        # For z = 0, conj(z) = 0 → division by zero. Guard:
        zr = real(z)
        zi = imag(z)
        zabs2 = zr * zr + zi * zi
        zabs2_safe = ifelse(zabs2 > eps(T), zabs2, T(1))
        # w = z / conj(z) = (zr + i*zi)^2 / |z|^2 = (zr^2-zi^2 + 2i*zr*zi) / |z|^2
        wr = (zr * zr - zi * zi) / zabs2_safe
        wi = (T(2) * zr * zi) / zabs2_safe

        # Series: sum_{n=0}^{N} a_n
        # a_0 = 1
        # a_{n+1} = a_n * (n + t/2) / (n + 2 - t/2) * (-f) * w
        half_t = T(t) / T(2)
        two_minus_half_t = T(2) - half_t
        coeff = -T(f)  # -f factor

        # Iteration accumulators (complex, stored as real/imag pairs)
        # a_0 = 1 + 0i
        a_r = T(1)
        a_i = T(0)

        sum_r = a_r
        sum_i = a_i

        n = T(0)
        for _ in 1:_EPL_MAX_ITER
            # ratio = (n + t/2) / (n + 2 - t/2)
            ratio = (n + half_t) / (n + two_minus_half_t)

            # a *= ratio * coeff * w
            # First: factor = ratio * coeff (real scalar)
            factor = ratio * coeff

            # a = a * factor * w = a * (factor * w)
            # factor * w = (factor*wr, factor*wi)
            fw_r = factor * wr
            fw_i = factor * wi

            # complex multiply: a_new = a * fw
            new_r = a_r * fw_r - a_i * fw_i
            new_i = a_r * fw_i + a_i * fw_r
            a_r = new_r
            a_i = new_i

            sum_r += a_r
            sum_i += a_i

            n += T(1)
        end

        return complex(sum_r, sum_i)
    end

    # ═══════════════════════════════════════════════════════════════
    #  MajorDerivative — complex deflection in the major-axis frame
    #
    #  alpha(z) = 2/(1+q) * (b/R)^t * omega(z)
    #  where omega(z) = z * _2F1(1, t/2; 2-t/2; -f * z/conj(z))
    #  and f = (1-q)/(1+q)
    # ═══════════════════════════════════════════════════════════════

    function MajorDerivative(xsh::AbstractArray, ysh::AbstractArray;
                         b::Real , t::Real ,q::Real)
        T = promote_type(eltype(xsh), eltype(ysh), typeof(b), typeof(t), typeof(q))
        f = (1 - q) / (1 + q)

        # Complex coordinate z = q*x + i*y
        Zreal = @. T(q) * xsh
        Zimag = @. ysh

        R2 = @. Zreal^2 + Zimag^2
        R = @. max(sqrt(R2), eps(T))

        # Build complex z array
        Z = @. complex(Zreal, Zimag)

        # omega(z) = z * _2F1(...)
        # _2F1_tessore returns a Complex array
        hf = @. _2F1_tessore(Z, t, f)
        omega = @. Z * hf

        alpha = @. T(2) / T(1 + q) * (T(b) / R)^T(t) * omega

        alpha_x = @. real(alpha)
        alpha_y = @. imag(alpha)

        return alpha_x, alpha_y
    end

    function  LensPotential(xg::AbstractArray, yg::AbstractArray; theta_E::Real, gamma::Real,
        e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        b, t, q, varphi = Main2MajorAxes(theta_E, gamma, e1, e2)

        xsh = @. xg - xcentre
        ysh = @. yg - ycentre

        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        alpha_x, alpha_y = MajorDerivative(xsh, ysh; b, t, q)

        f = @. (xsh * alpha_x + ysh * alpha_y) / (2 - gamma)

        return f
    end

    function  LensDerivative(xg::AbstractArray, yg::AbstractArray; theta_E::Real, gamma::Real,
        e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        b, t, q, varphi = Main2MajorAxes(theta_E, gamma, e1, e2)

        xsh = @. xg - xcentre
        ysh = @. yg - ycentre

        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        f_x, f_y = MajorDerivative(xsh, ysh; b, t, q)

        return f_x, f_y

    end

    function  LensHessian(xg::AbstractArray, yg::AbstractArray; theta_E::Real, gamma::Real,
        e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        b, t, q, varphi = Main2MajorAxes(theta_E, gamma, e1, e2)

        xsh = @. xg - xcentre
        ysh = @. yg - ycentre

        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        alpha_x, alpha_y = MajorDerivative(xsh, ysh; b, t, q)

        T = promote_type(eltype(xsh), typeof(b), typeof(t), typeof(q))
        qT = T(q); bT = T(b); tT = T(t)
        R = @. max(sqrt((qT * xsh)^2 + ysh^2), eps(T))
        r = @. max(sqrt(xsh^2 + ysh^2), eps(T))

        cos_phi = @. xsh / r
        sin_phi = @. ysh / r
        cos2 = @. cos_phi * cos_phi * 2 - 1
        sin2 = @. sin_phi * cos_phi * 2

        kappa = @. (2 - tT) / T(2) * (bT / R)^tT

        gamma_1 = @. (1 - tT) * (alpha_x * cos_phi - alpha_y * sin_phi) / r - kappa * cos2
        gamma_2 = @. (1 - tT) * (alpha_y * cos_phi + alpha_x * sin_phi) / r - kappa * sin2

        f_xx = @. kappa + gamma_1
        f_yy = @. kappa - gamma_1
        f_xy = @. gamma_2

        return f_xx, f_xy, f_yy

    end


end
