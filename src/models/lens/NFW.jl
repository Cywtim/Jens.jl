"""
    NFW — Navarro-Frenk-White

ρ(r) = ρ₀ / [(r/Rs)(1 + r/Rs)²]

The universal CDM halo density profile.  Analytic lensing quantities
use the Bartelmann (1996) and Wright & Brainerd (2000) closed forms.
Fully type-generic for GPU compatibility.

# Parameters
- `Rs`: scale radius [arcsec]
- `alpha_Rs`: deflection scale at Rs [arcsec]
- `xcentre`, `ycentre`: lens centre [arcsec]

# References
- Bartelmann (1996), arXiv:astro-ph/9602053
- Wright & Brainerd (2000), arXiv:astro-ph/9908213
- Golse & Kneib (2002), arXiv:astro-ph/0112138

# Example
    lens = SingleModel(NFW; Rs=5.0, alpha_Rs=0.5)
"""
module NFW
    
    # 10.48550/arXiv.astro-ph/9602053 
    # https://arxiv.org/abs/astro-ph/9611107)
    # https://doi.org/10.1051/0004-6361/202346308  eq(3),
    # https://doi.org/10.1046/j.1365-8711.2003.06276.x  

    using Jens.LensUtils


    function LensCheck(; Rs::Real, alpha_Rs::Real,
                         xcentre::Real=0., ycentre::Real=0.)
        para = [Rs, alpha_Rs, xcentre, ycentre]
        if all([0., 0., -100., -100.] .< para) && all([100., 100., 100., 100.] .> para)
            return
        else
            error("The NFW configuration is out of range!")
        end
    end

    function alpha2rho0(alpha_Rs::Real, Rs::Real)
        T = promote_type(typeof(alpha_Rs), typeof(Rs))
        rho0 = alpha_Rs / (T(4) * Rs^2 * (one(T) + log(one(T) / T(2))))
        return rho0
    end

    # GPU-compatible: ifelse + generic types (no Float64 literals).
    #  ifelse evaluates ALL branches, so we clamp sqrt/acosh/acos args
    #  to safe domains within each unused branch.
    function h(r_rs)
        T = typeof(r_rs)
        eps_t = T(1e-6)
        r = max(eps_t, r_rs)
        zero_t = zero(T)
        one_t  = one(T)
        two_t  = T(2)

        lt1 = r < one_t
        eq1 = r == one_t

        # a1 branch (r<1): clamp sqrt arg ≥0, acosh arg ≥1
        sq1 = max(one_t - r^2, zero_t)
        ac1 = max(one_t / r, one_t)
        a1 = log(r / two_t) + one_t / sqrt(sq1) * acosh(ac1)

        a2 = one_t + log(one_t / two_t)

        # a3 branch (r>1): clamp sqrt arg ≥0, acos arg ≤1
        sq3 = max(r^2 - one_t, zero_t)
        ac3 = min(one_t / r, one_t)
        a3 = log(r / two_t) + one_t / sqrt(sq3) * acos(ac3)

        return ifelse(lt1, a1, ifelse(eq1, a2, a3))
    end

    function potential(R, Rs, rho0)
        T = eltype(R)
        r_rs = @. R / Rs
        hx = h.(r_rs)
        p = @. T(2) * rho0 * Rs^2 * hx
        return p
    end

    function LensPotential(x, y; Rs, alpha_Rs, xcentre=0., ycentre=0.)
        rho0 = alpha2rho0(alpha_Rs, Rs)
        Rs = max(Rs, 1e-6)
        xsh = @. x - xcentre
        ysh = @. y - ycentre
        R = @. sqrt(xsh^2 + ysh^2)
        f = potential(R, Rs, rho0)
        return f
    end

    function alpha(R, Rs, rho0)
        T = eltype(R)
        R = max.(R, T(1e-6))
        r_rs = @. R / Rs
        gx = g.(r_rs)
        a = @. T(4) * rho0 * Rs * gx / r_rs^2
        return a 
    end

    # GPU-compatible: same pattern as h() — ifelse + domain clamping
    function g(r_rs)
        T = typeof(r_rs)
        eps_t = T(1e-6)
        r = max(eps_t, r_rs)
        zero_t = zero(T)
        one_t  = one(T)
        two_t  = T(2)

        lt1 = r < one_t
        eq1 = r == one_t

        # a1 (r<1): clamp sqrt arg ≥0, acosh arg ≥1
        sq1 = max(one_t - r^2, zero_t)
        ac1 = max(one_t / r, one_t)
        a1 = log(r / two_t) + one_t / sqrt(sq1) * acosh(ac1)

        a2 = one_t + log(one_t / two_t)

        # a3 (r>1): clamp sqrt arg ≥0, acos arg ≤1
        sq3 = max(r^2 - one_t, zero_t)
        ac3 = min(one_t / r, one_t)
        a3 = log(r / two_t) + one_t / sqrt(sq3) * acos(ac3)

        return ifelse(lt1, a1, ifelse(eq1, a2, a3))
    end

    function LensDerivative(x, y; Rs, alpha_Rs, xcentre=0., ycentre=0.)

        rho0 = alpha2rho0(alpha_Rs, Rs)
        Rs = max(Rs, 1e-6)

        xsh = @. x - xcentre
        ysh = @. y - ycentre
        R = @. sqrt(xsh^2 + ysh^2)

        a = alpha(R, Rs, rho0)
        f_x = @. a * xsh
        f_y = @. a * ysh

        return f_x, f_y

    end

    function kappa(x, y, Rs, rho0, xcentre=0., ycentre=0.)
        xsh = @. x - xcentre
        ysh = @. y - ycentre
        R = @. sqrt(xsh^2+ysh^2)
        T = eltype(R)
        r_rs = @. R / Rs
        Fx = f.(r_rs)
        kappa = @. T(2) * rho0 * Rs * Fx
        return kappa
    end

    function gamma(x, y, R, Rs, rho0)
        T = eltype(R)
        c = T(1e-8)
        R = max.(R, c)
        r_rs = @. R / Rs
        gx = g.(r_rs)
        Fx = f.(r_rs)
        a = @. T(2) * rho0 * Rs * (T(2) * gx / r_rs^2 - Fx)
        shear1 = @. a * (y^2 - x^2) / R^2
        shear2 = @. -a * T(2) * (x * y) / R^2
        return shear1, shear2
    end

    # GPU-compatible: ifelse + generic types + domain clamping.
    #  r=0 is a removable singularity; handled via ε bump.
    function f(r_rs)
        T = typeof(r_rs)
        zero_t  = zero(T)
        one_t   = one(T)
        two_t   = T(2)
        three_t = T(3)
        eps_t   = T(1e-8)

        # Bump r=0 to eps_t to keep all branches well-defined
        r = ifelse(r_rs == zero_t, eps_t, r_rs)

        lt1 = (r > zero_t) & (r < one_t)
        eq1 = r == one_t

        # a_lt1: r<1 → clamp sqrt arg ≥0, atanh arg <1
        sq_lt = max(one_t - r^2, zero_t)
        at_arg_lt = min(sqrt(max((one_t - r) / (one_t + r), zero_t)), one_t - eps_t)
        a_lt1 = one_t / (r^2 - one_t) *
                (one_t - two_t / sqrt(sq_lt) * atanh(at_arg_lt))

        a_eq1 = one_t / three_t

        # a_gt1: r>1 → clamp both sqrt args ≥0
        sq_gt = max(r^2 - one_t, zero_t)
        at_arg_gt = sqrt(max((r - one_t) / (one_t + r), zero_t))
        a_gt1 = one_t / (r^2 - one_t) *
                (one_t - two_t / sqrt(sq_gt) * atan(at_arg_gt))

        return ifelse(eq1, a_eq1, ifelse(lt1, a_lt1, a_gt1))
    end
                                                              

    function LensHessian(x, y; Rs, alpha_Rs, xcentre=0., ycentre=0.)

        rho0 = alpha2rho0(alpha_Rs, Rs)
        Rs = max(Rs, 1e-6)

        xsh = @. x - xcentre
        ysh = @. y - ycentre
        R = @. sqrt(xsh^2 + ysh^2)
        # kappa — pass shifted coords directly (no double-shift)
        kappa0 = kappa(xsh, ysh, Rs, rho0)

        # gamma
        gamma1, gamma2 = gamma(xsh, ysh, R, Rs, rho0)
        f_xx = @. kappa0 + gamma1
        f_yy = @. kappa0 - gamma1
        f_xy = @. gamma2
        return f_xx, f_xy, f_yy
 
    end

end