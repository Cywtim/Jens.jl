"""
    SersicLight — Sérsic Profile

I(R) = I₀ exp[−b_n (R/R_eff)^{1/n}]

The most commonly used galaxy surface brightness profile.
Spherical and elliptical variants.  n=1 recovers the exponential
disk, n=4 is the de Vaucouleurs profile.

# Functions
- `SersicSpheric`: spherical Sérsic
- `SersicElliptical`: elliptical Sérsic with q, φ

# Parameters
- `amp`: amplitude / central surface brightness
- `Rsersic`: effective (half-light) radius [arcsec]
- `n`: Sérsic index
- `varphi`: position angle [rad] (elliptical only)
- `q`: minor-to-major axis ratio (elliptical only)
- `xcentre`, `ycentre`: centre [arcsec]

# References
- Ciotti & Bertin (1999), arXiv:astro-ph/9911078
- Graham & Driver (2005), arXiv:astro-ph/0503176

# Example
    I = SersicSpheric(x, y; amp=1.0, Rsersic=0.5, n=2.0, xcentre=0, ycentre=0)
"""
module SersicLight
    # https://arxiv.org/pdf/1009.4713
    # https://arxiv.org/pdf/2306.05454
    # LensUtils available via parent module (Jens.LensUtils)

    function ConfigCheck()

    end

    # ── Shared b_n approximation ──────────────────────────────
    #  Ciotti & Bertin (1999) series expansion, valid for n > 0.36.
    #  Used by both SersicSpheric and SersicElliptical so that
    #  `amp` has a consistent meaning across the two variants.
    @inline _b_n(n::Real) = 2n - 1/3 + 4/(405n) + 46/(25515n^2) +
                            131/(1148175n^3) - 2194697/(30690717750n^4)

    function SersicSpheric(x::AbstractArray, y::AbstractArray;
          amp::Real=1., Rsersic::Real=0.5,
        n::Real=0.5, xcentre::Real=0., ycentre::Real=0., bmin=1e-4)

        # I(R) = amp * exp[-b_n * ((R/R_eff)^{1/n} - 1)]
        # b_n from Ciotti & Bertin (1999) series — same as Elliptical

        T = promote_type(eltype(x), eltype(y))
        ampT = T(amp)
        b = max(T(_b_n(n)), T(bmin))
        RsersicT = T(Rsersic)
        nT = T(n)
        xT = T(xcentre); yT = T(ycentre)

        R = @. sqrt((x - xT)^2 + (y - yT)^2)

        I = @. ampT * exp(-b * ((R / RsersicT)^(1/nT) - 1))

        return I
    end

    function SersicElliptical(x::AbstractArray, y::AbstractArray;
          amp::Real=1., Rsersic::Real=0.5, n::Real=0.5, varphi::Real=pi/3,
           q::Real=0.9, xcentre::Real=0., ycentre::Real=0., bmin::Real=1e-4)

        T = promote_type(eltype(x), eltype(y))
        ampT = T(amp)
        b = max(T(_b_n(n)), T(bmin))
        RsersicT = T(Rsersic)
        nT = T(n)
        xT = T(xcentre); yT = T(ycentre)

        A = @. (x - xT) * cos(varphi) + (y - yT) * sin(varphi)
        B = @. -(x - xT) * sin(varphi) + (y - yT) * cos(varphi)
        # Elliptical radius: R = sqrt(A^2 + (B/q)^2)
        R = @. sqrt(A^2 + (B / q)^2)

        I = @. ampT * exp(-b * ((R/RsersicT)^(1/nT) - 1))

        return I

    end

end