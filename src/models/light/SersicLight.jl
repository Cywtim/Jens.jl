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

    function SersicSpheric(x::AbstractArray, y::AbstractArray;
          amp::Real=1., Rsersic::Real=0.5,
        n::Real=0.5, xcentre::Real=0., ycentre::Real=0., bmin=1e-4)

        #=
        I(R) = amp \exp \left[ -b_n (R/R_{Sersic})^{\frac{1}{n}}\right]
        where $b_n \approx 1.999n-0.327$
        =#

        T = promote_type(eltype(x), eltype(y))
        two  = T(1.999)
        off  = T(0.327)
        bmin_T = T(bmin)
        b = max(two * n - off, bmin_T)

        R = @. sqrt((x - xcentre)^2 + (y - ycentre)^2)

        I = @. amp * exp( - b * ( R / Rsersic )^(1/n))

        return I
    end

    function SersicElliptical(x::AbstractArray, y::AbstractArray;
          amp::Real=1., Rsersic::Real=0.5, n::Real=0.5, varphi::Real=pi/3,
           q::Real=0.9, xcentre::Real=0., ycentre::Real=0., bmin::Real=1e-4)
        
        b = @. 2 * n - 1/3 + 4/(405 * n) + 46/(25515 * n^2) + 131/(1148175*n^3) - 2194697/(30690717750*n^4) # for n>0.36
        b = max(b, bmin)

        A = @. (x-xcentre) * cos(varphi) + (y-ycentre) * sin(varphi)
        B = @. -(x-xcentre) * sin(varphi) + (y-ycentre) * cos(varphi)
        R = @. sqrt(A^2 + (B / ( 1 - q ))^2)

        I = @. amp * exp( -b * ((R/Rsersic)^(1/n) - 1))

        return I 
    end


end