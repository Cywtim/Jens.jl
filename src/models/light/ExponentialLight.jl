"""
    ExponentialLight — Exponential Disk

I(R) = I₀ exp(−b R/R_eff)    with b ≈ 1.678

Special case of the Sérsic profile with n=1.  Models galaxy disks
with an exponential surface brightness falloff.

# Functions
- `ExponentialSpheric`: circular exponential
- `ExponentialElliptical`: elliptical exponential

# Parameters
- `amp`: amplitude / central surface brightness
- `Rsersic`: scale radius [arcsec]
- `varphi`: position angle [rad] (elliptical only)
- `q`: minor-to-major axis ratio (elliptical only)
- `xcentre`, `ycentre`: centre [arcsec]

# Example
    I = ExponentialSpheric(x, y; amp=1.0, Rsersic=0.5, xcentre=0, ycentre=0)
"""
module ExponentialLight
    # sersic index=1.0

    function ConfigCheck()

    end

    function ExponentialSpheric(x::AbstractArray, y::AbstractArray;
          amp::Real=1., Rsersic::Real=0.5,
         xcentre::Real=0., ycentre::Real=0., bmin=1e-4)

        T = promote_type(eltype(x), eltype(y))
        ampT = T(amp)
        b = max(T(1.999 - 0.327), T(bmin))
        RsersicT = T(Rsersic)
        xT = T(xcentre); yT = T(ycentre)

        R = @. sqrt((x - xT)^2 + (y - yT)^2)

        I = @. ampT * exp( - b * ( R / RsersicT ))

        return I

    end

    function ExponentialElliptical(x::AbstractArray, y::AbstractArray;
          amp::Real=1., Rsersic::Real=0.5, varphi::Real=pi/3,
           q::Real=0.9, xcentre::Real=0., ycentre::Real=0., bmin::Real=1e-4)

        T = promote_type(eltype(x), eltype(y))
        ampT = T(amp)
        b = max(T(2 - 1/3 + 4/405 + 46/25515 + 131/1148175 - 2194697/3069071775), T(bmin))
        RsersicT = T(Rsersic)
        xT = T(xcentre); yT = T(ycentre)

        A = @. (x - xT) * cos(varphi) + (y - yT) * sin(varphi)
        B = @. -(x - xT) * sin(varphi) + (y - yT) * cos(varphi)
        R = @. sqrt(A^2 + (B / q)^2)

        I = @. ampT * exp( -b * ((R/RsersicT) - 1))

        return I
    end
    

end