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

        #=
        I(R) = amp \exp \left[ -b_n (R/R_{Sersic})^{\frac{1}{n}}\right]
        where $b_n \approx 1.999n-0.327$
        =#

        b = max(1.999  - 0.327, bmin)

        R = @. sqrt((x - xcentre)^2 + (y - ycentre)^2)

        I = @. amp * exp( - b * ( R / Rsersic ))

        return I

    end

    function ExponentialElliptical(x::AbstractArray, y::AbstractArray;
          amp::Real=1., Rsersic::Real=0.5, varphi::Real=pi/3,
           q::Real=0.9, xcentre::Real=0., ycentre::Real=0., bmin::Real=1e-4)
        
        b = @. 2 - 1/3 + 4/(405 ) + 46/(25515) + 131/(1148175) - 2194697/(3069071775) # for n>0.36
        b = max(b, bmin)

        A = @. (x-xcentre) * cos(varphi) + (y-ycentre) * sin(varphi)
        B = @. -(x-xcentre) * sin(varphi) + (y-ycentre) * cos(varphi)
        R = @. sqrt(A^2 + (B / ( 1 - q ))^2)

        I = @. amp * exp( -b * ((R/Rsersic) - 1))

        return I 
    end
    

end