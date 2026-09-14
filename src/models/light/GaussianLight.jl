
"""
    GaussianLight — Gaussian Profile

I(R) = A/(2πσ²) exp(−R²/(2σ²))

Simple spherical and elliptical Gaussian light sources.

# Functions
- `GaussianSphere`: circular Gaussian
- `GaussianEllipse`: elliptical Gaussian with e1, e2

# Parameters
- `amp`: total flux
- `sigma`: Gaussian width [arcsec]
- `e1`, `e2`: ellipticity components (elliptical only)
- `xcentre`, `ycentre`: centre [arcsec]

# Example
    I = GaussianSphere(x, y; amp=1.0, sigma=0.5, xcentre=0, ycentre=0)
"""
module GaussianLight


    using Jens.LensUtils
    
    function ConfigCheck(amp::Real=1., sigma::Real=0.5, xcentre::Real=0., ycentre::Real=0.)


        para = [amp, sigma, xcentre, ycentre]

        if all([0., 0., -100, -100]< para) && all([100, 100, 100, 100]>para)
            return
        else
            error("The Gaussian configuration is out of range!")        
        end
        
    end

    function ConfigCheckEllipse(amp::Real=1., sigma::Real=0.5, e1::Real=0.0, e2::Real=0.0, xcentre::Real=0., ycentre::Real=0.)


        para = [amp, sigma, e1, e2, xcentre, ycentre]

        if all([0., 0., -0.5, -0.5, -100, -100]< para) && all([100, 100, 0.5, 0.5, 100, 100]>para)
            return
        else
            error("The Elliptical Gaussian configuration is out of range!")        
        end
        
    end

    function GaussianSphere(x::AbstractArray, y::AbstractArray;
          amp::Real, sigma::Real, xcentre::Real=0.0, ycentre::Real=0.0)

        T = promote_type(eltype(x), eltype(y))
        ampT = T(amp)
        sigmaT = T(sigma)
        twoT = T(2)
        xT = T(xcentre); yT = T(ycentre)

        I = ampT ./ (twoT .* T(pi) .* sigmaT .^ 2)
        R = (x .- xT) .^ 2 ./ sigmaT .^ 2 .+ (y .- yT) .^ 2 ./ sigmaT .^ 2

        return I .* exp.( .- R ./ twoT)

    end

    function GaussianEllipse(x::AbstractArray, y::AbstractArray;
        amp::Real=1., sigma::Real=0.5, e1::Real=0.0, e2::Real=0.0, xcentre::Real=0., ycentre::Real=0.)

        xsh, ysh = LensUtils.EllipticalDistortion(x, y; e1, e2, xcentre, ycentre)

        G = GaussianSphere(xsh, ysh; amp=amp, sigma=sigma, xcentre=0.0, ycentre=0.0)

        return G


    end

    

end