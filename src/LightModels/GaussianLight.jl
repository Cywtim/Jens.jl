
module GaussianLight

    using LazyGrids

    include("../LensUtils.jl")
    
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

        I = amp ./ (2 .* pi .* sigma .^ 2)
        R = (x .- xcentre) .^ 2 ./ sigma .^ 2 .+ (y .- ycentre) .^ 2 ./ sigma .^ 2
        
        return I .* exp.( .- R ./ 2.0)
        
    end

    function GaussianEllipse(x::AbstractArray, y::AbstractArray;
        amp::Real=1., sigma::Real=0.5, e1::Real=0.0, e2::Real=0.0, xcentre::Real=0., ycentre::Real=0.)

        xsh, ysh = LensUtils.EllipticalDistortion(x, y; e1, e2, xcentre, ycentre)

        G = GaussianSphere(xsh, ysh; amp=amp, sigma=sigma, xcentre=0.0, ycentre=0.0)

        return G


    end

    

end