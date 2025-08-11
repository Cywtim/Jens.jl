module LensNoise

    using FITSIO, AstroLib, Distributions, AstroImages
    using Statistics, BackgroundMeshes, Random, LazyGrids

    export PoissonNoise, GaussianNoise, BackgroundNoise

    function PoissonNoise(image::AbstractArray,
         exp_time::Real)

        sigma =@.  sqrt(abs(image) / exp_time)
        
        poisson = rand(Normal(0., 1.), size(image)) .* sigma
    
        return poisson
    
    end

    function GaussianNoise(image::AbstractArray,
                        sigma_bkd::Real)

        Gauss = rand(Normal(0., 1.), size(image))  .* sigma_bkd
    
        return Gauss
    
    end

    function BackgroundNoise(img::AbstractArray;
        clipara::Dict=Dict(:fill=>NaN, :center=>median(img), :std=>std(img, corrected=false)),
        sigma::Real=1, boxsize::Real=50, filtersize::Real=5)

        clipped = sigma_clip(img, sigma; clipara...)

        bkg, bkg_rms = estimate_background(clipped, boxsize, filter_size=filtersize)

        return bkg, bkg_rms
        
    end
    
end