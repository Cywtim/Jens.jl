module Jens

    push!(LOAD_PATH, @__DIR__)
    
    # including packages
    include("LensBase.jl")
    include("LensFITS.jl")
    include("LensUtils.jl")
    include("LensGenerator.jl")
    include("LensCosmo.jl")
    include("LensNoise.jl")

    include("LensModel.jl")
    include("LightModel.jl")

end
