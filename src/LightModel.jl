module LightModel
    
    push!(LOAD_PATH, @__DIR__)

    # including Lens models
    include("LightModels/ExponentialLight.jl")
    include("LightModels/GaussianLight.jl")
    include("LightModels/Sersic.jl")


end