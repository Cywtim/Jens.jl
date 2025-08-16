module LensModel

    push!(LOAD_PATH, @__DIR__)

    # including Lens models
    include("LensModels/EPL.jl")
    include("LensModels/NFW.jl")
    include("LensModels/NIE.jl")
    include("LensModels/Shear.jl")
    include("LensModels/SIE.jl")
    include("LensModels/SIS.jl")
    
end