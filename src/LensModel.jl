module LensModel

    push!(LOAD_PATH, @__DIR__)

    # including Lens models
    include("LensModels/EPL.jl")
    include("LensModels/NFW.jl")
    include("LensModels/NIEkappa.jl")
    include("LensModels/NIE.jl")
    include("LensModels/SIE.jl")
    include("LensModels/SIS.jl")
    include("LensModels/Gaussian.jl")
    include("LensModels/NFWE.jl")
    include("LensModels/Shear.jl")
    include("LensModels/SISreal.jl")
    include("LensModels/PointMass.jl")
    
end