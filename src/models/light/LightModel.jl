module LightModel

    # including Light models (same directory)
    include("ExponentialLight.jl")
    include("GaussianLight.jl")
    include("SersicLight.jl")
    include("PointSource.jl")

    export PointSource

end
