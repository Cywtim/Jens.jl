module LightModel

    # Core type hierarchy
    include("AbstractLight.jl")

    # including Light models (same directory)
    include("ExponentialLight.jl")
    include("GaussianLight.jl")
    include("SersicLight.jl")
    include("PointSource.jl")        # backward compat: PointSource alias

    export PointSource, PointImages

end