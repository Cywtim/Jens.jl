
module LensTuring

    # for Turing
    using Turing,StatsPlots,PairPlots, Distributions
    using CairoMakie
    include("./LensBase.jl")

    function LensPrior()
        arg_names = Base.argument_names(model)

    end

    function TuringModel()

    end
    
end
