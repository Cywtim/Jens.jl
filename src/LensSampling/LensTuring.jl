module LensTuring

    using Turing
    using Distributions, Random, Statistics
    using StatsPlots, Optim
    using StatsPlots
    using PairPlots
    using CairoMakie

    include("../LensUtils.jl")
    include("../LensBase.jl")
    
    

    @model function FitModel(; lens)

        



    end

    function LensFit(LensModelDict_List::Union{Array{Dict},Dict},
         LensParaDict_List::Union{Array{Dict},Dict},
         Source
         )

        




    end

end