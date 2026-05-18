module LensTuring

    using Turing
    using Distributions, Random, Statistics
    using StatsPlots, Optim
    using PairPlots
    using CairoMakie

    include("../LensUtils.jl")
    include("../LensBase.jl")
    
    #=
    LensModelList = []
    LensModelKwargs = {:b => 1., etc. }
    LensLightKwargs = {:amp => 1., etc. }
    SourceLightKwargs = {:amp=>1., etc. }
    =#

    @model function make_dist(Priors)
        Params = Dict{Symbol, Any}()
        for (name, spec) in Priors
            if spec isa Dict
                if haskey(spec, :mean) && haskey(spec, :sigma) && haskey(spec, :lower) && haskey(spec, :upper)
                    mean  = spec[:mean]
                    sigma = spec[:sigma]
                    lower = spec[:lower]
                    upper = spec[:upper]
                    para ~ truncated(Normal(mean, sigma), lower, upper)
                    Params[name] = para
                elseif haskey(spec, :mean) && haskey(spec, :sigma)
                    mean  = spec[:mean]
                    sigma = spec[:sigma]
                    para ~ Normal(mean, sigma)
                    Params[name] = para
                elseif haskey(spec, :lower) && haskey(spec, :upper)
                    lower = spec[:lower]
                    upper = spec[:upper]
                    para ~ Uniform(lower, upper)
                    Params[name] = para
                else
                    error("Unsupported prior for $name")
                end
            else
                # fixed parameter
                Params[name] = spec
            end
        end
        return Params

    end

    @model function SingleFitModel(x, y; lens_mean::Dict, 
        lens_upper::Dict, lens_lower::Dict,)

        
        



    end

    function LensFit(LensModelDict_List::Union{Array{Dict},Dict},
         LensParaDict_List::Union{Array{Dict},Dict},
         Source
         )

        




    end

end