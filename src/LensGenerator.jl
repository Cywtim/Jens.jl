module LensGenerator

    using AstroLib, NLsolve, Optim, LazyGrids, Cosmology
    using Cosmology:AbstractCosmology as AbstractCosmology

    export LensInstance, SourceInstance

    include("LensUtils.jl")
    #LensModelnames = LensUtils.include_folder("LensModels") # load all models in LensModels
    
    #bkg_noise::Float64 # background noise
    #exp_time::Float64 # expsure time

    struct SingleLensInstance
        LensModels::Dict # Lens model dict {model=>params}
        LightModels::Dict # Source model dict {model=>params}
        LensPlanes::Dict # Lens plane dict, start with image plane, {back=>(x1,y1),fore=>(x2,y2)}
        #------------------------Constructor------------------------#
        #Main Constructor
        function SingleLensInstance(;
            LensModels::Dict=Dict(), LightModels::Dict=Dict(), 
            LensPlanes::Dict=Dict())
            
            return new(redshift, cosmology,
            LensModels, LightModels, LensPlanes)
        end

        #Quick Constructor
        function SingleLensInstance(;LensModels::Dict=Dict(), 
            LightModels::Dict=Dict(), LensPlanes::Dict=Dict())
 

            return new(LensModels, LightModels, LensPlanes)
        end


    end


    mutable struct  LensInstance

        redshift::Float64 # redshift of lens
        cosmology::AbstractCosmology
        LensModels::Dict # Lens model dict {model=>params}
        LightModels::Dict # Source model dict {model=>params}
        LensPlanes::Dict # Lens plane dict, start with image plane, {back=>(x1,y1),fore=>(x2,y2)}
        #------------------------Constructor------------------------#
        #Main Constructor
        function LensInstance(;redshift::Float64, cosmology::AbstractCosmology,
            LensModels::Dict=Dict(), LightModels::Dict=Dict(), 
            LensPlanes::Dict=Dict())
            
            return new(redshift, cosmology,
            LensModels, LightModels, LensPlanes)
        end
    
        # First/Single Lens Constructor
        function LensInstance(;num::Int=50, deltap::Float64=0.09,
            bkg_noise::Float64=0.0, exp_time::Float64=1.,
            LensModels::Dict=Dict(), LightModels::Dict=Dict())
    
            x = range(-div(num,2)*deltap, div(num,2)*deltap, num+1)
            xg,yg = LazyGrids.ndgrid(x,x)
            LensPlanes = Dict(0.0=>(xg, yg))
            
            return new(num, deltap,
            bkg_noise, exp_time, LensModels, LightModels, 
            LensPlanes)
        end

        # No lens light, First/Single Lens Constructor
        function LensInstance(;num::Int=50, deltap::Float64=0.09,
            bkg_noise::Float64=0.0, exp_time::Float64=1.,
            LensModels::Dict=Dict())
            
            LightModels=Dict()

            x = range(-div(num,2)*deltap, div(num,2)*deltap, num+1)
            xg,yg = LazyGrids.ndgrid(x,x)
            LensPlanes = Dict(0.0=>(xg, yg))
            
            return new(num, deltap,
            bkg_noise, exp_time, LensModels, LightModels, 
            LensPlanes)
        end

        # No lens light Constructor
        function LensInstance(;num::Int=50, deltap::Float64=0.09,
            bkg_noise::Float64=0.0, exp_time::Float64=1.,
            LensModels::Dict=Dict(), LensPlanes::Dict=Dict())
            
            LightModels=Dict()

            return new(num, deltap,
            bkg_noise, exp_time, LensModels, LightModels, 
            LensPlanes)
        end

    end

    mutable struct  SourceInstance

        redshift::Float64 # redshift of lens

        LightModels::Dict # Source model dict {model=>params}
        LensPlanes::Dict # Lens plane dict, start with image plane, {back=>(x1,y1),fore=>(x2,y2)}
        #------------------------Constructor------------------------#
        #Main Constructor
        function LensInstance(;num::Int=50, deltap::Float64=0.09,
            bkg_noise::Float64=0.0, exp_time::Float64=1.,
            LensModels::Dict=Dict(), LightModels::Dict=Dict(), 
            LensPlanes::Dict=Dict())
            
            return new(num, deltap, bkg_noise, exp_time, 
            LensModels, LightModels, LensPlanes)
        end
    
        #Auxiliary Constructor
        function LensInstance(;num::Int=50, deltap::Float64=0.09,
            bkg_noise::Float64=0.0, exp_time::Float64=1.,
            LensModels::Dict=Dict(), LightModels::Dict=Dict())
    
            x = range(-div(num,2)*deltap, div(num,2)*deltap, num+1)
            xg,yg = LazyGrids.ndgrid(x,x)
            LensPlanes = Dict(0.0=>(xg, yg))
            
            return new(num, deltap,
            bkg_noise, exp_time, LensModels, LightModels, 
            LensPlanes)
        end
    end

    mutable struct LensConstruct

        num::Int # Size of the lens image
        deltap::Float64  # pixel size of the image
        bkg_noise::Float64 # background noise
        exp_time::Float64 # expsure time

        LensDict::Dict
        LightDict::Dict

    end


end