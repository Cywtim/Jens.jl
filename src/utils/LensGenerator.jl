module LensGenerator

    using Cosmology
    using ..LensUtils: ndgrid

    export LensInstance, SourceInstance

    # ═══════════════════════════════════════════════════════════════
    #  LensInstance
    # ═══════════════════════════════════════════════════════════════

    mutable struct LensInstance
        redshift::Float64
        cosmology::Cosmology.AbstractCosmology
        LensModels::Dict
        LightModels::Dict
        LensPlanes::Dict

        # Inner constructor — minimal, just validates & stores
        function LensInstance(
                redshift::Float64,
                cosmology::Cosmology.AbstractCosmology,
                LensModels::Dict,
                LightModels::Dict,
                LensPlanes::Dict,
            )
            return new(redshift, cosmology, LensModels, LightModels, LensPlanes)
        end
    end

    # ── Outer: keyword-based, all fields explicit ─────────────────
    function LensInstance(;
            redshift::Float64,
            cosmology::Cosmology.AbstractCosmology,
            LensModels::Dict  = Dict(),
            LightModels::Dict = Dict(),
            LensPlanes::Dict  = Dict(),
        )
        return LensInstance(redshift, cosmology, LensModels, LightModels, LensPlanes)
    end

    # ── Outer: convenience — auto-generate Cartesian grid ─────────
    function LensInstance(;
            num::Int           = 50,
            deltap::Float64    = 0.09,
            bkg_noise::Float64  = 0.0,
            exp_time::Float64   = 1.0,
            redshift::Float64   = 0.5,
            cosmology           = nothing,
            LensModels::Dict    = Dict(),
            LightModels::Dict   = Dict(),
        )
        if cosmology === nothing
            cosmology = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
        end
        x = range(-div(num, 2) * deltap, div(num, 2) * deltap; length = num + 1)
        xg, yg = ndgrid(collect(x), collect(x))
        LensPlanes = Dict(0.0 => (xg, yg))
        LensModels[:__noise__] = (; bkg_noise, exp_time)
        return LensInstance(redshift, cosmology, LensModels, LightModels, LensPlanes)
    end

    # ═══════════════════════════════════════════════════════════════
    #  SourceInstance
    # ═══════════════════════════════════════════════════════════════

    mutable struct SourceInstance
        redshift::Float64
        LightModels::Dict
        LensPlanes::Dict

        function SourceInstance(
                redshift::Float64,
                LightModels::Dict,
                LensPlanes::Dict,
            )
            return new(redshift, LightModels, LensPlanes)
        end
    end

    # ── Outer: keyword-based ─────────────────────────────────────
    function SourceInstance(;
            redshift::Float64 = 0.5,
            LightModels::Dict = Dict(),
            LensPlanes::Dict  = Dict(),
        )
        return SourceInstance(redshift, LightModels, LensPlanes)
    end

    # ── Outer: convenience — auto-grid ───────────────────────────
    function SourceInstance(;
            num::Int          = 50,
            deltap::Float64   = 0.09,
            redshift::Float64  = 0.5,
            LightModels::Dict  = Dict(),
        )
        x = range(-div(num, 2) * deltap, div(num, 2) * deltap; length = num + 1)
        xg, yg = ndgrid(collect(x), collect(x))
        LensPlanes = Dict(0.0 => (xg, yg))
        return SourceInstance(redshift, LightModels, LensPlanes)
    end

end
