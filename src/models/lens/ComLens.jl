module ComLens

    import ...LensBase: AbstractLens, lens_derivative, lens_hessian, lens_potential, lens_check

    # ═══════════════════════════════════════════════════════════════
    #  ComLens — Combined Lens Model Builder
    #
    #  Takes a list of (LensModel, parameters) pairs and returns
    #  a single Module whose LensDerivative / LensHessian /
    #  LensPotential sum the contributions of all sub-models.
    #
    #  The returned Module is a first-class citizen compatible
    #  with ALL LensBase functions (LensCaustic, LensFermat,
    #  LensMagnification, LensCriticalCurve, etc.).
    #
    #  USAGE ─────────────────────────────────────────────────
    #
    #    my_lens = ComLens.MyLens(
    #        NFW    => (Rs=1.0, alpha_Rs=0.5,  xcentre=0., ycentre=0.),
    #        NIEkappa => (b=0.6, s=0.1, q=0.5, varphi=pi/4,
    #                     xcentre=1.5, ycentre=-1.0),
    #    )
    #
    #    # With LensBase:
    #    mu = LB.LensMagnification(xg, yg;
    #        LensModel  = my_lens,
    #        LensKwargs = Dict{Symbol,Float64}())
    #
    #    # For critical curve / caustic, xcentre must be in Dict
    #    # (LensBase accesses it directly — ComLens ignores it):
    #    ccx, ccy = LB.LensCaustic(;
    #        LensModel  = my_lens,
    #        LensKwargs = Dict{Symbol,Float64}(:xcentre=>0., :ycentre=>0.))
    #
    # ═══════════════════════════════════════════════════════════

    export MyLens, LensPara, JitLens

    import ...LensBase: lens_derivative, lens_hessian, lens_potential, lens_check

    # ── Internal: convert NamedTuple pairs to (Module, Dict) ─
    function _to_pair(m::Module, nt::NamedTuple)
        d = Dict{Symbol, Float64}()
        for (k, v) in pairs(nt)
            d[Symbol(k)] = Float64(v)
        end
        return m => d
    end
    _to_pair(m::Module, d::Dict{Symbol, Float64}) = m => d

    # ── Resolve potential function (LensPotential or LensMass) ─
    _potential_func(m::Module) = isdefined(m, :LensPotential) ?
        m.LensPotential : m.LensMass

    """
        MyLens(name::Symbol, pairs::Pair...)
        MyLens(pairs::Pair...; name=:CombinedLens)

    Build a combined lens Module from (Model => params) pairs.

    Each `params` can be a `NamedTuple` or a `Dict{Symbol, Float64}`.
    It must include **all** keyword arguments required by that model,
    including `xcentre` and `ycentre`.

    Returns a Module ready for use with any LensBase function.
    """
    function MyLens(name::Symbol, pairs::Pair...)
        comps = [_to_pair(p[1], p[2]) for p in pairs]
        _build_module(name, comps)
    end

    function MyLens(pairs::Pair...; name::Symbol=:CombinedLens)
        MyLens(name, pairs...)
    end

    # ── Helpers for introspection ───────────────────────────
    function LensPara(mod::Module)
        if !isdefined(mod, :__comlens_components__)
            error("$mod is not a ComLens combined module")
        end
        comps = mod.__comlens_components__
        println("$(mod)  —  $(length(comps)) components:")
        for (i, (m, d)) in enumerate(comps)
            println("  [$i] $(nameof(m)) → ", d)
        end
    end

    # ═══════════════════════════════════════════════════════════
    #  Module builder — MyLens (baked params)
    # ═══════════════════════════════════════════════════════════

    function _build_module(name::Symbol,
                           comps::Vector{Pair{Module, Dict{Symbol,Float64}}})

        mod = Module(name)

        # Store for introspection
        Core.eval(mod, :(const __comlens_components__ = $comps))

        # ── LensDerivative ───────────────────────────────────
        Core.eval(mod, quote
            function LensDerivative(x, y; kwargs...)
                fx = zeros(Float64, size(x))
                fy = zeros(Float64, size(y))
                for (model, params) in __comlens_components__
                    fxi, fyi = model.LensDerivative(x, y; params...)
                    fx .+= fxi
                    fy .+= fyi
                end
                return fx, fy
            end
        end)

        # ── LensHessian ──────────────────────────────────────
        Core.eval(mod, quote
            function LensHessian(x, y; kwargs...)
                fxx = zeros(Float64, size(x))
                fxy = zeros(Float64, size(x))
                fyy = zeros(Float64, size(x))
                for (model, params) in __comlens_components__
                    fxx_i, fxy_i, fyy_i = model.LensHessian(x, y; params...)
                    fxx .+= fxx_i
                    fxy .+= fxy_i
                    fyy .+= fyy_i
                end
                return fxx, fxy, fyy
            end
        end)

        # ── LensPotential / LensMass ─────────────────────────
        Core.eval(mod, quote
            function LensPotential(x, y; kwargs...)
                psi = zeros(Float64, size(x))
                for (model, params) in __comlens_components__
                    # Resolve: models may have LensPotential or LensMass
                    f = isdefined(model, :LensPotential) ?
                        model.LensPotential : model.LensMass
                    psi .+= f(x, y; params...)
                end
                return psi
            end
            const LensMass = LensPotential
        end)

        return mod
    end

    # ═══════════════════════════════════════════════════════════
    #  Module builder — JitLens (params via LensKwargs)
    # ═══════════════════════════════════════════════════════════

    """
        JitLens(name::Symbol, pairs::Pair...)
        JitLens(pairs::Pair...; name=:JitLens_Lens)

    Build a combined-lens Module for JitLens / optimization.

    Instead of baking parameter **values** into the module (like
    `MyLens`), you declare which parameter **names** each
    sub-model expects.  At runtime, parameters are read from
    `LensKwargs` using `ModelName_key` prefixes.

    # Example
        lens = ComLens.JitLens(
            NFW => (:Rs, :alpha_Rs, :xcentre, :ycentre),
            NIEkappa => (:b, :s, :q, :varphi, :xcentre, :ycentre),
        )

        function log_prob(theta)
            kw = Dict{Symbol,Float64}(
                :NFW_Rs=>theta[1], :NFW_alpha_Rs=>theta[2],
                :NFW_xcentre=>0., :NFW_ycentre=>0.,
                :NIE_b=>theta[3], :NIE_s=>theta[4],
                :NIE_q=>theta[5], :NIE_varphi=>theta[6],
                :NIE_xcentre=>0., :NIE_ycentre=>0.,
                :xcentre=>0., :ycentre=>0.,  # LensCriticalCurve
            )
            fx, fy = LB.LensDeflection(xg, yg;
                LensModel=lens, LensKwargs=kw)
            # … compute chi^2 …
        end
    """
    function JitLens(name::Symbol, pairs::Pair...)
        # Build (module, global_keys, local_keys) triples
        T = Tuple{Module, Vector{Symbol}, Vector{Symbol}}
        comps = T[]
        for (model, param_names) in pairs
            prefix = string(nameof(model))
            local_keys  = [Symbol(k) for k in param_names]
            global_keys = [Symbol(prefix * "_" * string(k)) for k in param_names]
            push!(comps, (model, global_keys, local_keys))
        end
        _build_JitLens_module(name, comps)
    end

    function JitLens(pairs::Pair...; name::Symbol=:JitLens_Lens)
        JitLens(name, pairs...)
    end

    function _build_JitLens_module(name::Symbol,
                                 comps::Vector{Tuple{Module, Vector{Symbol}, Vector{Symbol}}})

        mod = Module(name)
        Core.eval(mod, :(const __JitLens_components__ = $comps))

        # ── LensDerivative ───────────────────────────────────
        Core.eval(mod, quote
            function LensDerivative(x, y; kwargs...)
                fx = zeros(Float64, size(x))
                fy = zeros(Float64, size(y))
                for (model, global_keys, local_keys) in __JitLens_components__
                    d = Dict{Symbol, Float64}()
                    for (gk, lk) in zip(global_keys, local_keys)
                        d[lk] = kwargs[gk]
                    end
                    fxi, fyi = model.LensDerivative(x, y; d...)
                    fx .+= fxi
                    fy .+= fyi
                end
                return fx, fy
            end
        end)

        # ── LensHessian ──────────────────────────────────────
        Core.eval(mod, quote
            function LensHessian(x, y; kwargs...)
                fxx = zeros(Float64, size(x))
                fxy = zeros(Float64, size(x))
                fyy = zeros(Float64, size(x))
                for (model, global_keys, local_keys) in __JitLens_components__
                    d = Dict{Symbol, Float64}()
                    for (gk, lk) in zip(global_keys, local_keys)
                        d[lk] = kwargs[gk]
                    end
                    fxx_i, fxy_i, fyy_i = model.LensHessian(x, y; d...)
                    fxx .+= fxx_i
                    fxy .+= fxy_i
                    fyy .+= fyy_i
                end
                return fxx, fxy, fyy
            end
        end)

        # ── LensPotential / LensMass ─────────────────────────
        Core.eval(mod, quote
            function LensPotential(x, y; kwargs...)
                psi = zeros(Float64, size(x))
                for (model, global_keys, local_keys) in __JitLens_components__
                    d = Dict{Symbol, Float64}()
                    for (gk, lk) in zip(global_keys, local_keys)
                        d[lk] = kwargs[gk]
                    end
                    f = isdefined(model, :LensPotential) ?
                        model.LensPotential : model.LensMass
                    psi .+= f(x, y; d...)
                end
                return psi
            end
            const LensMass = LensPotential
        end)

        return mod
    end

    # ═══════════════════════════════════════════════════════════
    #  Struct-based CombinedLens (idiomatic Julia, duck-typed)
    #  — works with LensBase since ::Module constraints removed
    # ═══════════════════════════════════════════════════════════

    """
        CombinedLens(pairs::Pair...)

    Idiomatic struct-based combined lens.  Faster setup than
    `MyLens` (no Module generation), works with all LensBase
    functions after the `::Module` constraint was removed.

        cl = ComLens.CombinedLens(
            NFW => (Rs=1.0, alpha_Rs=0.5, xcentre=0., ycentre=0.),
            NIEkappa => (b=0.6, s=0.1, q=0.5, varphi=pi/4),
        )
        mu = LB.LensMagnification(xg, yg; LensModel=cl, LensKwargs=Dict())
    """
    struct CombinedLens{M<:Tuple, P<:Tuple} <: AbstractLens
        models::M
        params::P
    end

    function CombinedLens(pairs::Pair{<:Module, <:NamedTuple}...)
        models = Tuple(first(p) for p in pairs)
        paramss = Tuple(last(p) for p in pairs)
        return CombinedLens{typeof(models), typeof(paramss)}(models, paramss)
    end

    function LensDerivative(cl::CombinedLens, x, y; kwargs...)
        fx = zeros(Float64, size(x))
        fy = zeros(Float64, size(y))
        for (m, p) in zip(cl.models, cl.params)
            fxi, fyi = m.LensDerivative(x, y; p...)
            fx .+= fxi; fy .+= fyi
        end
        return fx, fy
    end

    function LensHessian(cl::CombinedLens, x, y; kwargs...)
        fxx = zeros(Float64, size(x))
        fxy = zeros(Float64, size(x))
        fyy = zeros(Float64, size(x))
        for (m, p) in zip(cl.models, cl.params)
            fxx_i, fxy_i, fyy_i = m.LensHessian(x, y; p...)
            fxx .+= fxx_i; fxy .+= fxy_i; fyy .+= fyy_i
        end
        return fxx, fxy, fyy
    end

    function LensMass(cl::CombinedLens, x, y; kwargs...)
        psi = zeros(Float64, size(x))
        for (m, p) in zip(cl.models, cl.params)
            f = isdefined(m, :LensPotential) ? m.LensPotential : m.LensMass
            psi .+= f(x, y; p...)
        end
        return psi
    end

    function LensCheck(cl::CombinedLens; kwargs...)
        for (m, p) in zip(cl.models, cl.params)
            if isdefined(m, :LensCheck)
                m.LensCheck(; p...)
            end
        end
    end

    # ═══════════════════════════════════════════════════════════
    #  LensBase interface extensions for CombinedLens
    #
    #  These let CombinedLens work directly with LB.LensPlane(),
    #  LB.LensCaustic(), LB.LensMagnification(), etc. — no need
    #  for to_module().
    #
    #  Usage:
    #      cl = ComLens.CombinedLens(NFW=>(...), NIEkappa=>(...))
    #      beta_x, beta_y = LB.LensPlane(xg, yg;
    #          LensModel=cl, LensKwargs=Dict())
    # ═══════════════════════════════════════════════════════════

    function lens_derivative(cl::CombinedLens, x, y; z_source=nothing, kwargs...)
        fx = zeros(Float64, size(x))
        fy = zeros(Float64, size(y))
        for (m, p) in zip(cl.models, cl.params)
            fxi, fyi = m.LensDerivative(x, y; p...)
            fx .+= fxi; fy .+= fyi
        end
        return fx, fy
    end

    function lens_hessian(cl::CombinedLens, x, y; z_source=nothing, kwargs...)
        fxx = zeros(Float64, size(x))
        fxy = zeros(Float64, size(x))
        fyy = zeros(Float64, size(x))
        for (m, p) in zip(cl.models, cl.params)
            fxx_i, fxy_i, fyy_i = m.LensHessian(x, y; p...)
            fxx .+= fxx_i; fxy .+= fxy_i; fyy .+= fyy_i
        end
        return fxx, fxy, fyy
    end

    function lens_potential(cl::CombinedLens, x, y; z_source=nothing, kwargs...)
        psi = zeros(Float64, size(x))
        for (m, p) in zip(cl.models, cl.params)
            f = isdefined(m, :LensPotential) ? m.LensPotential : m.LensMass
            psi .+= f(x, y; p...)
        end
        return psi
    end

    function lens_check(cl::CombinedLens; z_source=nothing, kwargs...)
        for (m, p) in zip(cl.models, cl.params)
            if isdefined(m, :LensCheck)
                m.LensCheck(; p...)
            end
        end
    end

    """
        cl = ComLens.CombinedLens(NFW=>(...), NIEkappa=>(...))
        lm = ComLens.to_module(cl, :MyLens)
        mu = LB.LensMagnification(xg, yg; LensModel=lm, LensKwargs=Dict())
    """
    function to_module(cl::CombinedLens, name::Symbol=:CombinedLens)
        mod = Module(name)
        Core.eval(mod, :(import Jens))
        Core.eval(mod, :(const __cl = $cl))
        Core.eval(mod, quote
            function LensDerivative(x, y; kwargs...)
                return Jens.LensModel.ComLens.LensDerivative(__cl, x, y; kwargs...)
            end
            function LensHessian(x, y; kwargs...)
                return Jens.LensModel.ComLens.LensHessian(__cl, x, y; kwargs...)
            end
            function LensMass(x, y; kwargs...)
                return Jens.LensModel.ComLens.LensMass(__cl, x, y; kwargs...)
            end
            function LensCheck(; kwargs...)
                Jens.LensModel.ComLens.LensCheck(__cl; kwargs...)
            end
        end)
        return mod
    end

end # module ComLens
