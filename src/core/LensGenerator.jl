module LensGenerator

    using Cosmology
    using Jens.LensUtils
    using Jens.LensUtils: ndgrid

    import ..LensBase: AbstractLens, lens_derivative, lens_hessian, lens_potential, lens_check
    import ..LensCosmo: lens_distance_ratio

    export LensedPlane, MultiLensedPlane
    export LensInstance, SourceInstance


    # ═══════════════════════════════════════════════════════════════
    #  LensInstance
    # ═══════════════════════════════════════════════════════════════


    # ==============================================================
    #  LensedPlane — single-plane lens with cosmological scaling
    #
    #  beta = theta - (D_ls/D_s) * alpha_phys(theta)
    #
    #  Wraps any lens model with redshift + cosmology so the
    #  lens equation distance ratio is applied automatically.
    #
    #  USAGE:
    #      cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0., 0.)
    #      cl    = CombinedLens(SIS=>(b=0.5, xcentre=0., ycentre=0.))
    #      lp    = LensedPlane(cl; z_lens=0.3, z_source=1.5, cosmology=cosmo)
    #      bx, by = LB.LensPlane(xg, yg; LensModel=lp, LensKwargs=Dict())
    # ==============================================================

    struct LensedPlane{L, C<:Cosmology.AbstractCosmology} <: AbstractLens
        lens::L
        z_lens::Float64
        z_source::Float64
        cosmology::C
    end

    function LensedPlane(lens; z_lens::Float64, z_source::Float64,
                         cosmology::Cosmology.AbstractCosmology)
        return LensedPlane(lens, z_lens, z_source, cosmology)
    end

    function lens_derivative(lp::LensedPlane, x, y; kwargs...)
        ratio = lens_distance_ratio(lp.cosmology, lp.z_lens, lp.z_source)
        aphys_x, aphys_y = lens_derivative(lp.lens, x, y; kwargs...)
        return aphys_x .* ratio, aphys_y .* ratio
    end

    function lens_hessian(lp::LensedPlane, x, y; kwargs...)
        ratio = lens_distance_ratio(lp.cosmology, lp.z_lens, lp.z_source)
        fxx, fxy, fyy = lens_hessian(lp.lens, x, y; kwargs...)
        return fxx .* ratio, fxy .* ratio, fyy .* ratio
    end

    function lens_potential(lp::LensedPlane, x, y; kwargs...)
        ratio = lens_distance_ratio(lp.cosmology, lp.z_lens, lp.z_source)
        psi = lens_potential(lp.lens, x, y; kwargs...)
        return psi .* ratio
    end

    function lens_check(lp::LensedPlane; kwargs...)
        lens_check(lp.lens; kwargs...)
    end

    # ==============================================================
    #  MultiLensedPlane — recursive multi-plane ray-tracing
    #
    #  Exact multi-plane lens equation for N lens planes at
    #  different redshifts.
    #
    #      theta_1 = theta                    (image plane)
    #      theta_{i+1} = theta - sum_{j≤i} beta_{j,i+1} * aphys_j(theta_j)
    #      beta = theta - sum_i beta_{i,source} * aphys_i(theta_i)  (source)
    #
    #  where beta_{j,k} = D(z_j, z_k) / D(z_k) is the distance ratio
    #  and aphys_i is the *physical* deflection of plane i.
    #
    #  USAGE:
    #      cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0., 0.)
    #      ml = MultiLensedPlane(
    #          ((CombinedLens(SIS=>(b=0.5, ...)), 0.3),
    #           (NIEkappa, 0.8));
    #          z_source   = 1.5,
    #          cosmology  = cosmo,
    #      )
    #      bx, by = LB.LensPlane(xg, yg; LensModel=ml, LensKwargs=Dict())
    # ==============================================================
    struct MultiLensedPlane{P<:Tuple, C<:Cosmology.AbstractCosmology} <: AbstractLens
        planes::P
        z_source::Float64
        cosmology::C
    end

    # Keyword constructor for convenience
    function MultiLensedPlane(planes::Tuple; z_source::Float64,
                            cosmology::Cosmology.AbstractCosmology)
        return MultiLensedPlane(planes, z_source, cosmology)
    end

    # ── Internal: shared ray-tracing ──────────────────────────
    function _trace_rays!(aphys_x, aphys_y, thetax, thetay,
                          z_planes, ml::MultiLensedPlane, x, y; kwargs...)
        N = length(ml.planes)
        # Plane 1: evaluated at image position theta
        thetax[1] = x
        thetay[1] = y
        aphys_x[1], aphys_y[1] = lens_derivative(ml.planes[1][1], x, y; kwargs...)
        # Planes 2..N: exact position from all previous planes
        for i in 2:N
            tx = copy(x)
            ty = copy(y)
            for j in 1:(i-1)
                beta_ratio = lens_distance_ratio(ml.cosmology, z_planes[j], z_planes[i])
                tx .-= beta_ratio .* aphys_x[j]
                ty .-= beta_ratio .* aphys_y[j]
            end
            thetax[i] = tx
            thetay[i] = ty
            aphys_x[i], aphys_y[i] = lens_derivative(
                ml.planes[i][1], tx, ty; kwargs...)
        end
        return nothing
    end

    function _alloc_ray_buffers(ml::MultiLensedPlane, x, y)
        N = length(ml.planes)
        z_planes = Float64[z for (_, z) in ml.planes]
        T = typeof(x)
        return (Vector{T}(undef, N), Vector{T}(undef, N),
                Vector{T}(undef, N), Vector{T}(undef, N), z_planes)
    end

    # ── lens_derivative: exact recursive ray-tracing ──────────
    function lens_derivative(ml::MultiLensedPlane, x, y; kwargs...)
        aphys_x, aphys_y, thetax, thetay, z_planes = _alloc_ray_buffers(ml, x, y)
        _trace_rays!(aphys_x, aphys_y, thetax, thetay, z_planes, ml, x, y; kwargs...)

        N = length(ml.planes)
        alpha_tot_x = zeros(Float64, size(x))
        alpha_tot_y = zeros(Float64, size(y))
        for i in 1:N
            ratio = lens_distance_ratio(ml.cosmology, z_planes[i], ml.z_source)
            alpha_tot_x .+= ratio .* aphys_x[i]
            alpha_tot_y .+= ratio .* aphys_y[i]
        end
        return alpha_tot_x, alpha_tot_y
    end

    # ── lens_hessian: recursive Jacobian accumulation ─────────
    function lens_hessian(ml::MultiLensedPlane, x, y; kwargs...)
        aphys_x, aphys_y, thetax, thetay, z_planes = _alloc_ray_buffers(ml, x, y)
        _trace_rays!(aphys_x, aphys_y, thetax, thetay, z_planes, ml, x, y; kwargs...)

        N = length(ml.planes)
        # Per-plane Hessians at their ray positions
        Hxx = Vector{typeof(x)}(undef, N)
        Hxy = Vector{typeof(x)}(undef, N)
        Hyy = Vector{typeof(x)}(undef, N)
        for i in 1:N
            Hxx[i], Hxy[i], Hyy[i] = lens_hessian(
                ml.planes[i][1], thetax[i], thetay[i]; kwargs...)
        end

        # Recursive Jacobian: A_1 = I, A_{k+1} = I - sum beta * H * A
        Axx = ones(Float64, size(x))
        Axy = zeros(Float64, size(x))
        Ayx = zeros(Float64, size(x))
        Ayy = ones(Float64, size(x))
        for i in 1:N
            beta_ratio = lens_distance_ratio(ml.cosmology, z_planes[i], ml.z_source)
            kxx = beta_ratio .* Hxx[i]
            kxy = beta_ratio .* Hxy[i]
            kyy = beta_ratio .* Hyy[i]
            Axx, Axy, Ayx, Ayy = (
                Axx .- kxx .* Axx .- kxy .* Ayx,
                Axy .- kxx .* Axy .- kxy .* Ayy,
                Ayx .- kxy .* Axx .- kyy .* Ayx,
                Ayy .- kxy .* Axy .- kyy .* Ayy,
            )
        end
        return (1.0 .- Axx), (-Axy), (1.0 .- Ayy)
    end

    # ── lens_potential: sum of scaled potentials (approximate) ─────
    function lens_potential(ml::MultiLensedPlane, x, y; kwargs...)
        aphys_x, aphys_y, thetax, thetay, z_planes = _alloc_ray_buffers(ml, x, y)
        _trace_rays!(aphys_x, aphys_y, thetax, thetay, z_planes, ml, x, y; kwargs...)

        psi = zeros(Float64, size(x))
        for i in 1:length(ml.planes)
            ratio = lens_distance_ratio(ml.cosmology, z_planes[i], ml.z_source)
            psi .+= ratio .* lens_potential(ml.planes[i][1], thetax[i], thetay[i]; kwargs...)
        end
        return psi
    end

    function lens_check(ml::MultiLensedPlane; kwargs...)
        for (lens_model, _) in ml.planes
            lens_check(lens_model; kwargs...)
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  LensInstance
    # ═══════════════════════════════════════════════════════════════

    mutable struct LensInstance
        redshift::Float64
        cosmology::Cosmology.AbstractCosmology
        LensModels::Dict
        LightModels::Dict
        LensPlanes::Dict

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

    # ── Single outer constructor — handles both explicit and auto-grid ─
    function LensInstance(;
            # Core fields
            redshift::Float64  = 0.5,
            cosmology          = nothing,
            LensModels::Dict   = Dict(),
            LightModels::Dict  = Dict(),
            LensPlanes::Dict   = Dict(),
            # Grid mode (num > 0 triggers auto-grid)
            num::Int           = 0,
            deltap::Float64    = 0.09,
            bkg_noise::Float64 = 0.0,
            exp_time::Float64  = 1.0,
        )
        if cosmology === nothing
            cosmology = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
        end
        if num > 0
            x = range(-div(num, 2) * deltap, div(num, 2) * deltap; length = num + 1)
            xg, yg = ndgrid(collect(x), collect(x))
            LensPlanes = Dict(0.0 => (xg, yg))
            LensModels[:__noise__] = (; bkg_noise, exp_time)
        end
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

    # ── Single outer constructor — explicit or auto-grid ───────────
    function SourceInstance(;
            redshift::Float64  = 0.5,
            LightModels::Dict  = Dict(),
            LensPlanes::Dict   = Dict(),
            num::Int           = 0,
            deltap::Float64    = 0.09,
        )
        if num > 0
            x = range(-div(num, 2) * deltap, div(num, 2) * deltap; length = num + 1)
            xg, yg = ndgrid(collect(x), collect(x))
            LensPlanes = Dict(0.0 => (xg, yg))
        end
        return SourceInstance(redshift, LightModels, LensPlanes)
    end

end
