module LensGenerator

    using Cosmology
    using Jens.LensUtils
    using Jens.LensUtils: ndgrid
    using Jens.LensPSF
    using Jens.LensSolver
    using Jens.LightModel: AbstractLight, ExtendedSource, PointImage, CompositeImage, evaluate_source

    import ..LensBase
    import ..LensBase: AbstractLens, lens_derivative, lens_hessian, lens_potential, lens_check,
                       _render_source
    import ..LensCosmo: lens_distance_ratio

    export LensedPlane, MultiLensedPlane, LightPlane, MultiLightPlane
    export LensInstance#, SourceInstance
    export Grid, GenGrid
    export render_lens
    export add_point


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
    #      lp    = LensedPlane(cl; z_lens=0.3, cosmology=cosmo)
    #      bx, by = LB.LensPlane(xg, yg; LensModel=lp, LensKwargs=Dict(), z_source=1.5)
    # ==============================================================

    struct LensedPlane{L, C<:Cosmology.AbstractCosmology} <: AbstractLens
        lens::L
        z_lens::Float64
        cosmology::C
    end

    function LensedPlane(lens; z_lens::Float64,
                         cosmology::Cosmology.AbstractCosmology)
        return LensedPlane(lens, z_lens, cosmology)
    end

    function lens_derivative(lp::LensedPlane, x, y; z_source, kwargs...)
        ratio = lens_distance_ratio(lp.cosmology, lp.z_lens, Float64(z_source))
        aphys_x, aphys_y = lens_derivative(lp.lens, x, y; kwargs...)
        return aphys_x .* ratio, aphys_y .* ratio
    end

    function lens_hessian(lp::LensedPlane, x, y; z_source, kwargs...)
        ratio = lens_distance_ratio(lp.cosmology, lp.z_lens, Float64(z_source))
        fxx, fxy, fyy = lens_hessian(lp.lens, x, y; kwargs...)
        return fxx .* ratio, fxy .* ratio, fyy .* ratio
    end

    function lens_potential(lp::LensedPlane, x, y; z_source, kwargs...)
        ratio = lens_distance_ratio(lp.cosmology, lp.z_lens, Float64(z_source))
        psi = lens_potential(lp.lens, x, y; kwargs...)
        return psi .* ratio
    end

    function lens_check(lp::LensedPlane; kwargs...)
        lens_check(lp.lens; kwargs...)
    end

    # ==============================================================
    #  LightPlane — redshift wrapper for an AbstractLight component
    #
    #      host = LightPlane(
    #          ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.5),
    #          z=1.5,
    #      )
    #      agn  = LightPlane(PointImage(flux=100, beta_x=0.3), z=1.5)
    #
    #  The `z` field is the cosmological redshift of the source.
    #  Pairs with `LensedPlane` in `LensSystem` to compute
    #  distance ratios for ray-tracing.
    # ==============================================================

    struct LightPlane{L<:AbstractLight}
        light::L
        z::Float64
    end

    function LightPlane(light::AbstractLight; z::Real)
        return LightPlane(light, Float64(z))
    end

    # ==============================================================
    #  MultiLightPlane — N light sources at different redshifts
    #
    #      mlp = MultiLightPlane(
    #          (host, 1.5),
    #          (agn,  2.0),
    #      )
    #
    #  Each element is `(light::AbstractLight, z::Float64)`.
    #  See `LensSystem` for pairing with `LensedPlane` planes.
    # ==============================================================

    struct MultiLightPlane{P<:Tuple}
        planes::P    # each element: (light::AbstractLight, z::Float64)
    end

    function MultiLightPlane(planes::Pair{<:AbstractLight, <:Real}...)
        nt = Tuple((p.first, Float64(p.second)) for p in planes)
        return MultiLightPlane{typeof(nt)}(nt)
    end

    # ═══════════════════════════════════════════════════════════════
    #  _render_source bridge — unwrap LightPlane for LensRayShooting
    #
    #  LensRayShooting auto-detects z_source from LightPlane.z
    #  (see LensBase.jl), and this method unwraps .light on the
    #  source-plane side.  z_source is handled upstream.
    # ═══════════════════════════════════════════════════════════════

    function _render_source(lp::LightPlane, x, y; kwargs...)
        return _render_source(lp.light, x, y; kwargs...)
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
    #  USAGE (shared kwargs, backward compat):
    #      cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0., 0.)
    #      ml = MultiLensedPlane(
    #          ((CombinedLens(SIS=>(b=0.5, ...)), 0.3),
    #           (NIEkappa, 0.8));
    #          z_source   = 1.5,
    #          cosmology  = cosmo,
    #      )
    #      bx, by = LB.LensPlane(xg, yg; LensModel=ml,
    #          LensKwargs=Dict(:b=>0.6, :s=>0.1, :q=>0.7, :varphi=>0.3))
    #
    #  USAGE (per-plane kwargs):
    #      ml = MultiLensedPlane(
    #          ((CombinedLens(SIS=>(b=0.5, ...)), 0.3, (;)),        # plane 1: no extra kwargs
    #           (NIEkappa, 0.8, (b=0.6, s=0.1, q=0.7, varphi=0.3))); # plane 2: own kwargs baked in
    #          z_source   = 1.5,
    #          cosmology  = cosmo,
    #      )
    #      bx, by = LB.LensPlane(xg, yg; LensModel=ml, LensKwargs=Dict())
    # ==============================================================
    struct MultiLensedPlane{P<:Tuple, C<:Cosmology.AbstractCosmology} <: AbstractLens
        planes::P          # each element: (lens, z, kwargs::NamedTuple)
        z_source::Float64
        cosmology::C
    end

    # Keyword constructor: normalizes (lens, z) → (lens, z, NamedTuple())
    function MultiLensedPlane(planes::Tuple; z_source::Float64,
                              cosmology::Cosmology.AbstractCosmology)
        # Normalize each plane to (lens, z, kwargs) 3-tuple
        _normalized = Tuple(
            length(p) == 2 ? (p[1], p[2], NamedTuple()) : p
            for p in planes
        )
        return MultiLensedPlane(_normalized, z_source, cosmology)
    end

    # ── Internal: extract per-plane kwargs, merge with shared kwargs ──
    @inline _plane_kwargs(plane::Tuple, shared_kwargs) =
        merge(NamedTuple(shared_kwargs), plane[3])

    # ── Internal: shared ray-tracing ──────────────────────────
    function _trace_rays!(aphys_x, aphys_y, thetax, thetay,
                          z_planes, ml::MultiLensedPlane, x, y; kwargs...)
        N = length(ml.planes)
        # Plane 1: evaluated at image position theta
        thetax[1] = x
        thetay[1] = y
        kw1 = _plane_kwargs(ml.planes[1], kwargs)
        aphys_x[1], aphys_y[1] = lens_derivative(ml.planes[1][1], x, y; kw1...)
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
            kwi = _plane_kwargs(ml.planes[i], kwargs)
            aphys_x[i], aphys_y[i] = lens_derivative(
                ml.planes[i][1], tx, ty; kwi...)
        end
        return nothing
    end

    function _alloc_ray_buffers(ml::MultiLensedPlane, x, y)
        N = length(ml.planes)
        z_planes = Float64[z for (_, z, _) in ml.planes]
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
            kwi = _plane_kwargs(ml.planes[i], kwargs)
            Hxx[i], Hxy[i], Hyy[i] = lens_hessian(
                ml.planes[i][1], thetax[i], thetay[i]; kwi...)
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
            kwi = _plane_kwargs(ml.planes[i], kwargs)
            ratio = lens_distance_ratio(ml.cosmology, z_planes[i], ml.z_source)
            psi .+= ratio .* lens_potential(ml.planes[i][1], thetax[i], thetay[i]; kwi...)
        end
        return psi
    end

    function lens_check(ml::MultiLensedPlane; kwargs...)
        for (lens_model, _, _) in ml.planes
            lens_check(lens_model; kwargs...)
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  Grid — observation grid descriptor
    #
    #      grid = GenGrid(pix_n=256, pix_size=0.09)
    #      xg, yg = grid.xg, grid.yg
    #
    #  GenGrid constructs via LensUtils.LensGrid; Grid stores the
    #  metadata (pix_n, pix_size) alongside the coordinate arrays.
    # ═══════════════════════════════════════════════════════════════

    struct Grid
        pix_n::Int
        pix_size::Float64
        xg::Matrix{Float64}
        yg::Matrix{Float64}
    end

    function GenGrid(; pix_n::Int=256, pix_size::Float64=0.09)
        half = div(pix_n, 2) * pix_size
        xg, yg = LensGrid(; xl=half, nx=pix_n + 1)
        return Grid(pix_n, pix_size, xg, yg)
    end


    # ═══════════════════════════════════════════════════════════════
    #  LensInstance
    # ═══════════════════════════════════════════════════════════════

    struct LensInstance
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
    #=
    struct SourceInstance
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
    =#

    # ═══════════════════════════════════════════════════════════════
    #  Unified Render Pipeline (AbstractLight)
    #
    #   render_lens(src, grid, lens_model; psf, ...)
    #
    #  PointImage   → solve_images + render_point! (sub-pixel PSF)
    #  ExtendedSource → ray-trace + evaluate_source (+ optional PSF)
    #  CompositeImage  → sum of components
    #
    #  add_point is kept as a backward-compat convenience wrapper.
    # ═══════════════════════════════════════════════════════════════


    # ────  PointImage ────────────────────────────────────────────

    """
        img = render_lens(pt::PointImage, grid::Grid, lens_model;
                          psf=GaussianPSF(fwhm=0.052), half=7,
                          method=:supersample, n_sub=5)

    Render a point source through `lens_model`.  Pipeline:

    1. `LensSolver.solve_images` → image positions + magnification
    2. `LensPSF.render_point!`     → sub-pixel PSF placement per image

    The output grid and pixel scale come from `grid`.
    """
    function render_lens(pt::PointImage, grid::Grid, lens_model;
                         psf            = LensPSF.GaussianPSF(fwhm=0.052),
                         half::Int      = 7,
                         method::Symbol  = :supersample,
                         n_sub::Int     = 5,
                         z_source       = nothing,
                         kwargs...)

        xg, yg = grid.xg, grid.yg
        pixel_scale = grid.pix_size
        ny, nx = size(xg)

        # Solve lens equation
        images = LensSolver.solve_images(lens_model, pt.beta_x, pt.beta_y; z_source=z_source)

        # Arcsec → pixel coordinate origin
        x_min = xg[1, 1]
        y_min = yg[1, 1]

        img = zeros(Float64, ny, nx)
        for (tx, ty, mu) in images
            F = pt.flux * abs(mu)
            px = (tx - x_min) / pixel_scale + 1.0
            py = (ty - y_min) / pixel_scale + 1.0
            LensPSF.render_point!(img, psf, px, py, F;
                                  pixel_scale, half, method, n_sub)
        end
        return img
    end


    # ────  ExtendedSource ────────────────────────────────────────

    """
        img = render_lens(src::ExtendedSource, grid::Grid, lens_model;
                          psf=nothing, kwargs...)

    Ray-trace the lens model and evaluate the source profile on
    the source-plane grid.

    If `psf` is provided, the result is convolved with the PSF
    (requires `conv_psf` to be defined for that PSF type).
    Otherwise the unconvolved source-plane flux is returned.
    """
    function render_lens(src::ExtendedSource, grid::Grid,
                         lens_model; psf=nothing, z_source=nothing, kwargs...)

        xg, yg = grid.xg, grid.yg
        pixel_scale = grid.pix_size

        # Ray-trace: beta = theta - alpha(theta)
        alphax, alphay = lens_derivative(lens_model, xg, yg; z_source=z_source, kwargs...)
        betax = xg .- alphax
        betay = yg .- alphay

        result = evaluate_source(src, betax, betay)

        if psf !== nothing
            result = LensPSF.conv_psf(result, psf, pixel_scale)
        end
        return result
    end


    # ────  CompositeImage ────────────────────────────────────────

    """
        img = render_lens(src::CompositeImage, grid::Grid, lens_model;
                          kwargs...)

    Sum the `render_lens` output of each component.  Keyword arguments
    (psf, half, method, n_sub, etc.) are forwarded to each component.
    """
    function render_lens(src::CompositeImage, grid::Grid,
                         lens_model; kwargs...)

        xg = grid.xg
        ny, nx = size(xg)
        result = zeros(Float64, ny, nx)

        for component in src.sources
            result .+= render_lens(component, grid, lens_model; kwargs...)
        end
        return result
    end


    # ────  Backward-compat: add_point ────────────────────────────

    """
        img = add_point(grid, lens_model, flux, beta_x, beta_y; kwargs...)

    Deprecated convenience wrapper.  Equivalent to:

        render_lens(PointImage(flux=flux, beta_x=beta_x, beta_y=beta_y),
                    grid, lens_model; kwargs...)

    Prefer `render_lens(point_image, grid, lens; ...)` in new code.
    """
    function add_point(grid::Grid, lens_model,
                       flux::Real, beta_x::Real, beta_y::Real;
                       z_source=nothing, kwargs...)
        pt = PointImage(flux=Float64(flux),
                        beta_x=Float64(beta_x),
                        beta_y=Float64(beta_y))
        return render_lens(pt, grid, lens_model; z_source=z_source, kwargs...)
    end

end
