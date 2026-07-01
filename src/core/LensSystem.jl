module LensSystem

    # ═══════════════════════════════════════════════════════════════
    #  LensSystem — complete forward-model container
    #
    #  Bundles lens plane + source plane + PSF + grid into a single
    #  struct.  `render(sys)` runs the full pipeline:
    #
    #     1. lens equation  β = θ − α(θ)
    #     2. source-plane evaluation
    #     3. PSF convolution
    #
    #  All fields are GPU-compatible: struct passes directly into
    #  CUDA kernels without scalar indexing or heap allocations.
    # ═══════════════════════════════════════════════════════════════

    using Cosmology
    import Jens.LensBase: AbstractLens, lens_derivative, lens_hessian,
                           lens_potential, lens_check
    using Jens.LensPSF: AbstractPSF, conv_psf, render_point!
    using Jens.LightModel: AbstractLight, ExtendedSource, PointImage, PointImages,
                           CompositeImage, evaluate_source
    using Jens.LensGenerator: LensedPlane, MultiLensedPlane,
                              LightPlane, MultiLightPlane
    using Jens.LensSolver: solve_images, batch_solve_images

    export ForwardModel, render

    # ═══════════════════════════════════════════════════════════════
    #  Struct
    # ═══════════════════════════════════════════════════════════════

    """
        LensSystem(; lens_plane, source_plane, grid, psf=nothing)

    Complete forward-model container for gravitational lensing.

    # Arguments
    - `lens_plane`   : `LensedPlane` or `MultiLensedPlane` — mass model
    - `source_plane` : `LightPlane` or `MultiLightPlane` — light model
    - `grid`         : `Grid` or `GridGPU` — observation grid
    - `psf`          : `AbstractPSF` or `nothing` — instrumental blur

    # Example
    ```julia
    cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
    lens  = CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0))
    src   = ExtendedSource((x,y)->exp.(-(x.^2 .+ y.^2)./0.1^2), (;))

    sys = ForwardModel(
        lens_plane   = LensedPlane(lens; z_lens=0.3, cosmology=cosmo),
        source_plane = LightPlane(src; z=1.5),
        grid         = GenGrid(pix_n=256, pix_size=0.09),
        psf          = GaussianPSF(0.05),
    )
    img = render(sys)
    ```
    """
    struct ForwardModel{L, S, P, G} <: AbstractLens
        lens_plane::L
        source_plane::S
        psf::P
        grid::G
    end

    function ForwardModel(; lens_plane, source_plane, grid, psf=nothing, z_source=nothing)
        # Auto-wrap bare AbstractLight → LightPlane when z_source is given
        if source_plane isa AbstractLight
            z_source !== nothing || error(
                "source_plane is a bare $(typeof(source_plane)); " *
                "pass z_source=<redshift> to wrap it in LightPlane, " *
                "or wrap manually: LightPlane(src; z=...)"
            )
            source_plane = LightPlane(source_plane; z=z_source)
        end
        return ForwardModel(lens_plane, source_plane, psf, grid)
    end

    # ── AbstractLens interface: forward to lens_plane ──

    lens_derivative(sys::ForwardModel, x, y; z_source=nothing, kwargs...) =
        lens_derivative(sys.lens_plane, x, y;
                         z_source=something(z_source, _source_z(sys.source_plane)),
                         kwargs...)

    lens_hessian(sys::ForwardModel, x, y; z_source=nothing, kwargs...) =
        lens_hessian(sys.lens_plane, x, y;
                      z_source=something(z_source, _source_z(sys.source_plane)),
                      kwargs...)

    lens_potential(sys::ForwardModel, x, y; z_source=nothing, kwargs...) =
        lens_potential(sys.lens_plane, x, y;
                        z_source=something(z_source, _source_z(sys.source_plane)),
                        kwargs...)

    lens_check(sys::ForwardModel; kwargs...) =
        lens_check(sys.lens_plane; kwargs...)

    # ── Extract redshift from source plane ──

    _source_z(lp::LightPlane) = lp.z
    _source_z(mlp::MultiLightPlane) = mlp.planes[1][2]


    # ═══════════════════════════════════════════════════════════════
    #  render — main entry point
    # ═══════════════════════════════════════════════════════════════

    """
        img = render(sys::ForwardModel; solver=:nlsolve)

    Full forward render. Dispatches internally on source type:
    - ExtendedSource  → ray-trace + evaluate_source
    - PointImage      → solve_images + render_point!
    - CompositeImage  → recursive sum of components

    # Point-source solver
    - `solver=:nlsolve`  → serial NLsolve (default, robust)
    - `solver=:batch`    → batched Newton (GPU-native, 50–70× faster)
    """
    function render(sys::ForwardModel; solver::Symbol=:nlsolve)
        return _render(sys, sys.source_plane; solver=solver)
    end


    # ═══════════════════════════════════════════════════════════════
    #  Source-plane dispatchers
    # ═══════════════════════════════════════════════════════════════

    # ── LightPlane: single redshift, single light model ──

    function _render(sys::ForwardModel, lp::LightPlane; solver::Symbol=:nlsolve)
        return _render_light(sys, lp.light, lp.z; solver=solver)
    end

    # ── MultiLightPlane: multiple light components at different z ──
    #     _render_light handles PSF per light type:
    #       ExtendedSource → conv_psf      PointImage → render_point!
    #     Convolution is linear, so per-plane PSF = single PSF after sum.

    function _render(sys::ForwardModel, mlp::MultiLightPlane; solver::Symbol=:nlsolve)
        result = nothing
        for (light, z) in mlp.planes
            c = _render_light(sys, light, z; solver=solver)
            result = result === nothing ? c : result .+ c
        end
        return result
    end


    # ═══════════════════════════════════════════════════════════════
    #  Light-type dispatchers  (PSF handled per type: conv_psf or render_point!)
    # ═══════════════════════════════════════════════════════════════

    # ── ExtendedSource: ray-trace then evaluate ──

    function _render_light(sys::ForwardModel, src::ExtendedSource, z_src; solver::Symbol=:nlsolve)
        xg, yg = sys.grid.xg, sys.grid.yg
        ax, ay = lens_derivative(sys.lens_plane, xg, yg; z_source=z_src)
        betax = xg .- ax
        betay = yg .- ay
        result = evaluate_source(src, betax, betay)
        return _apply_psf(sys, result)
    end

    # ── PointImage: solve lens equation, render point spread ──

    function _render_light(sys::ForwardModel, pt::PointImage, z_src; solver::Symbol=:nlsolve)
        images = if solver == :batch
            batch_solve_images(sys.lens_plane, pt.beta_x, pt.beta_y; z_source=z_src)
        else
            solve_images(sys.lens_plane, pt.beta_x, pt.beta_y; z_source=z_src)
        end

        xg = sys.grid.xg
        T = eltype(xg)

        # Allocate result matching grid type (GPU or CPU)
        result = fill!(similar(xg), zero(T))

        # render_point! does scalar indexing → render on CPU buffer
        nx, ny = size(xg)
        buf = zeros(T, nx, ny)  # always CPU

        # Compute grid origin from metadata (avoids GPU scalar indexing)
        half = div(sys.grid.pix_n, 2) * Float64(sys.grid.pix_size)
        x_min = -half
        y_min = -half
        pixel_scale = sys.grid.pix_size

        for (tx, ty, mu) in images
            F = pt.flux * abs(mu)
            px = (tx - x_min) / pixel_scale + 1
            py = (ty - y_min) / pixel_scale + 1
            if sys.psf !== nothing
                render_point!(buf, sys.psf, px, py, F;
                              pixel_scale=pixel_scale, half=7)
            else
                ix = round(Int, px)
                iy = round(Int, py)
                if 1 <= ix <= size(buf, 2) && 1 <= iy <= size(buf, 1)
                    buf[iy, ix] += F
                end
            end
        end
        copyto!(result, buf)  # CPU→CPU no-op, CPU→GPU transfer
        return result
    end

    # ── PointImages: pre-solved positions, no lens equation ──
        #
        #  intrinsic=false: amp is observed flux (magnification already included)
        #  intrinsic=true:  amp is intrinsic flux → μ computed from lens_hessian

        function _render_light(sys::ForwardModel, pi::PointImages{<:Real}, z_src; solver=nothing)
            xg = sys.grid.xg
            T = eltype(xg)

            result = fill!(similar(xg), zero(T))
            nx, ny = size(xg)
            buf = zeros(T, nx, ny)
            half = div(sys.grid.pix_n, 2) * Float64(sys.grid.pix_size)
            x_min, y_min = -half, -half
            pixel_scale = sys.grid.pix_size

            for (amp, positions) in pi.components
                # ── compute per-image amplitudes ──
                per_image_amps = if pi.intrinsic
                    mus = _compute_magnifications(sys, positions, z_src)
                    amp .* mus
                else
                    fill(amp, length(positions))
                end

                for (i, (tx, ty)) in enumerate(positions)
                    px = (tx - x_min) / pixel_scale + 1
                    py = (ty - x_min) / pixel_scale + 1
                    if sys.psf !== nothing
                        render_point!(buf, sys.psf, px, py, per_image_amps[i]; pixel_scale=pixel_scale, half=7)
                    else
                        ix, iy = round(Int, px), round(Int, py)
                        if 1 <= ix <= nx && 1 <= iy <= ny
                            buf[iy, ix] += per_image_amps[i]
                        end
                    end
                end
            end
            copyto!(result, buf)
            return result
        end

        # ── Magnification helper for PointImages(intrinsic=true) ──
        #
        #  Computes μ = 1/|det(A)| at each (x, y) via lens_hessian.
        #  For N ≤ ~10 points this is ~5 μs → negligible vs render_point!.
        function _compute_magnifications(sys::ForwardModel, positions, z_src)
            mus = Real[]
            for (x, y) in positions
                fxx, fxy, fyy = lens_hessian(sys, [x], [y]; z_source=z_src)
                detA = (1 - fxx[1]) * (1 - fyy[1]) - fxy[1]^2
                push!(mus, 1 / abs(detA))
            end
            return mus
        end

    # ── CompositeImage: recursive sum of components ──

    function _render_light(sys::ForwardModel, comp::CompositeImage, z_src; solver::Symbol=:nlsolve)
        result = nothing
        for component in comp.sources
            c = _render_light(sys, component, z_src; solver=solver)
            result = result === nothing ? c : result .+ c
        end
        return result
    end


    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    @inline _apply_psf(sys::ForwardModel, result) = sys.psf === nothing ?
        result : conv_psf(result, sys.psf, sys.grid.pix_size)

end # module LensSystem