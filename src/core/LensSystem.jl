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
                           lens_potential, lens_check, LensFermat
    using Jens.LensPSF: AbstractPSF, conv_psf, render_point!
    using Jens.LightModel: AbstractLight, ExtendedSource, PointImage, PointImages,
                           CompositeImage, evaluate_source
    using Jens.LensGenerator: LensedPlane, MultiLensedPlane,
                              LightPlane, MultiLightPlane
    using Jens.LensSolver: solve_images, batch_solve_images
    import Jens.LensCosmo: time_delay_distance

    export ForwardModel, render, LensTimeDelay

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
    struct ForwardModel{L, S, P, G, M} <: AbstractLens
        lens_plane::L
        source_plane::S
        psf::P
        grid::G
        mask::M       # AbstractMatrix{Bool} or Nothing
    end

    """
        ForwardModel(; lens_plane, source_plane, grid, psf=nothing, mask=nothing, z_source=nothing)

    Complete forward-model container for gravitational lensing.

    # Arguments
    - `lens_plane`   : `LensedPlane` or `MultiLensedPlane` — mass model
    - `source_plane` : `LightPlane` or `MultiLightPlane` — light model
    - `grid`         : `Grid` or `GridGPU` — observation grid
    - `psf`          : `AbstractPSF` or `nothing` — instrumental blur
    - `mask`         : `AbstractMatrix{Bool}` or `nothing` — pixel mask.
      `true` = include in fit.  Stored in system; used automatically
      by `masked_logp(sys, data, σ)`.  Default: `nothing` (all pixels).

    # Example
    ```julia
    mask = combine(annular_mask(grid, 0.5, 2.0),
                   invert(circular_mask(grid, 0.2)); op=&)

    sys = ForwardModel(
        lens_plane   = LensedPlane(lens; z_lens=0.3, cosmology=cosmo),
        source_plane = LightPlane(src; z=1.5),
        grid         = GenGrid(pix_n=256, pix_size=0.09),
        psf          = GaussianPSF(0.05),
        mask         = mask,   # ← mask lives in the system
    )
    img = render(sys)               # full image (mask doesn't affect physics)
    logp = masked_logp(sys, data, σ)  # mask read automatically
    ```
    """
    function ForwardModel(; lens_plane, source_plane, grid,
                           psf=nothing, mask=nothing, z_source=nothing)
        # Auto-wrap bare AbstractLight → LightPlane when z_source is given
        if source_plane isa AbstractLight
            z_source !== nothing || error(
                "source_plane is a bare $(typeof(source_plane)); " *
                "pass z_source=<redshift> to wrap it in LightPlane, " *
                "or wrap manually: LightPlane(src; z=...)"
            )
            source_plane = LightPlane(source_plane; z=z_source)
        end
        return ForwardModel(lens_plane, source_plane, psf, grid, mask)
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

    # ── Extract source position from source plane ──

    _source_beta(lp::LightPlane) = lp.light isa PointImage ?
        Float64[lp.light.beta_x, lp.light.beta_y] : Float64[0., 0.]
    _source_beta(::MultiLightPlane) = Float64[0., 0.]

    # ── Extract cosmology info from lens plane ──

    _get_cosmology(lp::LensedPlane) = lp.cosmology
    _get_cosmology(ml::MultiLensedPlane) = ml.cosmology

    _get_z_lens(lp::LensedPlane) = lp.z_lens
    _get_z_lens(ml::MultiLensedPlane) = ml.planes[1][2]

    # MultiLensedPlane carries own z_source; LensedPlane does not
    _maybe_z_source(::LensedPlane) = nothing
    _maybe_z_source(ml::MultiLensedPlane) = ml.z_source


    # ═══════════════════════════════════════════════════════════════
    #  LensFermat — ForwardModel dispatch
    # ═══════════════════════════════════════════════════════════════

    """
        tau = LensFermat(sys::ForwardModel; beta=nothing, z_source=nothing, kwargs...)

    Compute the angular Fermat potential for a complete lens system.
    `z_source` and `beta` are auto-detected when the source plane
    provides them (e.g. `PointImage`).
    """
    function LensFermat(sys::ForwardModel; beta=nothing,
                                       z_source=nothing, kwargs...)
        zs = something(z_source, _source_z(sys.source_plane))
        bx = something(beta, _source_beta(sys.source_plane))
        xg, yg = sys.grid.xg, sys.grid.yg
        return LensFermat(xg, yg, bx;
                          LensModel=sys.lens_plane,
                          z_source=zs, kwargs...)
    end


    # ═══════════════════════════════════════════════════════════════
    #  time_delay_distance — ForwardModel dispatch
    # ═══════════════════════════════════════════════════════════════

    """
        D_dt = time_delay_distance(sys::ForwardModel) → Float64

    Auto-extract cosmology and redshifts from `sys.lens_plane`
    and `sys.source_plane`. Returns the time-delay distance in Mpc.
    """
    function time_delay_distance(sys::ForwardModel)
        cosmo = _get_cosmology(sys.lens_plane)
        z_lens = _get_z_lens(sys.lens_plane)
        zs = _source_z(sys.source_plane)
        return time_delay_distance(cosmo, z_lens, zs)
    end


    # ═══════════════════════════════════════════════════════════════
    #  LensTimeDelay — time-delay surface and image-pair delay
    # ═══════════════════════════════════════════════════════════════

    const _ARCSEC2_TO_RAD2 = (π / 180 / 3600)^2
    const _C_LIGHT_MPC_S   = 9.71561189025635e-15  # c in Mpc/s

    """
        Δt = LensTimeDelay(xg, yg, beta=[0.,0.];
                           LensModel, LensKwargs=NamedTuple(), z_source=nothing)

    Compute the time-delay surface (in **seconds**) on a grid.

        c · Δt(θ, β) = D_Δt  ·  τ(θ, β)

    where D_Δt is the time-delay distance and τ is the angular Fermat
    potential.  Requires `LensModel` to carry cosmology (`LensedPlane`,
    `MultiLensedPlane`, or `ForwardModel`).

    # Example
    ```julia
    # Single-plane via LensedPlane
    Δt_map = LensTimeDelay(xg, yg, [bx, by];
                           LensModel=lp, z_source=1.5)

    # Single-plane via ForwardModel (auto-detect everything)
    Δt_map = LensTimeDelay(sys)
    ```
    """
    function LensTimeDelay(xg::AbstractMatrix, yg::AbstractMatrix,
                            beta=[0.,0.]; LensModel,
                            LensKwargs=NamedTuple(), z_source=nothing)
        cosmo = _get_cosmology(LensModel)
        z_lens = _get_z_lens(LensModel)
        zs = something(z_source, _maybe_z_source(LensModel))

        D_dt = time_delay_distance(cosmo, z_lens, zs)
        tau  = LensFermat(xg, yg, beta;
                          LensModel=LensModel, LensKwargs=LensKwargs, z_source=zs)

        tau_rad2 = tau .* _ARCSEC2_TO_RAD2    # arcsec² → rad²
        return D_dt .* tau_rad2 ./ _C_LIGHT_MPC_S   # seconds
    end

    """
        Δt = LensTimeDelay(sys::ForwardModel; beta=nothing, z_source=nothing, kwargs...)

    Compute the time-delay surface for a complete lens system.
    All parameters are auto-detected from the system.

    # Example
    ```julia
    sys = ForwardModel(lens_plane=lp, source_plane=LightPlane(agn; z=1.5), grid=grid)
    Δt_map = LensTimeDelay(sys)   # everything auto-detected
    ```
    """
    function LensTimeDelay(sys::ForwardModel; beta=nothing,
                            z_source=nothing, kwargs...)
        zs = something(z_source, _source_z(sys.source_plane))
        bx = something(beta, _source_beta(sys.source_plane))
        xg, yg = sys.grid.xg, sys.grid.yg
        return LensTimeDelay(xg, yg, bx;
                             LensModel=sys.lens_plane, z_source=zs, kwargs...)
    end


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
        buf = zeros(T, nx, ny)  # always CPU: render_point! uses scalar indexing

        # Compute grid origin from metadata (avoids GPU scalar indexing)
        half = div(sys.grid.pix_n, 2) * Float64(sys.grid.pix_size)
        x_min = -half
        y_min = -half
        pixel_scale = sys.grid.pix_size

        for (tx, ty, mu) in images
            F = pt.flux * abs(mu)
            # Grid convention (ndgrid): rows=X, cols=Y
            # pixel_to_row  ← tx (X coordinate maps to row)
            # pixel_to_col  ← ty (Y coordinate maps to column)
            pix_row = (tx - x_min) / pixel_scale + 1
            pix_col = (ty - y_min) / pixel_scale + 1
            if sys.psf !== nothing
                # render_point! takes (x_src=column, y_src=row)
                # and writes to image[row, col], so:
                #   x_src ← pix_col (Y → column)
                #   y_src ← pix_row (X → row)
                render_point!(buf, sys.psf, pix_col, pix_row, F;
                              pixel_scale=pixel_scale, half=7)
            else
                ir = round(Int, pix_row)
                ic = round(Int, pix_col)
                if 1 <= ir <= size(buf, 1) && 1 <= ic <= size(buf, 2)
                    buf[ir, ic] += F
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
            buf = zeros(T, nx, ny)  # always CPU: render_point! uses scalar indexing
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
                    # Grid convention (ndgrid): rows=X, cols=Y
                    pix_row = (tx - x_min) / pixel_scale + 1
                    pix_col = (ty - y_min) / pixel_scale + 1
                    if sys.psf !== nothing
                        # render_point! takes (x_src=column, y_src=row)
                        # so: x_src ← pix_col, y_src ← pix_row
                        render_point!(buf, sys.psf, pix_col, pix_row, per_image_amps[i]; pixel_scale=pixel_scale, half=7)
                    else
                        ir, ic = round(Int, pix_row), round(Int, pix_col)
                        if 1 <= ir <= nx && 1 <= ic <= ny
                            buf[ir, ic] += per_image_amps[i]
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
    #  masked_logp / masked_chi2 — render → compare → scalar
    # ═══════════════════════════════════════════════════════════════

    """
        chi2 = masked_chi2(sys::ForwardModel, data, σ²::Real, mask)
        logp = masked_logp(sys::ForwardModel, data, σ::Real, mask)

    Full pipeline in one call: render → residual → masked χ² → logp.

    GPU-safe: all operations are pure broadcast, zero scalar indexing
    when `data`, `mask` are on the same device as the grid.

    # Arguments
    - `sys`:  `ForwardModel` — lens, source, PSF, grid
    - `data`: observed image (same shape as `render(sys)`)
    - `σ²` or `σ`: noise variance / standard deviation
    - `mask`: `AbstractMatrix{Bool}` — `true` = include pixel.
      Pass `nothing` to use all pixels.

    # Returns
    - `masked_chi2` → Σᵢ (dataᵢ − modelᵢ)² · maskᵢ / σ²
    - `masked_logp`  → −½ · masked_chi2

    # Example
    ```julia
    mask = combine(annular_mask(grid, 0.5, 2.0),
                   invert(circular_mask(grid, 0.2)); op=&)

    function my_logp(params)
        sys = build_system(params)
        return masked_logp(sys, data, 0.03, mask)  # one line
    end

    chain = lens_mh(my_logp, lower, upper; n=2000)
    ```
    """
    function masked_chi2(sys::ForwardModel, data, σ²::Real, mask)
        model = render(sys)
        diff² = (data .- model).^2
        T = eltype(data)
        return sum(diff² .* mask) / T(σ²)
    end

    function masked_chi2(sys::ForwardModel, data, σ²::Real, ::Nothing)
        model = render(sys)
        diff² = (data .- model).^2
        return sum(diff²) / eltype(data)(σ²)
    end

    # ── One-argument-less: read mask from sys.mask ──
    function masked_chi2(sys::ForwardModel, data, σ²::Real)
        return masked_chi2(sys, data, σ², sys.mask)
    end

    function masked_logp(sys::ForwardModel, data, σ::Real, mask)
        return -masked_chi2(sys, data, σ^2, mask) / 2
    end

    function masked_logp(sys::ForwardModel, data, σ::Real)
        return masked_logp(sys, data, σ, sys.mask)
    end

    export masked_chi2, masked_logp

    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    @inline _apply_psf(sys::ForwardModel, result) = sys.psf === nothing ?
        result : conv_psf(result, sys.psf, sys.grid.pix_size)

end # module LensSystem