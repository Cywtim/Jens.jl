module LensPlots

    import Plots
    using Plots 
    export Plots           # re-export the Plots module so `using LensPlots` gives full Plots access

    # ═══════════════════════════════════════════════════════════════
    #  LensPlots — composable lensing visualisation for Jens.jl
    #
    #  Three building blocks:
    #    1.  LensCanvas  → 创建画布
    #    2.  PlotPlane!  → 在画布上画 heatmap
    #    3.  PlotPoints! → 在画布上画 scatter（接受 AbstractArray）
    #
    #  Higher-level convenience:
    #    PlotLens           → 一行出图
    #    PlotFermat         → Fermat 势面
    #    PlotMagnification  → 放大率图
    #    PlotCriticalCurve! → 计算+叠加临界曲线
    #    PlotCaustic!       → 计算+叠加焦散线
    # ═══════════════════════════════════════════════════════════════

    # ═══════════════════════════════════════════════════════════════
    #  ORIGINAL v1 (2026-05-21 之前 — 保留作为参照)
    # ═══════════════════════════════════════════════════════════════
    #=
    module LensPlots

        using Plots

        export PlotPlane, PlotPoints!

        function PlotPlane(
                xg::AbstractArray, yg::AbstractArray,
                image::AbstractArray;
                figsize::Tuple{Int,Int} = (6, 6),
                kwargs...
            )
            xvec = xg[:, 1]
            yvec = yg[1, :]
            return heatmap(xvec, yvec, image';
                           size = (figsize[1] * 100, figsize[2] * 100),
                           xlabel = "x (arcsec)", ylabel = "y (arcsec)",
                           kwargs...)
        end

        function PlotPoints!(
                x::AbstractVector, y::AbstractVector;
                label::String = "",
                color          = :red,
                ms::Int        = 2,
                kwargs...
            )
            return scatter!(x, y;
                            label = label, color = color,
                            markersize = ms,
                            markerstrokewidth = 0,
                            kwargs...)
        end

    end
    =#

export LensCanvas
    export PlotPlane!, PlotPoints!
    export PlotCriticalCurve!, PlotCaustic!
    export PlotFermat, PlotMagnification
    export PlotLens
    export PlotVec!
    export arrow          # re-export from Plots for PlotVec!
    # ═══════════════════════════════════════════════════════════════
    #  Create canvas
    # ═══════════════════════════════════════════════════════════════
    function LensCanvas(;
            xlims::NTuple{2,Real} = (-2.0, 2.0),
            ylims::NTuple{2,Real} = (-2.0, 2.0),
            figsize::NTuple{2,Int} = (6, 6),
            kwargs...
        )
            plot(;
                aspect_ratio = :equal,
                xlims = xlims,
                ylims = ylims,
                size = (figsize[1] * 100, figsize[2] * 100),
                xlabel = "x (pix)",
                ylabel = "y (pix)",
                kwargs...
            )
    end

    # ═══════════════════════════════════════════════════════════════
    #  2.   Draw on canvas
    # ═══════════════════════════════════════════════════════════════

    _scale_label(scale::Symbol) = scale == :linear ? "" : " ($(scale))"

    function _apply_scale(image::AbstractArray, scale::Symbol)
        if scale == :linear
            return image
        elseif scale == :log
            min_val = minimum(image)
            offset  = max(0.0f0, -min_val) + 1f-30
            return log10.(image .+ offset)
        elseif scale == :sqrt
            min_val = minimum(image)
            offset  = max(0.0f0, -min_val)
            return sqrt.(max.(image .+ offset, 0.0f0))
        else
            error("Unknown scale=:$scale. Use :linear, :log, or :sqrt.")
        end
    end

    """
        PlotPlane!(canvas, xg, yg, image; scale, colormap, clim, colorbar, kwargs...)

    Draw a lens-plane image as a heatmap onto an existing `canvas`.

    # Keyword arguments
    - `scale::Symbol = :linear`  — `:linear`, `:log` (log₁₀), or `:sqrt`
    """
    function PlotPlane!(
            canvas,
            xg::AbstractArray, yg::AbstractArray,
            image::AbstractArray;
            scale::Symbol  = :linear,
            colormap = :dense,
            clim::Union{Tuple,Nothing}  = nothing,
            colorbar::Bool = true,
            kwargs...
        )
        xvec = xg[:, 1]
        yvec = yg[1, :]

        display_image = _apply_scale(image, scale)'
        clim_val      = clim === nothing ? extrema(display_image) : clim

        colorbar_title = colorbar ? _scale_label(scale) : ""

        heatmap!(canvas, xvec, yvec, display_image;
                 c               = colormap,
                 clim            = clim_val,
                 colorbar        = colorbar,
                 colorbar_title  = colorbar_title,
                 kwargs...)
    end

    """
        PlotPlane!(canvas, image; scale, pixel_scale, x0, y0, colormap, clim, colorbar, kwargs...)

    Draw an image matrix directly onto `canvas`, using pixel indices
    scaled by `pixel_scale` and offset by `(x0, y0)`.

    No grid arrays required — useful for pre-computed images, PSF-convolved
    outputs, or any matrix that already lives on a regular pixel grid.

    # Keyword arguments
    - `scale::Symbol = :linear`  — `:linear`, `:log` (log₁₀), or `:sqrt`

    **Example**
        # 51×51 image at WFC3 pixel scale, centred at (0,0)
        PlotPlane!(canvas, my_image; pixel_scale=0.04)

        # Pixel indices as coordinates (pixel_scale=1, origin at centre)
        PlotPlane!(canvas, my_image)

        # Log-scale to reveal faint structure
        PlotPlane!(canvas, my_image; scale=:log, pixel_scale=0.04)
    """
    function PlotPlane!(
            canvas,
            image::AbstractMatrix;
            scale::Symbol  = :linear,
            pixel_scale::Real = 1.0,
            x0::Real = 0.0,
            y0::Real = 0.0,
            colormap = :dense,
            clim::Union{Tuple,Nothing}  = nothing,
            colorbar::Bool = true,
            kwargs...
        )
        ny, nx = size(image)
        half_x = (nx - 1) / 2 * pixel_scale
        half_y = (ny - 1) / 2 * pixel_scale
        xvec = range(x0 - half_x, x0 + half_x; length=nx)
        yvec = range(y0 - half_y, y0 + half_y; length=ny)

        display_image = _apply_scale(image, scale)
        clim_val      = clim === nothing ? extrema(display_image) : clim

        colorbar_title = colorbar ? _scale_label(scale) : ""

        heatmap!(canvas, collect(xvec), collect(yvec), display_image;
                 c               = colormap,
                 clim            = clim_val,
                 colorbar        = colorbar,
                 colorbar_title  = colorbar_title,
                 kwargs...)
    end

    # ═══════════════════════════════════════════════════════════════
    #  3.  BUILDING BLOCK — DRAW SCATTER ON CANVAS
    #
    #      Accepts two calling conventions:
    #        (A) PlotPoints!(canvas, x, y; kwargs...)
    #            x, y are AbstractArrays (vectors, matrices, etc.)
    #        (B) PlotPoints!(canvas, points; kwargs...)
    #            points is an N×2 AbstractMatrix (col 1 = x, col 2 = y)
    # ═══════════════════════════════════════════════════════════════

    """
        PlotPoints!(canvas, x, y; label, color, ms, marker, kwargs...)
        PlotPoints!(canvas, points; label, color, ms, marker, kwargs...)

    Overlay scatter points onto an existing `canvas`.

    **Calling conventions**
    - `PlotPoints!(canvas, x, y)`  — `x`, `y` are `AbstractArray`
    - `PlotPoints!(canvas, points)` — N×2 matrix (col 1 = x, col 2 = y)

    **Keyword arguments**
    - `label`  → legend label         (default `""`)
    - `color`  → marker colour        (default `:red`)
    - `ms`     → marker size          (default `2`)
    - `marker` → marker shape         (default `:circle`)
    """
    # ── method A: separate x, y (any AbstractArray) ──
    function PlotPoints!(
            canvas,
            x::AbstractArray, y::AbstractArray;
            label::String   = "",
            color           = :red,
            ms::Int         = 2,
            marker::Symbol  = :circle,
            kwargs...
        )
        scatter!(canvas, x, y;
                 label              = label,
                 color              = color,
                 markersize         = ms,
                 marker             = marker,
                 markerstrokewidth  = 0,
                 kwargs...)
    end

    # ── method B: N×2 matrix → split into (x, y) ──
    function PlotPoints!(
            canvas,
            points::AbstractMatrix;
            label::String   = "",
            color = :red,
            ms::Int = 2,
            marker::Symbol  = :circle,
            kwargs...
        )
        x = vec(points[:, 1])
        y = vec(points[:, 2])
        PlotPoints!(canvas, x, y; label, color, ms, marker, kwargs...)
    end

    # ═══════════════════════════════════════════════════════════════
    #  4.  BUILDING BLOCK — DRAW VECTOR ARROWS ON CANVAS
    # ═══════════════════════════════════════════════════════════════

    """
        PlotVec!(canvas, xg, yg, ax, ay; step, color, label, arrow, kwargs...)

    Overlay vector arrows onto `canvas`.  Each grid point `(xg, yg)` gets
    an arrow with components `(ax, ay)` — e.g. deflection angles or
    source-plane displacements.

    **Parameters**
    - `step::Int` — subsample every `step`-th point (default: auto, ~20 arrows per axis)
    - `color`    — arrow colour            (default `:black`)
    - `label`    — legend label            (default `""`)
    - `arrow`    — Plots.jl arrow style    (e.g. `arrow(:closed, 0.3)`)

    **Example**
        canvas = LensCanvas(; xlims=(-2,2), ylims=(-2,2))
        PlotPlane!(canvas, xg, yg, image)
        PlotVec!(canvas, xg, yg, alphax, alphay; step=5, color=:white)
    """
    function PlotVec!(
            canvas,
            xg::AbstractMatrix{<:Real}, yg::AbstractMatrix{<:Real},
            ax::AbstractMatrix{<:Real}, ay::AbstractMatrix{<:Real};
            step::Int = 0,
            color = :black,
            label::String = "",
            arrow = nothing,
            kwargs...
        )
        # Auto-scale step to get ~20 arrows per axis
        n = size(xg, 1)
        s = step > 0 ? step : max(1, div(n, 20))

        xs = vec(xg[1:s:end, 1:s:end])
        ys = vec(yg[1:s:end, 1:s:end])
        us = vec(ax[1:s:end, 1:s:end])
        vs = vec(ay[1:s:end, 1:s:end])

        arrow_kw = arrow !== nothing ? (; arrow) : NamedTuple()
        quiver!(canvas, xs, ys; quiver=(us, vs), color=color, label=label,
                arrow_kw..., kwargs...)
    end

    # ── 1D vector fallback ──
    function PlotVec!(
            canvas,
            xg::AbstractVector{<:Real}, yg::AbstractVector{<:Real},
            ax::AbstractVector{<:Real}, ay::AbstractVector{<:Real};
            color                         = :black,
            label::String                 = "",
            arrow                         = nothing,
            kwargs...
        )
        arrow_kw = arrow !== nothing ? (; arrow) : NamedTuple()
        quiver!(canvas, xg, yg; quiver=(ax, ay), color=color, label=label,
                arrow_kw..., kwargs...)
    end

    # ═══════════════════════════════════════════════════════════════
    #  5.  COMPUTE-AND-OVERLAY HELPERS
    # ═══════════════════════════════════════════════════════════════

    using ..LensBase
    using ..LensUtils

    """
        PlotCriticalCurve!(canvas; LensModel, LensKwargs, adaptive, kwargs...)

    Compute the critical curve (lens-plane det J⁻¹ = 0) and overlay
    it as scatter points on `canvas`.
    """
    function PlotCriticalCurve!(canvas;
            LensModel,
            LensKwargs::Dict,
            adaptive::Bool = true,
            color = :cyan,
            ms::Int = 2,
            label::String = "critical curve",
            kwargs...
        )
        if adaptive
            ccx, ccy = LensAdaptiveCriticalCurve(;
                LensModel, LensKwargs, kwargs...)
        else
            ccx, ccy = LensCriticalCurve(;
                LensModel, LensKwargs, kwargs...)
        end
        if isempty(ccx)
            @warn "PlotCriticalCurve!: no critical-curve points found"
            return
        end
        PlotPoints!(canvas, ccx, ccy; label, color, ms, marker=:circle)
    end

    """
        PlotCaustic!(canvas; LensModel, LensKwargs, adaptive, kwargs...)

    Compute and overlay the caustic (source-plane det J⁻¹ = 0) on `canvas`.
    """
    function PlotCaustic!(canvas;
            LensModel,
            LensKwargs::Dict,
            adaptive::Bool = true,
            color = :red,
            ms::Int = 2,
            label::String = "caustic",
            kwargs...
        )
        if adaptive
            csx, csy = LensAdaptiveCaustic(;
                LensModel, LensKwargs, kwargs...)
        else
            csx, csy = LensCaustic(;
                LensModel, LensKwargs, kwargs...)
        end
        if isempty(csx)
            @warn "PlotCaustic!: no caustic points found"
            return
        end
        PlotPoints!(canvas, csx, csy; label, color, ms, marker=:circle)
    end

    # ═══════════════════════════════════════════════════════════════
    #  6.  HIGH-LEVEL CONVENIENCE
    # ═══════════════════════════════════════════════════════════════

    """
        PlotFermat(canvas, xg, yg; beta, LensModel, LensKwargs, style, kwargs...)

    Plot the Fermat potential Phi(theta; beta) = 0.5|theta-beta|^2 - psi(theta) onto `canvas`.

    `style`: `:heatmap` | `:contour` | `:surface`.
    """
    function PlotFermat(
            canvas,
            xg::AbstractArray, yg::AbstractArray;
            beta::Vector{Float64}   = [0.0, 0.0],
            LensModel,
            LensKwargs::Dict,
            style::Symbol = :heatmap,
            colormap = :viridis,
            colorbar::Bool = true,
            kwargs...
        )
        phi = LensBase.LensFermat(xg, yg, beta;
                                  LensModel=LensModel, LensKwargs=LensKwargs)
        xvec = xg[:, 1]
        yvec = yg[1, :]

        if style == :heatmap
            heatmap!(canvas, xvec, yvec, phi';
                     c=colormap, colorbar=colorbar, kwargs...)
        elseif style == :contour
            contour!(canvas, xvec, yvec, phi';
                     c=colormap, colorbar=colorbar, fill=true, kwargs...)
        elseif style == :surface
            surface!(canvas, xvec, yvec, phi';
                     c=colormap, zlabel="Φ", kwargs...)
        else
            error("PlotFermat: unknown style=:$style.  Use :heatmap, :contour, or :surface")
        end
    end

    """
        PlotMagnification(canvas, xg, yg; LensModel, LensKwargs, kwargs...)

    Plot the magnification |µ| = |det J⁻¹| map onto `canvas`.
    """
    function PlotMagnification(
            canvas,
            xg::AbstractArray, yg::AbstractArray;
            LensModel,
            LensKwargs::Dict,
            log_scale::Bool = true,
            colormap = :RdBu,
            colorbar::Bool = true,
            kwargs...
        )
        magr = LensBase.LensDetJacobian(xg, yg;
                                        LensModel=LensModel,
                                        LensKwargs=LensKwargs)
        image = log_scale ? log10.(abs.(magr)) : magr
        PlotPlane!(canvas, xg, yg, image;
                   colormap, colorbar, kwargs...)
    end

    """
        PlotLens(; xl, nx, LensModel, LensKwargs, ...)

    **Full one-shot lens visualisation.**  Creates a canvas, generates
    the grid, ray-shoots (or computes magnification), and overlays
    critical curves and caustics.

    Returns the `Plots.Plot` object — use `Plots.savefig` to save.

    **Example**
        p = PlotLens(;
            xl=2.0, nx=256,
            LensModel=SIE,
            LensKwargs=Dict(:theta_E=>1.0, :e1=>0.25, :e2=>0.0))
        Plots.savefig(p, "lens.png")
    """
    function PlotLens(;
            xl::Float64 = 2.0,
            nx::Int = 256,
            beta::Vector{Float64}   = [0.0, 0.0],
            LensModel,
            LensKwargs::Dict,
            SourceProfile = nothing,
            SourceKwargs::Dict = Dict(),
            log_mag::Bool = true,
            show_critical::Bool = true,
            show_caustic::Bool = true,
            adaptive::Bool = true,
            figsize::NTuple{2,Int}  = (6, 6),
            colormap = :inferno,
            title::String = "lens system",
            kwargs...
        )
        # 1. Grid
        xg, yg = LensUtils.LensGrid(; xl=xl, nx=nx)

        # 2. Image
        if SourceProfile !== nothing
            image = LensBase.LensRayShooting(xg, yg;
                        LensModel=LensModel, LensKwargs=LensKwargs,
                        SourceProfile=SourceProfile, SourceKwargs=SourceKwargs)
            ttl = title
        else
            magr = LensBase.LensDetJacobian(xg, yg;
                        LensModel=LensModel, LensKwargs=LensKwargs)
            if log_mag
                image = log10.(abs.(magr))
                ttl = "$title  log₁₀|µ|"
            else
                image = magr
                ttl = "$title  µ"
            end
            colormap = :RdBu
        end

        # 3. Canvas + plot
        canvas = LensCanvas(; xlims=(-xl, xl), ylims=(-xl, xl),
                            figsize, title=ttl)
        PlotPlane!(canvas, xg, yg, image; colormap, kwargs...)

        # 4. Overlays
        if show_critical
            PlotCriticalCurve!(canvas; LensModel, LensKwargs, adaptive,
                               color=:cyan, label="critical curve")
        end
        if show_caustic
            PlotCaustic!(canvas; LensModel, LensKwargs, adaptive,
                         color=:red, label="caustic")
        end

        return canvas
    end

end
