module LensBase

    using AstroLib, NLsolve, Optim

    include("LensUtils.jl")
    using .LensUtils: ndgrid, Pol2Car, Car2Pol

    export LensCheck, LensFermat, LensDeflection
    export LensRayShootingPosition, LensRayShooting
    export MultiLensRayShootingPosition, MultiLensRayShooting
    export LensCriticalCurve, LensCaustic
    export LensAdaptiveCriticalCurve, LensAdaptiveCaustic

    # ═══════════════════════════════════════════════════════════════
    #  LensBase — Core lensing framework
    #
    #  All models (SIE, SIS, NIE, EPL, NFW, Shear, etc.) expose:
    #    LensCheck(; kwargs...)          → validation
    #    LensMass(xg, yg; kwargs...)     → lensing potential ψ
    #    LensDerivative(xg, yg; kwargs...) → deflection (α_x, α_y)
    #    LensHessian(xg, yg; kwargs...)  → Hessian (f_xx, f_xy, f_yy)
    #
    #  Type convention: accept any AbstractArray (Matrix, SubArray, etc.)
    #  GridSL from LazyGrids is no longer required — use Matrix directly.
    # ═══════════════════════════════════════════════════════════════

    function LensCheck(LensModel; LensKwargs)
        
        LensModel.LensCheck(; LensKwargs...)

    end


    function LensFermat(xg::AbstractMatrix, yg::AbstractMatrix,
         beta=[0., 0.]; LensModel::Module, LensKwargs::Dict)
        
        phi = LensModel.LensMass(xg, yg; LensKwargs...)

        Fermat = @. ((beta[1] - xg)^2 + (beta[2] - yg)^2) / 2 - phi

        return Fermat

    end

    function LensDeflection(xg::AbstractArray, yg::AbstractArray;
         LensModel::Module, LensKwargs::Dict )
        #=?=#
        alpha_x, alpha_y = LensModel.LensDerivative(xg, yg; LensKwargs...)
        
        return alpha_x, alpha_y

    end

    function LensPlane(xg::AbstractArray, yg::AbstractArray;
       LensModel::Module, LensKwargs::Dict )
      #=?=#
      alpha_x, alpha_y = LensModel.LensDerivative(xg, yg; LensKwargs...)
      
      beta_x = @. xg - alpha_x
      beta_y = @. yg - alpha_y

      return beta_x, beta_y

    end

    function LensMagnificationR(thetax::AbstractArray,
      thetay::AbstractArray;
       LensModel::Module, LensKwargs::Dict)

          h_xx, h_xy, h_yy = LensModel.LensHessian(thetax, thetay; LensKwargs...)
          
          magr = @. (1. .- h_xx) .* (1. .- h_yy) .- h_xy.^2.;

          return magr

    end

    function LensRayShootingPosition(
      thetax::AbstractArray,
         thetay::AbstractArray;
          LensModel::Module, LensKwargs::Dict)

        alphax, alphay = LensModel.LensDerivative(thetax, thetay; LensKwargs...)
        betax = thetax .- alphax
        betay = thetay .- alphay

        return betax, betay
    end

    function MultiLensRayShootingPosition(
      thetax::AbstractArray,
        thetay::AbstractArray;
          LensModel::Union{Module, Vector{Module}},
            LensKwargs::Vector{Dict{Symbol, Float64}})

        xl = [thetax]
        yl = [thetay]
          
        len = length(LensKwargs)

        if typeof(LensModel) == Module

          for i in 1:len
            ax, ay = LensModel.LensDerivative(xl[i], yl[i]; LensKwargs[i]...)
            push!(xl, xl[i] .- ax)
            push!(yl, yl[i] .- ay)
          end
    
          return xl, yl

        elseif typeof(LensModel) == Vector{Module}

          for i in 1:len
            ax, ay = LensModel[i].LensDerivative(xl[i], yl[i]; LensKwargs[i]...)
            push!(xl, xl[i] .- ax)
            push!(yl, yl[i] .- ay)
          end

          return xl, yl
        
        else

          println("The type of LensModel is not Module or Moduel Vector.")

        end

     end


    function LensRayShooting(thetax::AbstractArray,
         thetay::AbstractArray;
          LensModel::Module, LensKwargs::Dict,
             SourceProfile::Function, SourceKwargs::Dict)

        alphax, alphay = LensModel.LensDerivative(thetax, thetay; LensKwargs...)
        betax = thetax .- alphax
        betay = thetay .- alphay

        light = SourceProfile(betax, betay; SourceKwargs...)

        return light

    end

    function MultiLensRayShooting(
      thetax::AbstractArray,
        thetay::AbstractArray;
          LensModel::Union{Module, Vector{Module}},
            LensKwargs::Vector{Dict{Symbol, Float64}},
              SourceProfile::Function, SourceKwargs::Dict)

        xl = [thetax]
        yl = [thetay]
          
        len = length(LensKwargs)

        if typeof(LensModel) == Module

          for i in 1:len
            ax, ay = LensModel.LensDerivative(xl[i], yl[i]; LensKwargs[i]...)
            push!(xl, xl[i] .- ax)
            push!(yl, yl[i] .- ay)
          end
    
          return xl, yl

        elseif typeof(LensModel) == Vector{Module}

          for i in 1:len
            ax, ay = LensModel[i].LensDerivative(xl[i], yl[i]; LensKwargs[i]...)
            push!(xl, xl[i] .- ax)
            push!(yl, yl[i] .- ay)
          end

          return xl, yl
        
        else

          println("The type of LensModel is not Module or Moduel Vector.")

        end

     end


    function LensCriticalCurve(; LensModel::Module, LensKwargs::Dict, hperr::Float64=0.01, 
      r_max::Float64=2., r_bins::Int=4000, theta_bins::Int=4000)

        r = range(0, r_max, r_bins)
        theta = range(0, 2 * pi, theta_bins)
        (rg, thetag) = ndgrid(collect(r), collect(theta));
        (xpg, ypg) = LensUtils.Pol2Car(rg, thetag, xc=-LensKwargs[:xcentre], yc=-LensKwargs[:ycentre])
      
        hp_xx, hp_xy, hp_yy = LensModel.LensHessian(xpg, ypg; LensKwargs...)
        hp = @. (1-hp_xx)*(1-hp_yy) - hp_xy^2

        ccindex = findall( 0.0 .< hp .< hperr)

        ccx = [xpg[i] for i in ccindex]
        ccy = [ypg[i] for i in ccindex]

        return ccx, ccy
    end

    function LensCaustic(; LensModel::Module, LensKwargs::Dict, hperr::Float64=0.01, 
            r_max::Float64=2., r_bins::Int=4000, theta_bins::Int=4000)

        r = range(0, r_max, r_bins)
        theta = range(0, 2 * pi, theta_bins)
        (rg, thetag) = ndgrid(collect(r), collect(theta));
        (xpg, ypg) = LensUtils.Pol2Car(rg, thetag, xc=-LensKwargs[:xcentre], yc=-LensKwargs[:ycentre])

        fp_x, fp_y = LensModel.LensDerivative(xpg, ypg; LensKwargs...) # deflection angles f_i
        xpb, ypb = xpg .- fp_x, ypg .- fp_y  # lensed image plane
      
        hp_xx, hp_xy, hp_yy = LensModel.LensHessian(xpg, ypg; LensKwargs...)
        hp = @. (1-hp_xx)*(1-hp_yy) - hp_xy^2

        ccindex = findall( 0.0 .< hp .< hperr)

        csx = [xpb[i] for i in ccindex]
        csy = [ypb[i] for i in ccindex]

        return csx, csy

    end

    # ════════════════════════════════════════════════════════════════════
    #  ADAPTIVE-GRID CAUSTIC / CRITICAL CURVE
    #
    #  Uses quadtree refinement: start coarse, subdivide only cells that
    #  contain a zero of det(1 - H). This concentrates evaluations where
    #  the curve actually lies, giving higher resolution with fewer
    #  Hessian evaluations vs. the uniform polar-grid approach.
    #
    #  Reference: Daněk & Heyrovský 2015, ApJ 806, 63
    # ════════════════════════════════════════════════════════════════════

    """
        _hp_at!(cache, x, y, LensModel, LensKwargs) → hp

    Evaluate µ⁻¹ = det(1 - H(θ)) = (1 − f_xx)(1 − f_yy) − f_xy² at a
    single point (x, y). Results are cached in `cache` (a Dict keyed by
    `(Float64, Float64)`) to avoid redundant Hessian evaluations at
    shared cell corners.
    """
    function _hp_at!(cache::Dict{Tuple{Float64,Float64},Float64},
                      x::Float64, y::Float64,
                      LensModel::Module, LensKwargs::Dict)
        key = (x, y)
        if haskey(cache, key)
            return cache[key]
        end
        f_xx, f_xy, f_yy = LensModel.LensHessian([x], [y]; LensKwargs...)
        hp = (1.0 - f_xx[1]) * (1.0 - f_yy[1]) - f_xy[1]^2
        cache[key] = hp
        return hp
    end

    """
        _cell_hp!(cache, x1, y1, x2, y2, LensModel, LensKwargs) → (hp_ll, hp_lr, hp_ul, hp_ur)

    Evaluate µ⁻¹ at the four corners of the cell [x1,x2]×[y1,y2].
    Corners: lower-left, lower-right, upper-left, upper-right.
    """
    function _cell_hp!(cache, x1, y1, x2, y2, LensModel, LensKwargs)
        return (
            _hp_at!(cache, x1, y1, LensModel, LensKwargs),
            _hp_at!(cache, x2, y1, LensModel, LensKwargs),
            _hp_at!(cache, x1, y2, LensModel, LensKwargs),
            _hp_at!(cache, x2, y2, LensModel, LensKwargs),
        )
    end

    """
        _cell_spans_zero(hp_corners) → Bool

    A cell "spans" the critical curve if its four corner values of
    µ⁻¹ = det(1−H) have opposite signs (guarantees a zero crossing by
    the intermediate value theorem), OR if they all have the same sign
    but are very close to zero.

    The second case captures tangential critical curves where µ⁻¹ stays
    on one side of zero but comes arbitrarily close (e.g., a ring).
    """
    function _cell_spans_zero(hp_ll, hp_lr, hp_ul, hp_ur, hp_threshold::Float64)
        hp_vals = (hp_ll, hp_lr, hp_ul, hp_ur)
        # Sign change → guaranteed zero crossing
        if sign(minimum(hp_vals)) != sign(maximum(hp_vals))
            return true
        end
        # All same sign but very close to zero (tangential caustics, rings)
        return maximum(abs.(hp_vals)) < hp_threshold
    end

    """
        LensAdaptiveCriticalCurve(;
            LensModel, LensKwargs,
            xlim=(-2.0, 2.0), ylim=(-2.0, 2.0),
            initial_nx=16, initial_ny=16,
            max_depth=6, hp_threshold=1e-4) → (ccx, ccy)

    Trace the critical curve (lens-plane points where µ⁻¹ = 0) using
    quadtree refinement.

    **Algorithm**: start with a coarse (initial_nx × initial_ny) Cartesian
    grid, evaluate µ⁻¹ = det(1−H) at each corner, subdivide cells that
    contain a zero crossing, and recurse up to `max_depth`. Returns the
    set of deepest-level cell corners that bracket the curve.

    **Parameters**:
    - `xlim`, `ylim`: bounding box in the image (lens) plane
    - `initial_nx`, `initial_ny`: coarsest grid resolution
    - `max_depth`: max quadtree subdivisions (final cell size ≈
      range / 2^max_depth / initial_n; default ~0.03)
    - `hp_threshold`: cells with |µ⁻¹| < threshold at all corners are
      refined even without a sign change (catches tangential curves)

    **Returns**: (ccx, ccy) — Float64 vectors of critical-curve points.
    """
    function LensAdaptiveCriticalCurve(;
        LensModel::Module, LensKwargs::Dict,
        xlim::NTuple{2,Float64}=(-2.0, 2.0), ylim::NTuple{2,Float64}=(-2.0, 2.0),
        initial_nx::Int=16, initial_ny::Int=16,
        max_depth::Int=6, hp_threshold::Float64=1e-4)

        (xmin, xmax) = xlim
        (ymin, ymax) = ylim
        dx0 = (xmax - xmin) / initial_nx
        dy0 = (ymax - ymin) / initial_ny

        # Hessian cache: shared corners across adjacent cells
        hp_cache = Dict{Tuple{Float64,Float64},Float64}()

        # Work queue: each entry is (x1, y1, x2, y2, depth)
        cells = Tuple{Float64,Float64,Float64,Float64,Int}[]

        # Seed initial coarse grid
        for ix in 1:initial_nx, iy in 1:initial_ny
            x1 = xmin + (ix - 1) * dx0
            x2 = x1 + dx0
            y1 = ymin + (iy - 1) * dy0
            y2 = y1 + dy0
            push!(cells, (x1, y1, x2, y2, 0))
        end

        # Result containers (avoid appending in hot loop — collect at end)
        result_x = Float64[]
        result_y = Float64[]

        # Process queue — "growing front" ensures cache benefit
        while !isempty(cells)
            (x1, y1, x2, y2, depth) = popfirst!(cells)

            hp_ll, hp_lr, hp_ul, hp_ur = _cell_hp!(
                hp_cache, x1, y1, x2, y2, LensModel, LensKwargs)

            if depth >= max_depth
                # Deepest level: collect the four corners
                append!(result_x, [x1, x2, x1, x2])
                append!(result_y, [y1, y1, y2, y2])
                continue
            end

            if _cell_spans_zero(hp_ll, hp_lr, hp_ul, hp_ur, hp_threshold)
                # Subdivide into 4
                xm = (x1 + x2) / 2.0
                ym = (y1 + y2) / 2.0
                d = depth + 1
                push!(cells, (x1, y1, xm, ym, d))  # lower-left
                push!(cells, (xm, y1, x2, ym, d))  # lower-right
                push!(cells, (x1, ym, xm, y2, d))  # upper-left
                push!(cells, (xm, ym, x2, y2, d))  # upper-right
            end
            # else: cell is far from the curve — discard
        end

        return result_x, result_y
    end

    """
        LensAdaptiveCaustic(;
            LensModel, LensKwargs,
            xlim=(-2.0, 2.0), ylim=(-2.0, 2.0),
            initial_nx=16, initial_ny=16,
            max_depth=6, hp_threshold=1e-4) → (csx, csy)

    Trace the caustic (source-plane points where µ⁻¹ = 0) using
    quadtree refinement in the image plane.

    Internally calls `LensAdaptiveCriticalCurve` to find the critical
    curve in the image plane, then maps each point to the source plane
    via the lens equation β = θ − α(θ).

    **Parameters**: same as `LensAdaptiveCriticalCurve`.

    **Returns**: (csx, csy) — Float64 vectors of caustic points.
    """
    function LensAdaptiveCaustic(;
        LensModel::Module, LensKwargs::Dict,
        xlim::NTuple{2,Float64}=(-2.0, 2.0), ylim::NTuple{2,Float64}=(-2.0, 2.0),
        initial_nx::Int=16, initial_ny::Int=16,
        max_depth::Int=6, hp_threshold::Float64=1e-4)

        # 1. Find critical curve in the image plane
        ccx, ccy = LensAdaptiveCriticalCurve(;
            LensModel, LensKwargs,
            xlim, ylim, initial_nx, initial_ny, max_depth, hp_threshold)

        if isempty(ccx)
            return Float64[], Float64[]
        end

        # 2. Batch-compute deflection for all critical points
        α_x, α_y = LensModel.LensDerivative(ccx, ccy; LensKwargs...)

        # 3. Map to source plane: β = θ − α(θ)
        csx = ccx .- α_x
        csy = ccy .- α_y

        return csx, csy
    end

end
