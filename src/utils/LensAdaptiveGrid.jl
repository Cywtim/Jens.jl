module LensAdaptiveGrid

    # ═══════════════════════════════════════════════════════════════
    #  LensAdaptiveGrid — critical-curve-aware pixel refinement
    #                       + adaptive rendering
    #
    #  Two-stage pipeline:
    #
    #    1. DETECTION  — RefinementMap(grid, lens; threshold, sub_n)
    #       Computes which pixels lie near the critical curve
    #       (|det(J)| ≪ 1) and pre-computes sub-pixel offsets.
    #
    #    2. RENDERING  — render_adaptive(sys, ref)
    #       Single-ray broadcast for most pixels, sub_n² rays for
    #       refined pixels.  Designed for MCMC: compute `ref` once
    #       before the chain, reuse across all iterations.
    #
    #  The critical curve is where det(A) = (1−κ)² − γ² = 0.
    #  Pixels crossing this curve are stretched across large
    #  source-plane areas — a single ray at pixel centre misses
    #  flux from the stretched arc.  Adaptive sub-sampling traces
    #  multiple rays per pixel in these regions.
    #
    #  Usage:
    #    ref = RefinementMap(grid, fiducial_lens; threshold=0.3, sub_n=4)
    #    model = render_adaptive(sys, ref)
    # ═══════════════════════════════════════════════════════════════

    using Jens.LensBase: lens_hessian, lens_derivative
    using Jens.LightModel: AbstractLight, ExtendedSource, PointImage,
                           PointImages, CompositeImage, evaluate_source
    using Jens.LensGenerator: LensedPlane, MultiLensedPlane,
                              LightPlane, MultiLightPlane
    using Jens.LensPSF: conv_psf
    import Jens.LensSystem: ForwardModel

    export RefinementMap, adaptive_grid_info, render_adaptive


    # ═══════════════════════════════════════════════════════════════
    #  PART 1: Detection — RefinementMap
    # ═══════════════════════════════════════════════════════════════

    """
        RefinementMap(grid, lens; threshold=0.1, sub_n=4, z_source=nothing)

    Pre-computed refinement map for a lens model on a grid.

    # Fields
    - `needs_refine::BitMatrix`  — pixels needing sub-sampling
    - `sub_n::Int`               — sub-pixel factor (2 → 2×2, 3 → 3×3)
    - `offsets::Matrix{Float64}` — (sub_n², 2) offset pairs in pixel units
    - `weights::Vector{Float64}` — 1/sub_n² for each sub-pixel (uniform)
    - `n_refined::Int`           — number of pixels flagged for refinement
    - `threshold::Float64`       — |det(J)| below which a pixel is refined

    # How it works

    For each pixel centre, compute the lensing Jacobian determinant:

        det(A) = (1 − κ)² − (γ₁² + γ₂²)

    where κ, γ₁, γ₂ are derived from `lens_hessian`.  If |det(A)| falls
    below `threshold`, the pixel is flagged for sub-sampling.

    # MCMC usage

    Compute once before the chain with a fiducial lens model and a
    conservative `threshold` (0.2–0.3) to cover the parameter range
    explored by MCMC.  Do NOT recompute `ref` inside the logp function.
    """
    struct RefinementMap
        needs_refine::BitMatrix
        sub_n::Int
        offsets::Matrix{Float64}
        weights::Vector{Float64}
        n_refined::Int
        threshold::Float64
    end

    function RefinementMap(grid, lens; threshold::Real=0.1,
                            sub_n::Int=4, z_source=nothing)
        xg, yg = grid.xg, grid.yg
        nx, ny = size(xg)

        needs_refine = falses(nx, ny)
        n_refined = 0

        for j in 1:ny, i in 1:nx
            fxx, fxy, fyy = lens_hessian(lens, [xg[i, j]], [yg[i, j]];
                                         z_source=z_source)
            kappa  = (fxx[1] + fyy[1]) / 2
            gamma1 = (fxx[1] - fyy[1]) / 2
            gamma2 = fxy[1]
            detA   = (1 - kappa)^2 - (gamma1^2 + gamma2^2)

            if abs(detA) < threshold
                needs_refine[i, j] = true
                n_refined += 1
            end
        end

        # Pre-compute sub-pixel offsets (uniform grid within [-0.5, 0.5]²)
        offsets = Matrix{Float64}(undef, sub_n^2, 2)
        w = 1.0 / sub_n^2
        weights = fill(w, sub_n^2)
        k = 1
        for si in 1:sub_n, sj in 1:sub_n
            offsets[k, 1] = (si - 0.5) / sub_n - 0.5
            offsets[k, 2] = (sj - 0.5) / sub_n - 0.5
            k += 1
        end

        return RefinementMap(needs_refine, sub_n, offsets, weights,
                             n_refined, Float64(threshold))
    end

    """
        adaptive_grid_info(ref::RefinementMap, grid)

    Print a summary of the refinement map: total pixels, refined count,
    fraction, threshold, and estimated ray overhead.
    """
    function adaptive_grid_info(ref::RefinementMap, grid)
        total = length(ref.needs_refine)
        frac  = ref.n_refined / total * 100
        n_rays = (total - ref.n_refined) + ref.n_refined * ref.sub_n^2
        println("AdaptiveGrid: $(ref.n_refined)/$(total) pixels refined ",
                "($(round(frac, digits=1))%)")
        println("  threshold: |detJ| < $(ref.threshold)")
        println("  sub-sampling: $(ref.sub_n)×$(ref.sub_n)")
        println("  total rays: $(n_rays) (vs $(total) uniform)")
        println("  overhead: $(round(n_rays/total*100 - 100, digits=1))%")
    end


    # ═══════════════════════════════════════════════════════════════
    #  PART 2: Adaptive rendering
    # ═══════════════════════════════════════════════════════════════

    """
        model = render_adaptive(sys::ForwardModel, ref::RefinementMap)

    Render with adaptive sub-sampling.

    Pixels flagged by `ref.needs_refine` are ray-traced at `sub_n²`
    sub-positions and averaged; all other pixels use a single ray
    (identical to `LensSystem.render`).

    `ref` should be computed once before MCMC with a fiducial lens
    model and a conservative `threshold`.

    # Example
    ```julia
    ref = RefinementMap(grid, fiducial_lens; threshold=0.3, sub_n=4)
    model = render_adaptive(sys, ref)
    ```
    """
    function render_adaptive(sys::ForwardModel, ref::RefinementMap)
        return _render_adaptive(sys, sys.source_plane, ref)
    end


    # ── Source-plane dispatchers ──────────────────────────────────

    function _render_adaptive(sys::ForwardModel, lp::LightPlane,
                               ref::RefinementMap)
        return _render_adaptive_light(sys, lp.light, lp.z, ref)
    end

    function _render_adaptive(sys::ForwardModel, mlp::MultiLightPlane,
                               ref::RefinementMap)
        result = nothing
        for (light, z) in mlp.planes
            c = _render_adaptive_light(sys, light, z, ref)
            result = result === nothing ? c : result .+ c
        end
        return result
    end


    # ── ExtendedSource: adaptive sub-sampling ※ the main workhorse ──

    function _render_adaptive_light(sys::ForwardModel, src::ExtendedSource,
                                     z_src, ref::RefinementMap)
        xg, yg = sys.grid.xg, sys.grid.yg
        pix_size = Float64(sys.grid.pix_size)
        T = eltype(xg)

        # Step 1: single-ray base image (same as current render, GPU broadcast)
        ax, ay = lens_derivative(sys.lens_plane, xg, yg; z_source=z_src)
        result = evaluate_source(src, xg .- ax, yg .- ay)

        # Step 2: overwrite refined pixels with sub-pixel average
        offsets = ref.offsets    # (sub_n², 2)
        weights = ref.weights    # (sub_n²,)
        n_sub   = size(offsets, 1)

        @inbounds for j in axes(xg, 2), i in axes(xg, 1)
            ref.needs_refine[i, j] || continue

            x_c = xg[i, j]
            y_c = yg[i, j]
            s = zero(T)

            for k in 1:n_sub
                dx = T(offsets[k, 1] * pix_size)
                dy = T(offsets[k, 2] * pix_size)
                x_sub = x_c + dx
                y_sub = y_c + dy

                # Single-point lens derivative + source eval
                ax_s, ay_s = lens_derivative(
                    sys.lens_plane, [x_sub], [y_sub]; z_source=z_src)
                bx = x_sub - ax_s[1]
                by = y_sub - ay_s[1]
                s += evaluate_source(src, [bx], [by])[1] * T(weights[k])
            end

            result[i, j] = s
        end

        # Step 3: PSF convolution (same as current render)
        return _apply_psf(sys, result)
    end


    # ── PointImage / PointImages: no adaptive needed ──
    #     render_point! already does 5×5 supersampling via n_sub=5.
    #     Delegate directly to LensSystem's internal point-source render path.

    function _render_adaptive_light(sys::ForwardModel, pt::PointImage,
                                     z_src, ::RefinementMap)
        return LensSystem._render_light(sys, pt, z_src)
    end

    function _render_adaptive_light(sys::ForwardModel, pi::PointImages,
                                     z_src, ::RefinementMap)
        return LensSystem._render_light(sys, pi, z_src)
    end

    # ── CompositeImage: recurse into components ──

    function _render_adaptive_light(sys::ForwardModel, comp::CompositeImage,
                                     z_src, ref::RefinementMap)
        result = nothing
        for component in comp.sources
            c = _render_adaptive_light(sys, component, z_src, ref)
            result = result === nothing ? c : result .+ c
        end
        return result
    end


    # ═══════════════════════════════════════════════════════════════
    #  Helpers
    # ═══════════════════════════════════════════════════════════════

    # PSF application — mirrors LensSystem._apply_psf (private there)
    @inline _apply_psf(sys::ForwardModel, result) =
        sys.psf === nothing ? result :
        conv_psf(result, sys.psf, sys.grid.pix_size)

end # module LensAdaptiveGrid