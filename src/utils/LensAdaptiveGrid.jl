module LensAdaptiveGrid

    # ═══════════════════════════════════════════════════════════════
    #  LensAdaptiveGrid — critical-curve-aware pixel refinement
    #
    #  Detects pixels near the critical curve (|det J| ≪ 1) and
    #  returns sub-pixel sampling positions for those pixels.
    #
    #  The critical curve is where det(A) = (1−κ)² − γ² = 0.
    #  Pixels crossing this curve are stretched across large
    #  source-plane areas — a single ray at pixel centre misses
    #  flux from the stretched arc.  Adaptive sub-sampling traces
    #  multiple rays per pixel in these regions.
    #
    #  Usage:
    #    ref = RefinementMap(grid, lens; threshold=0.1, sub_n=4)
    #    ref.needs_refine   # BitMatrix — which pixels need sub-sampling
    #    offsets, weights   # sub-pixel (dx, dy) offsets and weights
    # ═══════════════════════════════════════════════════════════════

    using Jens.LensBase: lens_hessian

    export RefinementMap, adaptive_grid_info

    # ── Struct ─────────────────────────────────────────────────────

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

    # Performance note

    Computing hessian at every pixel is O(N²).  For large grids, consider:
    - Using a coarser detection grid (stride every 2-4 pixels)
    - Computing once for a fiducial model and reusing during MCMC
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

    # ── Info ───────────────────────────────────────────────────────

    """
        info = adaptive_grid_info(ref::RefinementMap, grid)

    Print a summary of the refinement map: total pixels, refined count,
    fraction, threshold.
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

end