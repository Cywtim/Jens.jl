# ═══════════════════════════════════════════════════════════════
#  LensMask — pixel mask construction for selective fitting
#
#  Masks are boolean arrays: true = include pixel, false = exclude.
#  GPU-safe — all operations are pure broadcast, no scalar indexing.
#
#  Usage:
#    mask  = circular_mask(grid, 2.0; centre=(0.3, -0.1))
#    mask  = annular_mask(grid, 0.5, 2.0; centre=(0, 0))
#    mask  = rectangular_mask(grid, -1.0, 1.0, -0.5, 0.5)
#    mask  = from_matrix(user_array)         # 0/1 matrix → Bool mask
#    mask  = combine(m1, m2; op=:|)          # union / intersection
#
#  In MCMC logp:
#    chi2  = sum((data .- model).^2 .* mask)
#    logp  = -chi2 / (2σ²)
# ═══════════════════════════════════════════════════════════════

module LensMask

    using Jens.LensGenerator: Grid, GenGrid

    export circular_mask, annular_mask, rectangular_mask
    export from_matrix, combine, invert
    export n_valid

    # ═══════════════════════════════════════════════════════════
    #  Grid dispatch — extract coordinate arrays
    # ═══════════════════════════════════════════════════════════

    "Extract (xg, yg) from Grid or raw coordinate arrays."
    @inline _coords(grid::Grid) = grid.xg, grid.yg
    @inline _coords((xg, yg)::Tuple{<:AbstractMatrix, <:AbstractMatrix}) = xg, yg

    # ═══════════════════════════════════════════════════════════
    #  circular_mask
    # ═══════════════════════════════════════════════════════════

    """
        mask = circular_mask(grid_or_coords, radius; centre=(0.0, 0.0))

    Boolean mask: `true` for pixels within `radius` of `centre`.

    # Arguments
    - `grid_or_coords`: `Grid`, `GridGPU`, or `(xg, yg)` tuple of coordinate arrays.
    - `radius`: inclusion radius in arcseconds.
    - `centre`: `(x, y)` centre in arcseconds (default: origin).

    # Example
        mask = circular_mask(grid, 2.0; centre=(0.3, -0.1))
        chi2 = sum((data .- model).^2 .* mask)
    """
    function circular_mask(grid_or_coords, radius::Real;
                           centre::Tuple{Real,Real}=(0.0, 0.0))
        xg, yg = _coords(grid_or_coords)
        xc, yc = centre
        R2 = @. (xg - xc)^2 + (yg - yc)^2
        return @. R2 <= radius^2
    end

    # ═══════════════════════════════════════════════════════════
    #  annular_mask
    # ═══════════════════════════════════════════════════════════

    """
        mask = annular_mask(grid_or_coords, r_in, r_out; centre=(0.0, 0.0))

    Boolean mask: `true` for pixels in the annulus  `r_in < R ≤ r_out`.

    Useful for masking out the lens galaxy centre (where model is poor)
    while keeping the Einstein ring region.

    # Example
        mask = annular_mask(grid, 0.3, 2.5; centre=(0, 0))
    """
    function annular_mask(grid_or_coords, r_in::Real, r_out::Real;
                          centre::Tuple{Real,Real}=(0.0, 0.0))
        xg, yg = _coords(grid_or_coords)
        xc, yc = centre
        R2 = @. (xg - xc)^2 + (yg - yc)^2
        return @. (R2 > r_in^2) & (R2 <= r_out^2)
    end

    # ═══════════════════════════════════════════════════════════
    #  rectangular_mask
    # ═══════════════════════════════════════════════════════════

    """
        mask = rectangular_mask(grid_or_coords, x_min, x_max, y_min, y_max)

    Boolean mask: `true` for pixels inside the rectangle `[x_min, x_max] × [y_min, y_max]`.

    # Example
        mask = rectangular_mask(grid, -1.5, 1.5, -1.0, 1.0)
    """
    function rectangular_mask(grid_or_coords,
                              x_min::Real, x_max::Real,
                              y_min::Real, y_max::Real)
        xg, yg = _coords(grid_or_coords)
        return @. (xg >= x_min) & (xg <= x_max) & (yg >= y_min) & (yg <= y_max)
    end

    # ═══════════════════════════════════════════════════════════
    #  from_matrix  — user-provided 0/1 matrix
    # ═══════════════════════════════════════════════════════════

    """
        mask = from_matrix(A)

    Convert a user-provided matrix to a boolean mask.
    Convention:  `0` = masked out (excluded),  `1` = unmasked (included).

    Input `A` can be:
    - `AbstractMatrix{<:Integer}`  —  zeros and ones
    - `AbstractMatrix{Bool}`       —  passed through unchanged
    - `AbstractMatrix{<:Real}`     —  `> 0.5` treated as included

    # Example
        # From FITS file or manual definition
        A = [1 1 0 0;
             1 1 0 0;
             0 0 1 1;
             0 0 1 1]
        mask = from_matrix(A)

        # In MCMC:
        chi2 = sum((data .- model).^2 .* mask)
    """
    function from_matrix(A::AbstractMatrix{Bool})
        return A
    end

    function from_matrix(A::AbstractMatrix{<:Integer})
        return @. A != 0
    end

    function from_matrix(A::AbstractMatrix{<:Real})
        return @. A > 0.5
    end

    # ═══════════════════════════════════════════════════════════
    #  Utility: combine / invert / n_valid
    # ═══════════════════════════════════════════════════════════

    """
        mask = combine(m1, m2; op=:|)

    Combine two boolean masks.  Default `op=:|` gives the union
    (pixels included in *either* mask).  Use `op=&` for intersection.

    # Example
        inner = circular_mask(grid, 0.5)
        outer = annular_mask(grid, 1.5, 2.5)
        both  = combine(inner, outer)       # union
        only  = combine(inner, outer; op=&) # intersection
    """
    function combine(m1::AbstractMatrix{Bool}, m2::AbstractMatrix{Bool}; op::Function=Base.:(|))
        return @. op(m1, m2)
    end

    """
        mask = invert(mask)

    Invert a boolean mask: included ↔ excluded.
    """
    function invert(mask::AbstractMatrix{Bool})
        return @. !mask
    end

    """
        n = n_valid(mask)

    Number of included pixels.  Use for normalisation or diagnostics.
    """
    function n_valid(mask::AbstractMatrix{Bool})
        return sum(mask)
    end

    # ═══════════════════════════════════════════════════════════
    #  overlay_mask — visualise mask on an image
    # ═══════════════════════════════════════════════════════════

    """
        rgb = overlay_mask(image, mask; color=:red, alpha=0.4)

    Return an RGB array (N×M×3) with masked pixels tinted.
    Unmasked pixels show the original image; masked pixels are
    overlaid with a semi-transparent colour.

    # Arguments
    - `image`: 2-D grayscale array (e.g. render output)
    - `mask`:  boolean mask — `true` = unmasked, `false` = masked
    - `color`: `:red`, `:grey`, `:blue`, or RGB tuple e.g. `(1.0, 0.2, 0.2)`
    - `alpha`: opacity of the overlay on masked pixels (0–1)

    # Example
    ```julia
    using Plots
    rgb = overlay_mask(render(sys), sys.mask; color=:grey, alpha=0.5)
    plot(heatmap(rgb[:,:,1]', rgb[:,:,2]', rgb[:,:,3]'))  # needs combining
    # Or save directly:
    save(\"lens_masked.png\", colorview(RGB, permutedims(rgb, (3,1,2))))
    ```
    """
    function overlay_mask(image::AbstractMatrix{<:Real},
                          mask::AbstractMatrix{Bool};
                          color::Union{Symbol, Tuple{Real,Real,Real}}=:red,
                          alpha::Real=0.4)
        # Normalise image to [0, 1]
        im_min, im_max = extrema(image)
        rng = im_max - im_min
        normed = rng > 0 ? (image .- im_min) ./ rng : zeros(size(image))

        # Resolve colour
        rgb_tint = _resolve_color(color)

        # Build RGB output
        R = similar(normed); G = similar(normed); B = similar(normed)
        R .= normed; G .= normed; B .= normed

        # Blend masked pixels:  (1-α)*original + α*tint
        r_t, g_t, b_t = rgb_tint
        masked = @. !mask
        R = @. ifelse(masked, (1-alpha)*normed + alpha*r_t, normed)
        G = @. ifelse(masked, (1-alpha)*normed + alpha*g_t, normed)
        B = @. ifelse(masked, (1-alpha)*normed + alpha*b_t, normed)

        return cat(R, G, B; dims=3)
    end

    # ── colour lookup ──
    function _resolve_color(c::Symbol)
        if c == :red;   return (1.0, 0.2, 0.2)
        elseif c == :grey || c == :gray; return (0.5, 0.5, 0.5)
        elseif c == :blue;  return (0.2, 0.4, 1.0)
        elseif c == :green; return (0.2, 0.8, 0.2)
        else; error("Unknown color: $c. Use :red, :grey, :blue, :green, or an RGB tuple.")
        end
    end
    _resolve_color(c::Tuple{Real,Real,Real}) = Float64.(c)

    export overlay_mask

end # module LensMask