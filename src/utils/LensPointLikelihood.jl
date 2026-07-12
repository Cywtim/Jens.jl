module LensPointLikelihood

    # ═══════════════════════════════════════════════════════════════
    #  LensPointLikelihood — χ² for discrete observables
    #
    #  Provides likelihood functions for fitting lens models to
    #  point-source observables: image positions and time delays.
    #  Complements the pixel-level `masked_logp`/`masked_chi2`.
    #
    #  Usage:
    #    images, delays = image_time_delays(sys, βx, βy)
    #    χ² = point_image_chi2(images, obs_positions, σ_pos)
    #      + time_delay_chi2(delays, obs_dt_pairs, σ_dt)
    #    logp = -0.5 * χ²
    # ═══════════════════════════════════════════════════════════════

    export point_image_chi2, time_delay_chi2

    # ── Image position χ² ─────────────────────────────────────────

    """
        χ² = point_image_chi2(predicted, observed, σ_pos)
        χ² = point_image_chi2(predicted, observed, σ_pos; by=identity)

    Compute χ² between predicted and observed image positions.

    Images are matched by sorting both lists.  The default sort key
    is distance from origin (works for double / quad systems).

    # Arguments
    - `predicted` : vector of (θx, θy, μ) from `solve_images`
    - `observed`  : vector of (θx_obs, θy_obs)
    - `σ_pos`     : 1-σ positional uncertainty (arcsec)
    - `by`        : sort key function, default: t -> t[1]^2 + t[2]^2

    # Returns
    `Inf` if counts differ, else the reduced χ².

    # Example
    ```julia
    images = solve_images(sys, βx, βy)
    obs    = [(0.842, -0.300), (-0.336, 0.662)]
    χ²_pos = point_image_chi2(images, obs, 0.003)
    ```
    """
    function point_image_chi2(predicted, observed, σ_pos::Real;
                               by=nothing)
        length(predicted) != length(observed) && return Inf

        # default sort key: distance from origin
        key = by === nothing ? (t -> t[1]^2 + t[2]^2) : by
        pred_sorted = sort(predicted; by=key)
        obs_sorted  = sort(observed;  by=key)

        χ² = 0.0
        for i in eachindex(pred_sorted)
            dx = pred_sorted[i][1] - obs_sorted[i][1]
            dy = pred_sorted[i][2] - obs_sorted[i][2]
            χ² += (dx^2 + dy^2) / σ_pos^2
        end
        return χ²
    end

    # ── Time-delay χ² ─────────────────────────────────────────────

    """
        χ² = time_delay_chi2(delays, pairs, σ_dt)

    Compute χ² for pairwise time-delay differences.

    Time delays are measured as arrival-time *differences* between
    image pairs, not absolute delays.  Fitting the differences (rather
    than the individual delays) removes the D_Δt ∝ H₀⁻¹ degeneracy.

    # Arguments
    - `delays` : vector of absolute delays (seconds), aligned with image order
    - `pairs`  : vector of (i, j, Δt_obs) — 1-based image indices + observed delay
    - `σ_dt`   : 1-σ uncertainty on each pair (seconds)

    # Example
    ```julia
    delays = LensTimeDelay(sys, tx, ty)
    pairs  = [(1, 2, 8.57 * DAY_TO_SEC)]  # image 1 arrives 8.57 days after image 2
    χ²_dt  = time_delay_chi2(delays, pairs, 0.5 * DAY_TO_SEC)
    ```
    """
    function time_delay_chi2(delays, pairs, σ_dt::Real)
        χ² = 0.0
        for (i, j, dt_obs) in pairs
            dt_pred = delays[i] - delays[j]
            χ² += (dt_pred - dt_obs)^2 / σ_dt^2
        end
        return χ²
    end

end