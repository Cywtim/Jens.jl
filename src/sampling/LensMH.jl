# ═══════════════════════════════════════════════════════════════
#  LensMH — Standalone adaptive Metropolis-Hastings sampler
#
#  Zero dependencies beyond stdlib. Accepts any callable
#  `logp_fn(params::Vector) -> Float64` plus prior bounds.
#
#  Usage:
#    using Jens.LensMH
#
#    function my_logp(p)
#        theta_E, amp, Rsersic = p
#        # ... build model, render, compute log-likelihood ...
#        return log_prior + log_likelihood
#    end
#
#    # Single chain
#    chain = lens_mh(my_logp, [0.2, 0.1, 0.05], [2.0, 5.0, 1.0];
#                    n=2000, seed=42)
#
#    # Multi-start: try N random starts, pick best, run full chain
#    chain = lens_mh_multistart(my_logp, [0.2,0.1,0.05], [2.0,5.0,1.0];
#                               n_starts=8, n=2000, seed=42)
#    # chain.samples  → (n_params, n_samples) Matrix
#    # chain.logp     → log-posterior at each step
#    # chain.accepted → acceptance count
# ═══════════════════════════════════════════════════════════════

module LensMH

    using Random, Statistics, LinearAlgebra

    export MHResult, lens_mh, lens_mh_multi, lens_mh_multistart, chain, burnin, chain_stats

    """
        MHResult

    Holds the output of a Metropolis-Hastings run.

    Fields:
      - samples::Matrix{Float64}   (n_params, n_samples)
      - logp::Vector{Float64}      log-posterior at each sample
      - accepted::Int              number of accepted proposals
      - step_final::Vector{Float64} final per-parameter step sizes
    """
    struct MHResult
        samples::Matrix{Float64}
        logp::Vector{Float64}
        accepted::Int
        step_final::Vector{Float64}
    end

    """
        result = lens_mh(logp_fn, lower, upper; kw...)

    Adaptive Metropolis-Hastings sampler.

    # Arguments
      - `logp_fn(params::Vector{Float64}) -> Float64`:
        log-posterior (log prior + log likelihood).  Called once
        per proposal evaluation, i.e. once per iteration.
      - `lower::Vector{Float64}`:  prior lower bounds.
      - `upper::Vector{Float64}`:  prior upper bounds.

    # Keyword arguments
      - `n::Int = 2000`:  number of iterations.
      - `step0::Vector{Float64}`:  initial per-parameter step sizes.
        Defaults to `(upper - lower) / 20` (5% of prior width).
        Step size floor = `range / 200` (0.5%), ceiling = `range / 10` (10%).
      - `adapt::Bool = true`:  whether to adapt step sizes online.
        Target acceptance rate ≈ 35%.
      - `init::Union{Nothing, Vector{Float64}}`:
        Initial parameter vector.  Defaults to the midpoint
        `(lower + upper) / 2`.
      - `seed::Int = 42`:  RNG seed.

    # Returns
      `MHResult` with fields `samples`, `logp`, `accepted`,
      `step_final`.

    # Boundary handling
    Proposals that fall outside `[lower, upper]` are reflected back
    into the domain (mirror reflection at the violated bound).
    """
    function lens_mh(
        logp_fn,
        lower::Vector{Float64},
        upper::Vector{Float64};
        n::Int = 2000,
        step0::Union{Nothing, Vector{Float64}} = nothing,
        adapt::Bool = true,
        init::Union{Nothing, Vector{Float64}} = nothing,
        seed::Int = 42,
        n_adapt_delay::Int = 0,
    )
        d = length(lower)
        @assert length(upper) == d "lower and upper must have same length"

        Random.seed!(seed)

        # ── Initialisation ────────────────────────────────
        θ = init === nothing ? (lower .+ upper) ./ 2 : copy(init)
        range_all = upper .- lower
        step = step0 === nothing ? range_all ./ 20 : copy(step0)

        # Clamp initial value inside bounds
        θ .= clamp.(θ, lower, upper)

        current_lp = logp_fn(θ)
        accepted   = 0

        # Pre-allocate output
        samples = zeros(d, n)
        logp_hist = zeros(n)
        rng = MersenneTwister(seed)

        # ── Main loop ─────────────────────────────────────
        # n_adapt_delay: freeze steps for first N iterations (pure random walk)
        adaptive_now = adapt && (n_adapt_delay <= 0)

        for i in 1:n
            # Propose
            prop = θ .+ step .* randn(rng, d)

            # Reflect at boundaries
            _reflect!(prop, lower, upper)

            # Evaluate
            prop_lp = logp_fn(prop)
            α = prop_lp - current_lp  # log acceptance ratio

            if α >= 0 || rand(rng) < exp(α)
                # Accept
                θ = prop
                current_lp = prop_lp
                accepted += 1
                if adaptive_now
                    step .= min.(step .* 1.02, range_all ./ 10)
                end
            else
                if adaptive_now
                    step .= max.(step .* 0.98, range_all ./ 2000)
                end
            end

            # Enable adaptation after delay
            if !adaptive_now && i >= n_adapt_delay
                adaptive_now = true
            end

            samples[:, i] .= θ
            logp_hist[i]   = current_lp
        end

        return MHResult(samples, logp_hist, accepted, copy(step))
    end

    # ══════════════════════════════════════════════════════════
    #  Multi-chain MH  (CPU parallel via Threads.@threads)
    # ══════════════════════════════════════════════════════════

    """
            chains = lens_mh_multi(logp_fn, lower, upper; kw...)

        Run `n_chains` independent MH chains in parallel using Julia
        threads.  Each chain runs with a different random seed.

        # GPU usage
        `lens_mh_multi` works with GPU `logp_fn` out of the box — CUDA.jl
        automatically assigns a per-thread stream.  **Do NOT call**
        `CUDA.synchronize()` inside `logp_fn`; `Array(gpu_array)` already
        performs per-stream synchronization.  Explicit `CUDA.synchronize()`
        blocks ALL streams and kills multi-chain scaling.

        On a single GPU, total throughput is bounded by SM count, not by
        CPU threads: expect ~10 chain-calls/s for a 256² grid on an RTX
        4080 SUPER, regardless of `n_chains`.  For production-scale
        fitting, prefer a single GPU chain at full speed (14× CPU) over
        many underpowered chains competing for GPU resources.

        # Keyword arguments (same as `lens_mh`, plus):
          - `n_chains::Int = 4`:  number of parallel chains.
          - `seed::Int = 42`:  base seed; chain k uses `seed + k`.

        # Returns
          `Vector{MHResult}` of length `n_chains`.

        # Example
        ```julia
        chains = lens_mh_multi(lpfn, [0.2,0.1], [2.0,5.0];
                               n_chains=4, n=2000, init=[0.8,1.0])

        # Per-chain diagnostics
        for (k, c) in enumerate(chains)
            println("Chain \$k: accepted \$(c.accepted) / 2000")
        end

        # Combined posterior
        S = chain(chains; burn=500)
        m, s = chain_stats(chains; burn=500)
        ```
        """
    function lens_mh_multi(
        logp_fn,
        lower::Vector{Float64},
        upper::Vector{Float64};
        n_chains::Int = 4,
        n::Int = 2000,
        step0::Union{Nothing, Vector{Float64}} = nothing,
        adapt::Bool = true,
        init::Union{Nothing, Vector{Float64}} = nothing,
        seed::Int = 42,
    )
        n_chains >= 1 || error("n_chains must be >= 1, got $n_chains")
        results = Vector{MHResult}(undef, n_chains)

        Threads.@threads for c in 1:n_chains
            results[c] = lens_mh(logp_fn, lower, upper;
                                 n=n, step0=step0, adapt=adapt,
                                 init=init, seed=seed + c)
        end

        return results
    end

    # ══════════════════════════════════════════════════════════
    #  Multi-start MH — avoids local minima
    # ══════════════════════════════════════════════════════════

    """
        result = lens_mh_multistart(logp_fn, lower, upper; kw...)

    Evaluate `logp_fn` at `n_starts` random points, pick the best,
    run a short tuning chain to adapt step sizes, then run a full
    production chain.

    This is the recommended entry point for lens-model MCMC where
    the posterior may have multiple local minima.

    # Keyword arguments (same as `lens_mh`, plus):
      - `n_starts::Int = 8`:  number of random-start evaluations.
        Increase for complex posteriors (16–32).
      - `n_warmup::Int = max(500, n ÷ 2)`:  tuning steps to adapt
        step sizes near the best start point.
      - `n::Int = 2000`:  production chain length.
      - `seed::Int = 42`:  RNG seed.

    # Example
    ```julia
    result = lens_mh_multistart(logp_fn, [0.1, 0.1], [2.0, 5.0];
                                n_starts=16, n=3000, seed=42)
    ```
    """
    function lens_mh_multistart(
        logp_fn,
        lower::Vector{Float64},
        upper::Vector{Float64};
        n_starts::Int = 8,
        n_warmup::Int = 0,
        n::Int = 2000,
        step0::Union{Nothing, Vector{Float64}} = nothing,
        adapt::Bool = true,
        seed::Int = 42,
    )
        n_starts >= 1 || error("n_starts must be >= 1, got $n_starts")
        warmup_n = n_warmup > 0 ? n_warmup : max(500, n ÷ 2)

        best_lp = -Inf
        best_init = nothing

        for k in 1:n_starts
            init_k = lower .+ rand(length(lower)) .* (upper .- lower)
            lp_k = logp_fn(init_k)
            if lp_k > best_lp
                best_lp = lp_k
                best_init = copy(init_k)
            end
        end

        # Short tuning run to let step sizes adapt near the best point
        if adapt
            best_init = lens_mh(logp_fn, lower, upper;
                                n=warmup_n, step0=step0, adapt=true,
                                init=best_init, seed=seed + n_starts + 1).samples[:, end]
        end

        # Production chain from tuned start
        return lens_mh(logp_fn, lower, upper;
                       n=n, step0=step0, adapt=true,
                       init=best_init, seed=seed + n_starts + 2)
    end

    # ══════════════════════════════════════════════════════════
    #  Internal helpers
    # ══════════════════════════════════════════════════════════

    """
        _reflect!(x, lo, hi)

    Mirror-reflect each coordinate of `x` that is outside `[lo, hi]`.
    """
    function _reflect!(x::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64})
        @inbounds for j in eachindex(x)
            while true
                if x[j] < lo[j]
                    x[j] = 2lo[j] - x[j]
                elseif x[j] > hi[j]
                    x[j] = 2hi[j] - x[j]
                else
                    break
                end
            end
        end
        return x
    end

    # ══════════════════════════════════════════════════════════
    #  Convenience accessors
    # ══════════════════════════════════════════════════════════

    """
        S = chain(result::MHResult; burn=0)

    Extract the sample chain as a `(n_params, n_post)` Matrix after
    discarding the first `burn` samples.  Default `burn=0` returns
    all samples.
    """
    function chain(result::MHResult; burn::Int=0)
        burn = max(0, min(burn, size(result.samples, 2) - 1))
        return result.samples[:, burn+1:end]
    end

    """
        S = burnin(result::MHResult, n::Int)

    Drop the first `n` samples (warm-up).  Equivalent to
    `chain(result; burn=n)`.
    """
    burnin(result::MHResult, n::Int) = chain(result; burn=n)

    """
        (means, stds) = chain_stats(result::MHResult; burn=0)

    Return `(means::Vector{Float64}, stds::Vector{Float64})` of the
    posterior after `burn` samples.
    """
    function chain_stats(result::MHResult; burn::Int=0)
        S = chain(result; burn=burn)
        m = vec(mean(S; dims=2))
        s = vec(std(S; dims=2))
        return m, s
    end

    # ── Multi-chain dispatches ────────────────────────────

    """
        S = chain(chains::Vector{MHResult}; burn=0)

    Stack all chains into one `(n_params, total_post)` matrix.
    """
    function chain(chains::Vector{MHResult}; burn::Int=0)
        mats = [chain(c; burn=burn) for c in chains]
        return reduce(hcat, mats)
    end

    burnin(chains::Vector{MHResult}, n::Int) = chain(chains; burn=n)

    function chain_stats(chains::Vector{MHResult}; burn::Int=0)
        S = chain(chains; burn=burn)
        m = vec(mean(S; dims=2))
        s = vec(std(S; dims=2))
        return m, s
    end

end # module