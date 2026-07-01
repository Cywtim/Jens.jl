# ═══════════════════════════════════════════════════════════════
#  LensPSO — Particle Swarm Optimization + M-H refinement
#
#  Stage 1: PSO — cooperative global search (N particles share info)
#  Stage 2: M-H — adaptive refinement from PSO optimum
#
#  Unlike multi-start (independent random probes), PSO particles
#  communicate: each moves toward its personal best AND the swarm's
#  global best.  Typically needs fewer evaluations than multi-start
#  for 5+ dimensional, correlated posteriors.
# ═══════════════════════════════════════════════════════════════

module LensPSO

using Random, Statistics
using Jens.LensMH: lens_mh, MHResult, chain, chain_stats
import Jens.LensMH: chain as mh_chain_fn  # for clarity

export PSOResult, lens_pso_mh, chain, chain_stats

"""
    PSOResult

Fields:
  - gbest::Vector{Float64}           global best position from PSO
  - gbest_logp::Float64              log-posterior at gbest
  - pso_history::Vector{Float64}     best logp per PSO iteration
  - final_particles::Matrix{Float64} (n_params, n_particles) final positions
  - mh_result::MHResult              M-H refinement chain
"""
struct PSOResult
    gbest::Vector{Float64}
    gbest_logp::Float64
    pso_history::Vector{Float64}
    final_particles::Matrix{Float64}
    mh_result::MHResult
end

# ── Accessors: default to M-H chain ──
chain(r::PSOResult; burn::Int=0) = chain(r.mh_result; burn=burn)
chain_stats(r::PSOResult; burn::Int=0) = chain_stats(r.mh_result; burn=burn)

"""
    result = lens_pso_mh(logp_fn, lower, upper;
                         n_particles=30, n_iter=80,
                         n_mh=5000, mh_burn=1000,
                         seed=42)

PSO global search → M-H refinement.

# PSO parameters
- `n_particles`: swarm size (default 30). Use 20-50.
- `n_iter`: PSO iterations (default 80). Use 50-200.
- `w_start, w_end`: inertia weight range (0.9 → 0.4)
- `c1`: cognitive weight (personal best attraction, default 2.0)
- `c2`: social weight (global best attraction, default 2.0)

# M-H parameters  
- `n_mh`: production M-H samples (default 5000)
- `mh_burn`: M-H burn-in (default 1000)

# Example
```julia
result = lens_pso_mh(logp, lower, upper;
    n_particles=30, n_iter=80, n_mh=5000)
post = chain(result; burn=1000)
```
"""
function lens_pso_mh(
    logp_fn,
    lower::Vector{Float64},
    upper::Vector{Float64};
    # PSO
    n_particles::Int = 30,
    n_iter::Int = 80,
    w_start::Float64 = 0.9,
    w_end::Float64   = 0.4,
    c1::Float64 = 2.0,
    c2::Float64 = 2.0,
    # M-H
    n_mh::Int = 5000,
    mh_burn::Int = 1000,
    seed::Int = 42,
)
    d = length(lower)
    @assert length(upper) == d
    rng = MersenneTwister(seed)
    rng_pso = MersenneTwister(seed + 1000)  # separate stream for PSO

    range_all = upper .- lower

    # ═══════════════════════════════════════════════════════
    #  Stage 1: PSO
    # ═══════════════════════════════════════════════════════

    # ── Initialize particles uniformly in bounds ──
    X = zeros(d, n_particles)  # positions
    V = zeros(d, n_particles)  # velocities
    P = zeros(d, n_particles)  # personal bests
    pbest_lp = Vector{Float64}(undef, n_particles)
    pso_hist = Vector{Float64}(undef, n_iter)

    for j in 1:n_particles
        for i in 1:d
            X[i, j] = lower[i] + rand(rng_pso) * (upper[i] - lower[i])
        end
        V[:, j] .= (rand(rng_pso, d) .- 0.5) .* range_all .* 0.1
        P[:, j] .= X[:, j]
        pbest_lp[j] = logp_fn(P[:, j])
    end

    # Global best
    best_idx = argmax(pbest_lp)
    gbest = copy(P[:, best_idx])
    gbest_lp = pbest_lp[best_idx]

    # ── PSO main loop ──
    for iter in 1:n_iter
        w = w_start - (w_start - w_end) * (iter - 1) / max(n_iter - 1, 1)

        for j in 1:n_particles
            r1 = rand(rng_pso, d)
            r2 = rand(rng_pso, d)

            # Velocity update
            V[:, j] .= w .* V[:, j] .+
                       c1 .* r1 .* (P[:, j] .- X[:, j]) .+
                       c2 .* r2 .* (gbest .- X[:, j])

            # Position update
            X[:, j] .+= V[:, j]

            # Reflect at boundaries
            _reflect_pso!(X[:, j], lower, upper)

            # Evaluate
            lp = logp_fn(X[:, j])

            # Update personal best
            if lp > pbest_lp[j]
                pbest_lp[j] = lp
                P[:, j] .= X[:, j]

                # Update global best
                if lp > gbest_lp
                    gbest_lp = lp
                    gbest .= X[:, j]
                end
            end
        end

        pso_hist[iter] = gbest_lp
    end

    # ═══════════════════════════════════════════════════════
    #  Stage 2: M-H refinement from PSO optimum
    # ═══════════════════════════════════════════════════════

    mh_result = lens_mh(logp_fn, lower, upper;
                        n=n_mh, init=gbest, adapt=true,
                        seed=seed + 2000, n_adapt_delay=200)

    return PSOResult(gbest, gbest_lp, pso_hist, X, mh_result)
end

# ── Boundary reflection (same logic as LensMH._reflect!) ──
function _reflect_pso!(x::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64})
    @inbounds for i in eachindex(x)
        while true
            if x[i] < lo[i]
                x[i] = 2lo[i] - x[i]
            elseif x[i] > hi[i]
                x[i] = 2hi[i] - x[i]
            else
                break
            end
        end
    end
    return x
end

end # module