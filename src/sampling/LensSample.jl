# ═══════════════════════════════════════════════════════════════
#  LensSample — Two-stage sampler: HMC exploration + M-H refinement
#
#  Stage 1: HMC (NUTS) with few samples to locate posterior mode
#  Stage 2: M-H with narrow bounds centred on HMC posterior
#
#  Result: HMC's robustness + M-H's speed per sample.
# ═══════════════════════════════════════════════════════════════

module LensSample

using Statistics
using Jens.LensMH: lens_mh, MHResult
import Jens.LensMH: chain, chain_stats
using Jens.LensHMC: lens_hmc, HMCResult
import Jens.LensHMC: chain as hmc_chain

export LensSampleResult, lens_sample, chain, chain_stats

"""
    LensSampleResult

Result of two-stage sampling.

Fields:
  - hmc_result::HMCResult       raw HMC output (exploration stage)
  - mh_result::MHResult         raw M-H output (refinement stage)
  - hmc_mean::Vector{Float64}   posterior mean from HMC
  - hmc_std::Vector{Float64}    posterior std from HMC
  - refined_lower::Vector{Float64}  narrowed bounds for M-H stage
  - refined_upper::Vector{Float64}
"""
struct LensSampleResult
    hmc_result::HMCResult
    mh_result::MHResult
    hmc_mean::Vector{Float64}
    hmc_std::Vector{Float64}
    refined_lower::Vector{Float64}
    refined_upper::Vector{Float64}
end

# ── Accessors: default to M-H (refinement) chain ──

chain(r::LensSampleResult; burn::Int=0) = chain(r.mh_result; burn=burn)
chain_stats(r::LensSampleResult; burn::Int=0) = chain_stats(r.mh_result; burn=burn)

"""
    result = lens_sample(logp_fn, lower, upper;
                         n_hmc=300, hmc_warmup=100,
                         n_mh=5000, mh_burn=1000,
                         n_sigma=5.0, seed=42)

Two-stage sampling: HMC exploration → M-H refinement.

# Arguments
- `lower, upper`: prior bounds (used for both stages)
- `n_hmc`: HMC samples (exploration, default 300)
- `hmc_warmup`: HMC warmup/adaptation steps (default 100)
- `n_mh`: M-H samples (refinement, default 5000)
- `mh_burn`: M-H burn-in to discard (default 1000)
- `n_sigma`: M-H bounds = HMC mean ± n_sigma × HMC std
- `seed`: random seed

# Returns
`LensSampleResult` with both chains.

# Example
```julia
result = lens_sample(logp, [0.3, 0.2, 0.05], [2.0, 5.0, 1.5];
                     n_hmc=300, n_mh=5000)

# Access refinement chain (M-H)
post = chain(result; burn=1000)
means, stds = chain_stats(result; burn=1000)

# Access exploration chain (HMC) for diagnostics
hmc_post = hmc_chain(result.hmc_result; burn=100)
```
"""
function lens_sample(
    logp_fn,
    lower::Vector{Float64},
    upper::Vector{Float64};
    n_hmc::Int = 300,
    hmc_warmup::Int = 100,
    n_mh::Int = 5000,
    mh_burn::Int = 1000,
    n_sigma::Float64 = 5.0,
    seed::Int = 42,
)
    d = length(lower)
    @assert length(upper) == d

    # ═══════════════════════════════════════════════════════
    #  Stage 1: HMC exploration
    # ═══════════════════════════════════════════════════════

    hmc_result = lens_hmc(logp_fn, lower, upper;
                          n=n_hmc, n_warmup=hmc_warmup, seed=seed)

    # Discard warmup, extract mean/std (use LensHMC.chain for HMCResult)
    hmc_post = hmc_chain(hmc_result; burn=hmc_warmup)
    hmc_mean = vec(mean(hmc_post; dims=2))
    hmc_std  = vec(std(hmc_post; dims=2))

    # ── Guard: if HMC produced degenerate std, fall back ──
    for i in 1:d
        if !isfinite(hmc_std[i]) || hmc_std[i] <= 0
            hmc_std[i] = (upper[i] - lower[i]) / 10
        end
    end

    # ═══════════════════════════════════════════════════════
    #  Stage 2: M-H refinement with narrowed bounds
    # ═══════════════════════════════════════════════════════

    refined_lower = hmc_mean .- n_sigma .* hmc_std
    refined_upper = hmc_mean .+ n_sigma .* hmc_std

    # Clamp to original bounds
    refined_lower .= max.(refined_lower, lower)
    refined_upper .= min.(refined_upper, upper)

    # Clamp init to refined bounds
    init_mh = clamp.(hmc_mean, refined_lower, refined_upper)

    mh_result = lens_mh(logp_fn, refined_lower, refined_upper;
                        n=n_mh, init=init_mh, adapt=true,
                        seed=seed + 1, n_adapt_delay=200)

    return LensSampleResult(
        hmc_result, mh_result,
        hmc_mean, hmc_std,
        refined_lower, refined_upper,
    )
end

end # module