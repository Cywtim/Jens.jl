# ═══════════════════════════════════════════════════════════════
#  LensHMC — NUTS sampler with FiniteDiff gradients
#
#  Wraps AdvancedHMC.jl + FiniteDiff.jl behind the same API as
#  LensMH.jl:  lens_hmc(logp_fn, lower, upper; n=2000, ...)
# ═══════════════════════════════════════════════════════════════

module LensHMC

using Random, Statistics, LinearAlgebra
using AdvancedHMC, FiniteDiff, LogDensityProblems

export HMCResult, lens_hmc, chain, burnin, chain_stats

struct HMCResult
    samples::Matrix{Float64}
    logp::Vector{Float64}
    stats
end

# ── elementwise logit: bounded ↔ unbounded ──

function _to_u(p, lo, hi)
    u = similar(p)
    for i in eachindex(p)
        s = clamp((p[i]-lo[i])/(hi[i]-lo[i]), 1e-12, 1-1e-12)
        u[i] = log(s / (1 - s))
    end
    return u
end

function _to_x(u, lo, hi)
    p = similar(u)
    for i in eachindex(u)
        p[i] = lo[i] + (hi[i]-lo[i])/(1 + exp(-u[i]))
    end
    return p
end

function _logj(u, lo, hi)
    lj = 0.0
    for i in eachindex(u)
        s = 1/(1 + exp(-u[i]))
        lj += log(hi[i]-lo[i]) + log(s) + log(1-s)
    end
    return lj
end

# ── LogDensityProblems interface ──

struct LogDensityAdapter{F,Lo,Hi}
    logp_fn::F
    lower::Lo
    upper::Hi
end

# Evaluate: unconstrained → constrained → logp + Jacobian
function (ada::LogDensityAdapter)(u::AbstractVector)
    x = _to_x(u, ada.lower, ada.upper)
    return ada.logp_fn(x) + _logj(u, ada.lower, ada.upper)
end

# logdensity_and_gradient for AdvancedHMC
LogDensityProblems.logdensity(ada::LogDensityAdapter, u::AbstractVector) = ada(u)

function LogDensityProblems.logdensity_and_gradient(ada::LogDensityAdapter, u::AbstractVector)
    val = ada(u)
    grad = FiniteDiff.finite_difference_gradient(v -> ada(v), u)
    return val, grad
end

LogDensityProblems.dimension(ada::LogDensityAdapter) = length(ada.lower)

# Mark as gradient-capable
LogDensityProblems.capabilities(::Type{<:LogDensityAdapter}) = LogDensityProblems.LogDensityOrder{1}()

# ═══════════════════════════════════════════════════════════════

function lens_hmc(
    logp_fn,
    lower::Vector{Float64},
    upper::Vector{Float64};
    n::Int = 2000,
    n_warmup::Int = 500,
    init::Union{Nothing, Vector{Float64}} = nothing,
    seed::Int = 42,
)
    d = length(lower)
    @assert length(upper) == d
    Random.seed!(seed)

    # Initial unconstrained point
    theta0 = init === nothing ? (lower .+ upper) ./ 2 : copy(init)
    theta0 .= clamp.(theta0, lower .+ 1e-6, upper .- 1e-6)
    u0 = _to_u(theta0, lower, upper)

    # Build model + sampler
    model = LogDensityAdapter(logp_fn, lower, upper)
    sampler = NUTS(0.8)

    # Sample via AbstractMCMC
    chain_data = sample(Random.default_rng(), model, sampler, n;
                        n_adapts = n_warmup,
                        init_params = u0,
                        progress = false, verbose = false)

    # Extract samples and transform back
    n_total = length(chain_data)
    S = Matrix{Float64}(undef, d, n_total)
    L = Vector{Float64}(undef, n_total)
    for i in 1:n_total
        u = chain_data[i].z.θ          # unconstrained parameter from PhasePoint
        x = _to_x(u, lower, upper)
        S[:, i] .= x
        L[i] = logp_fn(x)
    end

    return HMCResult(S, L, nothing)
end

# ── accessors ──

chain(r::HMCResult; burn::Int=0) =
    r.samples[:, max(0, min(burn, size(r.samples,2)-1))+1:end]
burnin(r::HMCResult, n::Int) = chain(r; burn=n)

function chain_stats(r::HMCResult; burn::Int=0)
    S = chain(r; burn=burn)
    return vec(mean(S; dims=2)), vec(std(S; dims=2))
end

end # module