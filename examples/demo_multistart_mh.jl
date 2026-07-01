#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#  Multi-Start M-H sampler — standalone script
#
#  16 random starts → evaluate logp → pick best →
#  short warmup chain → full adaptive M-H from best point.
#
#  Best for: 5-10 parameter, narrow-unimodal posteriors
#  where logp evaluation is fast (< 1ms).
# ═══════════════════════════════════════════════════════════════

using Random, Statistics, Printf

# ═══════════════════════════════════════════════════════════════
#  Core M-H sampler (adaptive, bounded)
# ═══════════════════════════════════════════════════════════════

"""
    samples, accepted, lp_hist = mh_sample(
        logp_fn, lower, upper;
        n=5000, init=nothing, step0=nothing,
        adapt=true, n_adapt_delay=200, seed=42)

Adaptive Metropolis-Hastings with boundary reflection.

- `logp_fn(params::Vector{Float64}) -> Float64`: log-posterior
- `lower, upper`: parameter bounds
- `n`: number of samples
- `adapt`: enable adaptive step-size tuning (target 23.4% acceptance)
- `n_adapt_delay`: freeze step size for first N iterations
"""
function mh_sample(
    logp_fn,
    lower::Vector{Float64},
    upper::Vector{Float64};
    n::Int              = 5000,
    init                = nothing,
    step0               = nothing,
    adapt::Bool         = true,
    n_adapt_delay::Int  = 200,
    seed::Int           = 42,
)
    d = length(lower)
    rng = MersenneTwister(seed)

    # Init
    theta = init === nothing ? (lower .+ upper) ./ 2 : copy(init)
    theta .= clamp.(theta, lower, upper)
    rng_all = upper .- lower
    step = step0 === nothing ? rng_all ./ 50 : copy(step0)

    cur_lp = logp_fn(theta)
    accepted = 0
    samples  = zeros(d, n)
    lp_hist  = zeros(n)
    adaptive_now = adapt && n_adapt_delay <= 0

    for i in 1:n
        prop = theta .+ step .* randn(rng, d)
        _reflect!(prop, lower, upper)

        prop_lp = logp_fn(prop)
        alpha = min(1.0, exp(prop_lp - cur_lp))

        if rand(rng) < alpha
            theta .= prop
            cur_lp = prop_lp
            accepted += 1
        end

        samples[:, i] .= theta
        lp_hist[i] = cur_lp

        if adaptive_now
            acc_rate = accepted / i
            # Target 23.4% (optimal for Gaussian targets in d>1)
            step .= step .* exp.(0.01 .* (acc_rate - 0.234) .* randn(rng, d))
        end

        if adapt && i == n_adapt_delay
            adaptive_now = true
        end
    end

    return samples, accepted, lp_hist
end

# ═══════════════════════════════════════════════════════════════
#  Multi-start: random probes → warmup → production chain
# ═══════════════════════════════════════════════════════════════

"""
    samples, accepted, lp_hist = multistart_mh(
        logp_fn, lower, upper;
        n_starts=16, n_warmup=500, n=5000,
        adapt=true, n_adapt_delay=200, seed=42)

Multi-start adaptive M-H.

1. Try `n_starts` random points, pick the one with highest logp.
2. Short warmup chain from that point (adapt step sizes).
3. Full production chain from warmup's final position.
"""
function multistart_mh(
    logp_fn,
    lower::Vector{Float64},
    upper::Vector{Float64};
    n_starts::Int       = 16,
    n_warmup::Int       = 500,
    n::Int              = 5000,
    adapt::Bool         = true,
    n_adapt_delay::Int  = 200,
    seed::Int           = 42,
)
    d = length(lower)
    rng = MersenneTwister(seed)

    # ── Stage 1: random probes ──
    best_lp = -Inf
    best_pt = zeros(d)

    for k in 1:n_starts
        pt = lower .+ rand(rng, d) .* (upper .- lower)
        lp = logp_fn(pt)
        if lp > best_lp
            best_lp = lp
            best_pt .= pt
        end
    end

    # ── Stage 2: warmup chain ──
    warmup_samples, _, _ = mh_sample(
        logp_fn, lower, upper;
        n=n_warmup, init=best_pt, adapt=adapt,
        n_adapt_delay=0, seed=seed + n_starts + 1)
    warmup_final = warmup_samples[:, end]

    # ── Stage 3: production chain ──
    return mh_sample(
        logp_fn, lower, upper;
        n=n, init=warmup_final, adapt=adapt,
        n_adapt_delay=n_adapt_delay, seed=seed + n_starts + 2)
end

# ═══════════════════════════════════════════════════════════════
#  Helpers
# ═══════════════════════════════════════════════════════════════

function _reflect!(x, lo, hi)
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

"""
    post_mean, post_std = summarize(samples; burn=1000)

Discard `burn` samples, return posterior mean and std.
"""
function summarize(samples::Matrix{Float64}; burn::Int=1000)
    post = samples[:, max(1, burn+1):end]
    return vec(mean(post; dims=2)), vec(std(post; dims=2))
end

# ═══════════════════════════════════════════════════════════════
#  Demo: SIS lens + Sersic host + AGN point source
# ═══════════════════════════════════════════════════════════════

using CUDA, Cosmology
using Jens
using Jens.LensModel: SIS
using Jens.LensModel.ComLens: CombinedLens
using Jens.LightModel: ExtendedSource, PointImages, CompositeImage
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensPSF: GaussianPSF
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render

Random.seed!(42)

# ── Setup ──
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens, z_src = 0.3, 1.5
grid = Jens.gpu_grid(pix_n=128, pix_size=Float32(0.08))
psf  = GaussianPSF(; fwhm=0.12)
agn_images = [(0.206, -0.232), (-0.415, 0.311)]
sigma_data = 0.05f0

truth = Float64[0.8, 1.0, 0.3, 2.0, 5.0]
param_names = ["theta_E", "amp", "Rsersic", "n", "flux_agn"]

# ── Mock data ──
lens_t = LensedPlane(CombinedLens(SIS => (theta_E=0.8f0, xcentre=0f0, ycentre=0f0));
                     z_lens, cosmology=cosmo)
host_t = ExtendedSource(SersicSpheric; amp=1f0, Rsersic=0.3f0, n=2f0, xcentre=0f0, ycentre=0f0)
agn_t  = PointImages((5f0, agn_images); intrinsic=true)
fwd_t  = ForwardModel(lens_plane=lens_t,
    source_plane=LightPlane(CompositeImage(host_t, agn_t); z=z_src),
    grid=grid, psf=psf)
truth_img = render(fwd_t)
data = truth_img .+ sigma_data .* CUDA.randn(Float32, size(truth_img)...)

# ── log-posterior ──
function make_logp(data, grid, psf, cosmo, z_lens, z_src, agn_images)
    return function logp_fn(p::Vector{Float64})
        thE, amp, rsr, n, flux = p
        lens = LensedPlane(CombinedLens(SIS => (theta_E=thE, xcentre=0.0, ycentre=0.0));
                           z_lens, cosmology=cosmo)
        host = ExtendedSource(SersicSpheric; amp=amp, Rsersic=rsr, n=n, xcentre=0.0, ycentre=0.0)
        agn  = PointImages((flux, agn_images); intrinsic=true)
        fwd  = ForwardModel(lens_plane=lens,
            source_plane=LightPlane(CompositeImage(host, agn); z=z_src),
            grid=grid, psf=psf)
        img  = render(fwd)
        chi2 = sum(((data .- img) ./ sigma_data).^2)
        return Float64(-0.5 * chi2)
    end
end

logp = make_logp(data, grid, psf, cosmo, z_lens, z_src, agn_images)
@printf("logp(truth) = %.1f\n", logp(truth))

lower = Float64[0.3, 0.2, 0.05, 0.5, 1.0]
upper = Float64[2.0, 5.0, 1.5,  6.0, 50.0]

# ═══════════════════════════════════════════════════════════════
#  Run
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("  Multi-Start M-H — 128×128, 16 starts, 5000 samples")
println("="^60)

@time samples, accepted, lp_hist = multistart_mh(
    logp, lower, upper; n_starts=16, n_warmup=500, n=5000, seed=42)

# ── Summary ──
post_mean, post_std = summarize(samples; burn=1000)
n_post = size(samples, 2) - 1000

println(rpad("\nParam", 12), rpad("Truth", 10), rpad("Mean±Std", 22), "Δ/σ")
println("-"^55)
for i in 1:5
    ds = (post_mean[i] - truth[i]) / post_std[i]
    println(rpad(param_names[i], 12),
            rpad(string(truth[i]), 10),
            rpad(@sprintf("%.4f±%.4f", post_mean[i], post_std[i]), 22),
            round(ds, digits=2))
end
max_ds = maximum(abs.((post_mean[i] - truth[i]) / post_std[i]) for i in 1:5)
println(max_ds < 3.0 ? "\n✓ All within 3σ" : "\n⚠ max Δ/σ = $(round(max_ds,digits=1))")
@printf("M-H accepted: %d/%d (%.1f%%)\n", accepted, 5000, accepted/5000*100)
@printf("Posterior samples (post-burn): %d\n", n_post)