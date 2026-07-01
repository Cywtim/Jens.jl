#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#  PSO + MCMC demo — standalone script
#
#  Particle Swarm Optimization for global mode finding,
#  followed by adaptive Metropolis-Hastings for refinement.
# ═══════════════════════════════════════════════════════════════

using Random, Statistics, Printf

# ═══════════════════════════════════════════════════════════════
#  PSO engine
# ═══════════════════════════════════════════════════════════════

"""
    gbest, gbest_lp, history = pso_search(
        logp_fn, lower, upper;
        n_particles=30, n_iter=80, seed=42)

Particle Swarm Optimization for bounded parameter search.

- `logp_fn(params::Vector{Float64}) -> Float64`: function to maximize
- `lower, upper::Vector{Float64}`: parameter bounds
"""
function pso_search(
    logp_fn,
    lower::Vector{Float64},
    upper::Vector{Float64};
    n_particles::Int = 30,
    n_iter::Int      = 80,
    w_start::Float64 = 0.9,
    w_end::Float64   = 0.4,
    c1::Float64      = 2.0,
    c2::Float64      = 2.0,
    seed::Int        = 42,
)
    d = length(lower)
    rng = MersenneTwister(seed)
    rng_pos = MersenneTwister(seed + 1000)

    rng_all = upper .- lower

    # Initialize
    X = zeros(d, n_particles)       # positions
    V = zeros(d, n_particles)       # velocities
    P = zeros(d, n_particles)       # personal bests
    pbest_lp = zeros(n_particles)

    for j in 1:n_particles
        X[:, j] .= lower .+ rand(rng_pos, d) .* rng_all
        V[:, j] .= (rand(rng_pos, d) .- 0.5) .* rng_all .* 0.1
        P[:, j] .= X[:, j]
        pbest_lp[j] = logp_fn(X[:, j])
    end

    best_idx = argmax(pbest_lp)
    gbest = copy(P[:, best_idx])
    gbest_lp = pbest_lp[best_idx]
    history = zeros(n_iter)

    # Main loop
    for iter in 1:n_iter
        w = w_start - (w_start - w_end) * (iter - 1) / max(n_iter - 1, 1)
        r1 = rand(rng_pos, d, n_particles)
        r2 = rand(rng_pos, d, n_particles)

        for j in 1:n_particles
            # Velocity
            V[:, j] .= w .* V[:, j] .+
                       c1 .* r1[:, j] .* (P[:, j] .- X[:, j]) .+
                       c2 .* r2[:, j] .* (gbest .- X[:, j])
            # Position
            X[:, j] .+= V[:, j]
            _reflect!(X[:, j], lower, upper)

            lp = logp_fn(X[:, j])
            if lp > pbest_lp[j]
                pbest_lp[j] = lp
                P[:, j] .= X[:, j]
                if lp > gbest_lp
                    gbest_lp = lp
                    gbest .= X[:, j]
                end
            end
        end
        history[iter] = gbest_lp
    end

    return gbest, gbest_lp, history
end

# ═══════════════════════════════════════════════════════════════
#  Adaptive M-H sampler
# ═══════════════════════════════════════════════════════════════

"""
    samples, accepted = mh_sample(logp_fn, lower, upper;
                                   n=5000, init=nothing, seed=42)

Adaptive Metropolis-Hastings with boundary reflection.
"""
function mh_sample(
    logp_fn,
    lower::Vector{Float64},
    upper::Vector{Float64};
    n::Int      = 5000,
    init        = nothing,
    seed::Int   = 42,
    adapt::Bool = true,
    n_adapt_delay::Int = 200,
)
    d = length(lower)
    rng = MersenneTwister(seed)

    theta = init === nothing ? (lower .+ upper) ./ 2 : copy(init)
    theta .= clamp.(theta, lower, upper)

    rng_all = upper .- lower
    step = rng_all ./ 50

    cur_lp = logp_fn(theta)
    accepted = 0

    samples = zeros(d, n)
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

        if adaptive_now
            acc_rate = accepted / i
            step .= step .* (1.0 .+ 0.01 .* (acc_rate - 0.234) .* randn(rng, d))
        end

        if adapt && i == n_adapt_delay
            adaptive_now = true
        end
    end

    return samples, accepted
end

# ── Boundary reflection ──
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

# ═══════════════════════════════════════════════════════════════
#  Demo: PSO + MCMC on SIS lens + Sersic + AGN
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
grid = Jens.gpu_grid(pix_n=64, pix_size=Float32(0.08))
psf  = GaussianPSF(; fwhm=0.12)
agn_images = [(0.206, -0.232), (-0.415, 0.311)]
sigma_data = 0.05f0

truth = Float64[0.8, 1.0, 0.3, 2.0, 5.0]
param_names = ["theta_E", "amp", "Rsersic", "n", "flux_agn"]

# Mock data from truth
lens_t = LensedPlane(CombinedLens(SIS => (theta_E=0.8f0, xcentre=0f0, ycentre=0f0));
                     z_lens, cosmology=cosmo)
host_t = ExtendedSource(SersicSpheric; amp=1f0, Rsersic=0.3f0, n=2f0, xcentre=0f0, ycentre=0f0)
agn_t  = PointImages((5f0, agn_images); intrinsic=true)
fwd_t  = ForwardModel(lens_plane=lens_t,
    source_plane=LightPlane(CompositeImage(host_t, agn_t); z=z_src),
    grid=grid, psf=psf)
data = render(fwd_t) .+ sigma_data .* CUDA.randn(Float32, size(render(fwd_t))...)

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
#  Run: PSO → MCMC
# ═══════════════════════════════════════════════════════════════

println("\n" * "="^60)
println("  Stage 1: PSO — swarm search")
println("="^60)

@time gbest, gbest_lp, pso_hist = pso_search(
    logp, lower, upper; n_particles=30, n_iter=100, seed=42)

println("PSO best logp: ", round(gbest_lp, digits=1))
println("PSO best:      ", round.(gbest, digits=4))

println("\n" * "="^60)
println("  Stage 2: M-H — refinement")
println("="^60)

@time samples, accepted = mh_sample(
    logp, lower, upper; n=5000, init=gbest, seed=123, n_adapt_delay=200)

burn = 1000
post = samples[:, burn+1:end]
post_mean = vec(mean(post; dims=2))
post_std  = vec(std(post; dims=2))

println("\n" * "="^60)
println("  Results")
println("="^60)
println(rpad("Param", 12), rpad("Truth", 10), rpad("PSO", 12), rpad("M-H Mean±Std", 22), "Δ/σ")
println("-"^65)
for i in 1:5
    ds = (post_mean[i] - truth[i]) / post_std[i]
    println(rpad(param_names[i], 12),
            rpad(string(truth[i]), 10),
            rpad(round(gbest[i], digits=3), 12),
            rpad(@sprintf("%.4f±%.4f", post_mean[i], post_std[i]), 22),
            round(ds, digits=2))
end

max_ds = maximum(abs.((post_mean[i]-truth[i])/post_std[i]) for i in 1:5)
println(max_ds < 3.0 ? "\n✓ All within 3σ" : "\n⚠ max Δ/σ = $(round(max_ds,digits=1))")
@printf("M-H accepted: %d/5000 (%.1f%%)\n", accepted, accepted/5000*100)