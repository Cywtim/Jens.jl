#!/usr/bin/env julia
#  Turing @model + manual MH loop — optimized
#  Usage:   julia --project=. examples/demo_turing_mh.jl

push!(LOAD_PATH, "@stdlib")

using Cosmology, LinearAlgebra, Distributions, Statistics, Random
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Jens.LensPSF: GaussianPSF
using Turing, DynamicPPL
import Random; Random.seed!(42)

# ── Generate data ─────────────────────────────────────────────
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens, z_src = 0.3, 1.5
grid = GenGrid(pix_n=64, pix_size=0.16)
psf  = GaussianPSF(; fwhm=3.0)

true_params = (theta_E=0.8, amp=1.0, Rsersic=0.3)
lens_mass = CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0))
lp = LensedPlane(lens_mass; z_lens, cosmology=cosmo)
host = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)
sys = ForwardModel(lens_plane=lp, source_plane=LightPlane(host; z=z_src), grid=grid, psf=psf)
data = render(sys)
sigma_noise = max(median(abs.(data)) * 0.05, 1e-3)
data .+= sigma_noise * randn(size(data))

println("Grid: $(size(data)),  σ≈$(round(sigma_noise, digits=5))")
println("True: θE=$(true_params.theta_E), amp=$(true_params.amp), Rsersic=$(true_params.Rsersic)")

# ── @model ────────────────────────────────────────────────────
@model function simple_fit(data)
    theta_E ~ Uniform(0.2, 2.0)
    amp     ~ Uniform(0.1, 5.0)
    Rsersic ~ Uniform(0.05, 1.0)

    m = CombinedLens(SIS => (theta_E=theta_E, xcentre=0.0, ycentre=0.0))
    lp = LensedPlane(m; z_lens, cosmology=cosmo)
    h = ExtendedSource(SersicSpheric; amp=amp, Rsersic=Rsersic, n=2.0, xcentre=0.0, ycentre=0.0)
    s = ForwardModel(lens_plane=lp, source_plane=LightPlane(h; z=z_src), grid=grid, psf=psf)
    img = render(s; solver=:batch)
    for i in eachindex(data)
        data[i] ~ Normal(img[i], sigma_noise)
    end
end

# ══════════════════════════════════════════════════════════════
#  Adaptive MH
# ══════════════════════════════════════════════════════════════

model = simple_fit(data)
vi = DynamicPPL.VarInfo(model)
vn_theta = @varname(theta_E)
vn_amp   = @varname(amp)
vn_rs    = @varname(Rsersic)

function run_adaptive_mh!(samples, vi, model, n, step0)
    θ = [0.8, 1.0, 0.3]
    step = copy(step0)
    vi = DynamicPPL.setindex!!(vi, θ[1], vn_theta)
    vi = DynamicPPL.setindex!!(vi, θ[2], vn_amp)
    vi = DynamicPPL.setindex!!(vi, θ[3], vn_rs)
    current_lp = DynamicPPL.logjoint(model, vi)
    accepted = 0

    for i in 1:n
        prop = θ .+ step .* randn(3)
        vi2 = deepcopy(vi)
        vi2 = DynamicPPL.setindex!!(vi2, prop[1], vn_theta)
        vi2 = DynamicPPL.setindex!!(vi2, prop[2], vn_amp)
        vi2 = DynamicPPL.setindex!!(vi2, prop[3], vn_rs)
        prop_lp = DynamicPPL.logjoint(model, vi2)
        α = prop_lp - current_lp

        if α >= 0 || rand() < exp(α)
            θ = prop
            vi = vi2
            current_lp = prop_lp
            accepted += 1
            # Increase step on accept (target ~40%)
            step .*= 1.02
        else
            # Decrease step on reject
            step .*= 0.98
        end
        samples[i, :] .= θ
    end
    return accepted, step
end

# ── Run ───────────────────────────────────────────────────────
n = 2000
samples = zeros(n, 3)
step0 = [0.003, 0.01, 0.003]  # ~0.5% of prior width

println("\n=== Adaptive MH × $n ===")
t = @elapsed accepted, step_final = run_adaptive_mh!(samples, vi, model, n, step0)
println("Time: $(round(t, digits=1)) s  ($(round(t/n*1000, digits=1)) ms/step)")
println("Acceptance: $(accepted)/$n = $(round(accepted/n*100, digits=1))%")
println("Final step sizes: $(round.(step_final, digits=5))")

# ── Results ───────────────────────────────────────────────────
burn = accepted ÷ 3
post = samples[burn:end, :]
println("\n=== Posterior (burn=$burn) ===")
names_ = ["theta_E", "amp", "Rsersic"]
for (i, (name, tv)) in enumerate(zip(names_, values(true_params)))
    s = post[:, i]
    m, σ = mean(s), std(s)
    σ_str = σ > 1e-9 ? "±$(round(σ, digits=5))" : "(stuck)"
    b = σ > 1e-9 ? "$(round((m-tv)/σ, digits=1))σ" : "—"
    println("  $name = $(round(m,digits=5)) $σ_str  (true=$tv, $b)")
end
println("Done.")