#!/usr/bin/env julia
#  LensMH smoke test — standalone MH, no Turing dependency
#  Usage:   julia --project=. examples/demo_lens_mh.jl

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
using Jens.LensMH

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

# ══════════════════════════════════════════════════════════════
#  Pure logp_fn — no Turing, no DynamicPPL
# ══════════════════════════════════════════════════════════════

function my_logp(p::Vector{Float64})
    theta_E, amp, Rsersic = p

    # Prior (uniform, already handled by bounds reflection)
    log_prior = 0.0

    # Forward model
    m = CombinedLens(SIS => (theta_E=theta_E, xcentre=0.0, ycentre=0.0))
    lp2 = LensedPlane(m; z_lens, cosmology=cosmo)
    h = ExtendedSource(SersicSpheric; amp=amp, Rsersic=Rsersic, n=2.0, xcentre=0.0, ycentre=0.0)
    s = ForwardModel(lens_plane=lp2, source_plane=LightPlane(h; z=z_src), grid=grid, psf=psf)
    img = render(s; solver=:batch)

    # Gaussian log-likelihood
    diff = data .- img
    log_lik = -sum(diff .^ 2) / (2 * sigma_noise^2)

    return log_prior + log_lik
end

# ══════════════════════════════════════════════════════════════
#  Run
# ══════════════════════════════════════════════════════════════

lower = [0.2, 0.1, 0.05]
upper = [2.0, 5.0, 1.0]
init  = [0.8, 1.0, 0.3]

println("\n=== LensMH.lens_mh × 2000 ===")
t = @elapsed result = lens_mh(my_logp, lower, upper; n=2000, init=init, seed=42)
println("Time: $(round(t, digits=1)) s")
println("Acceptance: $(result.accepted)/2000 = $(round(result.accepted/20, digits=1))%")
println("Final step: $(round.(result.step_final, digits=5))")

# ── Posterior ─────────────────────────────────────────────────
burn = result.accepted ÷ 3
post = result.samples[:, burn:end]
println("\n=== Posterior (burn=$burn) ===")
for (i, name) in enumerate(["theta_E", "amp", "Rsersic"])
    s = post[i, :]
    m, σ = mean(s), std(s)
    tv = [0.8, 1.0, 0.3][i]
    b = round((m - tv) / max(σ, 1e-9), digits=1)
    println("  $name = $(round(m,digits=5)) ± $(round(σ,digits=5))  (true=$tv, $(b)σ)")
end
println("Done.")