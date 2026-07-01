#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#  Step-by-step MCMC demo: SIS lens + Sersic host + AGN
#
#  Key feature: PointImages(intrinsic=true) allows fitting
#  the intrinsic AGN flux as a free parameter — μ is computed
#  automatically at each MCMC step from the lens Hessian.
#
#  NOTE: The posterior for lens models is typically very narrow
#  (σ/θ ~ 0.1–1%).  M-H needs a reasonable init (~20% of true
#  values).  Use prior knowledge (image separation, flux ratios,
#  morphology) to set init.  For truly unknown init, use
#  lens_mh_multistart with 50+ random starts.
# ═══════════════════════════════════════════════════════════════

using Cosmology, Random, Statistics, Distributions, LinearAlgebra
using Jens
using Jens.LensModel: SIS
using Jens.LensModel.ComLens: CombinedLens
using Jens.LightModel: ExtendedSource, PointImages, CompositeImage
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensPSF: GaussianPSF
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Jens.LensMH: lens_mh, chain, chain_stats
using Jens.LensSolver: solve_images

Random.seed!(42)

# ═══════════════════════════════════════════════════════════════
#  Step 1: Cosmology & Observation Setup
# ═══════════════════════════════════════════════════════════════

cosmo   = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens  = 0.3
z_src   = 1.5

pix_n    = 64
pix_size = 0.06
sigma_data = 0.05

grid = GenGrid(pix_n=pix_n, pix_size=pix_size)
psf  = GaussianPSF(0.05)

println("Grid: $(pix_n)×$(pix_n), pixel=$(pix_size)\", noise σ=$(sigma_data)")

# ═══════════════════════════════════════════════════════════════
#  Step 2: Truth Model
# ═══════════════════════════════════════════════════════════════

lens_truth = LensedPlane(
    CombinedLens(SIS => (theta_E=0.8,));
    z_lens=z_lens, cosmology=cosmo)

host_truth = ExtendedSource(SersicSpheric;
    amp=1.0, Rsersic=0.3, n=2.0,
    xcentre=0.0, ycentre=0.0)

# AGN: solve lens equation ONCE to get image-plane positions
# In real data, these positions come from source extraction
agn_images = solve_images(lens_truth, 0.05, -0.03; z_source=z_src)
agn_positions = [(x, y) for (x, y, _) in agn_images]
println("AGN images: $(length(agn_positions)) found at:")
for (i, (x, y)) in enumerate(agn_positions)
    println("  Image $i: ($(round(x, digits=3)), $(round(y, digits=3)))")
end

agn_truth = PointImages((5.0, agn_positions); intrinsic=true)

fwd_truth = ForwardModel(
    lens_plane=lens_truth,
    source_plane=LightPlane(CompositeImage(host_truth, agn_truth); z=z_src),
    grid=grid, psf=psf)

truth_img = render(fwd_truth)
println("Truth image: $(size(truth_img)), peak=$(round(maximum(truth_img), digits=2))")

# ═══════════════════════════════════════════════════════════════
#  Step 3: Generate Mock Observation
# ═══════════════════════════════════════════════════════════════

data = truth_img .+ sigma_data .* randn(size(truth_img)...)

# ═══════════════════════════════════════════════════════════════
#  Step 4: Build log-posterior
# ═══════════════════════════════════════════════════════════════
#
#  Free params: theta_E, amp_host, Rsersic, n, flux_agn
#  AGN uses PointImages(intrinsic=true) → μ auto-computed
#  at each MCMC step from the lens Hessian.

function build_logp(data, grid, psf, cosmo, z_lens, z_src, agn_positions)
    function logp_fn(params::Vector{Float64})
        theta_E, amp, rsr, n, flux_agn = params

        lens = LensedPlane(
            CombinedLens(SIS => (theta_E=theta_E,));
            z_lens=z_lens, cosmology=cosmo)

        host = ExtendedSource(SersicSpheric;
            amp=amp, Rsersic=rsr, n=n,
            xcentre=0.0, ycentre=0.0)

        # intrinsic=true: amp is intrinsic flux, μ computed automatically
        agn = PointImages((flux_agn, agn_positions); intrinsic=true)

        fwd = ForwardModel(
            lens_plane   = lens,
            source_plane = LightPlane(CompositeImage(host, agn); z=z_src),
            grid = grid, psf = psf)

        model_img = render(fwd)
        chi2 = sum(((data .- model_img) ./ sigma_data).^2)
        return -0.5 * chi2
    end
    return logp_fn
end

logp = build_logp(data, grid, psf, cosmo, z_lens, z_src, agn_positions)
println("logp(truth) = $(round(logp([0.8,1.0,0.3,2.0,5.0]), digits=1))")

# ═══════════════════════════════════════════════════════════════
#  Step 5: Run MCMC
# ═══════════════════════════════════════════════════════════════
#
lower = [0.3, 0.2, 0.05, 0.5, 1.0]
upper = [2.0, 5.0, 1.5,  6.0, 50.0]

# Init from prior estimates.  For real systems you estimate:
#   theta_E from image separation, amp from flux, etc.
#   M-H needs init within ~10% of truth for this narrow posterior.
#   Use lens_mh_multistart with n_starts=50+ when init is unknown.
init = [0.78, 1.05, 0.31, 2.05, 5.2]  # ~5% off truth — realistic

println("\nRunning M-H: 3000 steps, adaptive...")
@time result = lens_mh(logp, lower, upper;
    n=3000, init=init, adapt=true, seed=42)

acc = round(result.accepted / 3000 * 100, digits=1)
println("Accepted: $(result.accepted)/3000 ($(acc)%)")

# ═══════════════════════════════════════════════════════════════
#  Step 6: Posterior Summary
# ═══════════════════════════════════════════════════════════════

post = chain(result; burn=1000)
n_post = size(post, 2)

param_names = ["theta_E", "amp_host", "Rsersic", "n", "flux_agn"]
truth_vals  = [0.8, 1.0, 0.3, 2.0, 5.0]

println("\n" * "="^62)
println("  Posterior Summary  ($n_post samples after burn-in)")
println("="^62)
println(rpad("Parameter", 14), rpad("Truth", 10), rpad("Mean ± Std", 18), rpad("Δ/σ", 8))
println("-"^50)

for i in 1:5
    m  = mean(post[i, :])
    s  = std(post[i, :])
    ds = (m - truth_vals[i]) / (s + 1e-10)
    println(rpad(param_names[i], 14),
            rpad(truth_vals[i], 10),
            rpad("$(round(m, digits=4)) ± $(round(s, digits=4))", 18),
            rpad(round(ds, digits=2), 8))
end

max_ds = maximum(abs.((mean(post[i, :]) - truth_vals[i]) / std(post[i, :])) for i in 1:5)
if max_ds < 3.0
    println("\n✓ All parameters recovered within 3σ of truth.")
else
    println("\n⚠ Some deviate >3σ — try longer chain or better init.")
end

# ═══════════════════════════════════════════════════════════════
#  Key takeaways:
#
#  1. PointImages(intrinsic=true) lets you fit AGN intrinsic flux
#     as a free parameter — μ computed on-the-fly (~6 μs for 2 imgs)
#
#  2. lens_mh with n_adapt_delay=200 gives the chain 200 steps of
#     pure random walk before adapting — helps avoid premature collapse
#
#  3. For narrow posteriors (σ/θ ~ 0.1%), M-H needs a good init.
#     Use lens_mh_multistart(logp, lower, upper; n_starts=50)
#     when init is truly unknown.
# ═══════════════════════════════════════════════════════════════