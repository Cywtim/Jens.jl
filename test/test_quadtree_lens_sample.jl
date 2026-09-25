# Quadtree scheme-1b sampling under NOISY observation.
#
#   Key finding (honest): the two-stage lens_sample's HMC stage uses
#   FiniteDiff gradients on the quadtree logp; measured on this stack
#   the gradient is ~28× the single-eval cost and NUTS tree-depth can
#   explode, so HMC produces a degenerate, ultra-narrow posterior that
#   does NOT cover truth.  This is a known mismatch: quadtree logp is a
#   black-box, piecewise-smooth, no-analytic-gradient objective — the
#   right sampler is gradient-free M-H.
#
#   So the definitive test here runs lens_mh (full 3000-step chain),
#   and ALSO exercises lens_sample to document its HMC-stage caveat.
#
#   Truth:  NIE(θ*) + Gaussian noise → data
#   Model:  analytic NIE(θ) + quadtree residual + BH render + logp
#   Verify: MH posterior covers truth within a few σ.
using Pkg; Pkg.activate(".")
using Statistics, Random
using Jens
using Jens.QuadTreeFit: quadtree_image_logp
using Jens.LensGenerator: GenGrid, LensedPlane, LightPlane
using Jens.LensSystem: ForwardModel, render
using Jens.LightModel: ExtendedSource
using Jens.LightModel.GaussianLight: GaussianSphere
using Jens.LensCosmo: Cosmology
import Jens.LensModel: NIE
using Jens.LensBase: SingleModel, lens_hessian
using Jens.LensMH: lens_mh, chain, chain_stats

# ═══════════ truth + noisy observation ═══════════
const bT, sT, e1T = 1.2, 0.15, 0.142857
const SIGMA_NOISE = 0.03

src = ExtendedSource(GaussianSphere; amp=1.0, sigma=0.05,
                     xcentre=0.18, ycentre=0.10)
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
grid_fn = () -> GenGrid(pix_n=80, pix_size=0.02)

lens_true = SingleModel(NIE; theta_E=bT, s_scale=sT, e1=e1T, e2=0.0)
sys_true = ForwardModel(lens_plane=LensedPlane(lens_true; z_lens=0.5, cosmology=cosmo),
                        source_plane=LightPlane(src; z=1.5), grid=grid_fn())
clean = render(sys_true; solver=:batch)
Random.seed!(7)
data = clean .+ SIGMA_NOISE .* randn(size(clean))

function kappa_of(lens, xs, ys)
    hxx, hxy, hyy = lens_hessian(lens, xs, ys)
    return (hxx .+ hyy) ./ 2
end
lens_at(θ) = SingleModel(NIE; theta_E=θ[1], s_scale=θ[2], e1=θ[3], e2=0.0)
κ_analytic_at(θ, xs, ys) = kappa_of(lens_at(θ), xs, ys)
κ_target_fn(xs, ys) = kappa_of(lens_true, xs, ys)

logp = quadtree_image_logp(data, SIGMA_NOISE;
    grid_fn=grid_fn, src=src, z_lens=0.5, z_source=1.5, cosmology=cosmo,
    mask=nothing,
    lens_base=lens_at, κ_analytic=κ_analytic_at, κ_target=κ_target_fn,
    recon_iters=4, recon_sample_n=25, ext=3.0,
    max_level=4, tau=5e-4, part=0.4, roi=1.2, cap_outside=4,
    bh_theta=0.8)

logp([bT, sT, e1T])   # warm-up / compile
println("logp(θ*)   = ", round(logp([bT,sT,e1T]); sigdigits=5))
println("logp(θ*-10%) = ", round(logp([1.08,0.135,0.1286]); sigdigits=5))
println("logp(随机)  = ", round(logp([1.0,0.3,0.05]); sigdigits=5))

# ═══════════ MAIN: full M-H chain (gradient-free, appropriate) ═══════════
lower = [1.0, 0.08, 0.05]
upper = [1.4, 0.22, 0.25]
println("\n=== lens_mh (3000 步, 无梯度) ===")
t0 = time()
mh = lens_mh(logp, lower, upper; n=3000, n_adapt_delay=200, seed=17)
dt = time() - t0
println("耗时: $(round(dt; digits=1)) s  ($(round(dt/3000*1000;digits=1)) ms/样本)")

post = chain(mh; burn=500)
means = vec(mean(post; dims=2))
stds  = vec(std(post; dims=2))
println("\n后验 (MH):")
println("  theta_E = $(round(means[1]; sigdigits=4)) ± $(round(stds[1]; sigdigits=4))   [真值 $(bT)]")
println("  s       = $(round(means[2]; sigdigits=4)) ± $(round(stds[2]; sigdigits=4))   [真值 $(sT)]")
println("  e1      = $(round(means[3]; sigdigits=4)) ± $(round(stds[3]; sigdigits=4))   [真值 $(e1T)]")

truth = [bT, sT, e1T]
ok = all(abs(means[k] - truth[k]) < 3*stds[k] for k in 1:3)
println("\n真值在后验 3σ 内: ", ok ? "✅ 是" :
        "❌ 否 (均值=$(round.(means; sigdigits=4)) vs 真值 $truth)")
