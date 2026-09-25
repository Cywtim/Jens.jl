# Scheme 1b image-plane fitness: minimal closed loop validation.
#   Truth:  NIE-SingleModel(θ*)  with θ* = (theta_E, s_scale, e1)
#   Model:  analytic NIE(θ)  +  quadtree absorbing κ_res = κ_true − κ_NIE(θ)
#   Q: does logp(θ) distinguish the true parameters, despite the
#      quadtree being free to absorb residual structure?
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.QuadTreeFit: quadtree_image_logp
using Jens.LensGenerator: GenGrid, LensedPlane, LightPlane
using Jens.LensSystem: ForwardModel, render
using Jens.LightModel: ExtendedSource
using Jens.LightModel.GaussianLight: GaussianSphere
using Jens.LensCosmo: Cosmology
import Jens.LensModel: NIE
using Jens.LensBase: SingleModel, lens_hessian

# ═══════════ truth = NIE (SingleModel parametrization) ═══════════
const bT, sT, e1T = 1.2, 0.15, 0.142857

function nie_κ(lens, xs, ys)
    hxx, hxy, hyy = lens_hessian(lens, xs, ys)
    return (hxx .+ hyy) ./ 2
end

src = ExtendedSource(GaussianSphere; amp=1.0, sigma=0.05,
                     xcentre=0.18, ycentre=0.10)
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
grid_fn = () -> GenGrid(pix_n=120, pix_size=0.02)

# synthetic observation from TRUE lens (noiseless)
lens_true = SingleModel(NIE; theta_E=bT, s_scale=sT, e1=e1T, e2=0.0)
sys_true = ForwardModel(
    lens_plane=LensedPlane(lens_true; z_lens=0.5, cosmology=cosmo),
    source_plane=LightPlane(src; z=1.5), grid=grid_fn())
data = render(sys_true; solver=:batch)

# ═══════════ θ = (theta_E, s_scale, e1) ═══════════
lens_at(θ) = SingleModel(NIE; theta_E=θ[1], s_scale=θ[2], e1=θ[3], e2=0.0)
κ_analytic_at(θ, xs, ys) = nie_κ(lens_at(θ), xs, ys)
κ_target_fn(xs, ys) = nie_κ(lens_true, xs, ys)

# ═══════════ build logp ═══════════
logp = quadtree_image_logp(data, 0.02;
    grid_fn=grid_fn, src=src, z_lens=0.5, z_source=1.5, cosmology=cosmo,
    mask=nothing,
    lens_base=lens_at, κ_analytic=κ_analytic_at, κ_target=κ_target_fn,
    recon_iters=6, recon_sample_n=60, ext=3.0,
    max_level=6, tau=5e-4, part=0.4, roi=1.2, cap_outside=4,
    bh_theta=0.8)

θT = [bT, sT, e1T]
println("logp(θ=true) = ", round(logp(θT); sigdigits=5))
println("logp(θ=随机远) = ", round(logp([1.0, 0.3, 0.05]); sigdigits=5))

# ═══════════ distinguishing power ═══════════
println("\n=== logp 沿单参数微扰（其余=真值）===")
names = ["theta_E", "s_scale", "e1"]
for (k, d) in ((1,0.1), (2,0.05), (3,0.06))
    θlo = copy(θT); θlo[k] -= d
    θhi = copy(θT); θhi[k] += d
    println("  $names[k]±$d :  ",
            "logp(θ*-$d)=", round(logp(θlo); sigdigits=4),
            "  logp(θ*)=", round(logp(θT); sigdigits=4),
            "  logp(θ*+$d)=", round(logp(θhi); sigdigits=4))
end
println("\n判别: 真值处 logp 应最高（否则方案1b不可辨识）")
