#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
# NFWE lens + Sersic host + AGN: Narrow-bound M-H
#
# KEY INSIGHT: NFW's Rs and alpha_Rs are nearly perfectly
# degenerate.  You MUST fix Rs (from mass-concentration relation
# or external data).  With Rs fixed, the remaining 7 params are
# identifiable — but need narrow search bounds or good init.
# ═══════════════════════════════════════════════════════════════
using Pkg
Pkg.develop(path="/home/cyan/Documents/GitHub/Jens/Jens.jl")

using Cosmology, Random, Statistics
using Jens, CUDA
using Jens.LensModel: NFWE
using Jens.LensModel.ComLens: CombinedLens
using Jens.LightModel: ExtendedSource, PointImages, CompositeImage
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensPSF: GaussianPSF
using Jens:LensGenerator as LG
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Jens.LensSolver: solve_images
using Jens.LensMH: lens_mh_multistart, chain
using Jens.LensUtils: randn_like

Random.seed!(42)
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens, z_src, sigma_data = 0.3, 1.5, 0.1
grid = LG.GenGrid(pix_n=64, pix_size=0.06) #gpu_grid(pix_n=64, pix_size=0.06)
psf  = GaussianPSF(0.05)
println("Grid: $(grid.pix_n)×$(grid.pix_n), pixel=$(grid.pix_size)\"")

truth = (Rs=3.5, alpha_Rs=2.0, e1=0.15, e2=0.08, xcentre=0.02, ycentre=-0.01,
         amp=1.0, Rsersic=0.3, n=2.0, flux=5.0)

lens_truth = LensedPlane(CombinedLens(NFWE => (
    Rs=truth.Rs, alpha_Rs=truth.alpha_Rs, e1=truth.e1, e2=truth.e2,
    xcentre=truth.xcentre, ycentre=truth.ycentre)); z_lens=z_lens, cosmology=cosmo)

agn_pos = [(x,y) for (x,y,_) in solve_images(lens_truth, 0.05, -0.03;
    z_source=z_src, search_radius=3.0, n_radial=12, n_angular=24)]
println("AGN images: $(length(agn_pos))")

data = render(ForwardModel(
    lens_plane=lens_truth,
    source_plane=LightPlane(CompositeImage(
        ExtendedSource(SersicSpheric; amp=truth.amp, Rsersic=truth.Rsersic, n=truth.n),
        PointImages((truth.flux, agn_pos); intrinsic=true)); z=z_src),
    grid=grid, psf=psf))
data .+= sigma_data * randn_like(data)
println("Data peak ≈ $(round(maximum(data), digits=1))")

# === log-posterior (7 free params) ===
function build_logp(data, grid, psf, cosmo, z_lens, z_src, agn_pos)
    function logp_fn(p)
        alpha_Rs, e1, e2, amp, rsr, n, flux = p
        (alpha_Rs > 0 && rsr > 0 && n > 0.2 && amp > 0 && flux > 0) || return -Inf
        abs(e1) < 1 && abs(e2) < 1 && e1^2 + e2^2 < 1 || return -Inf
        lens = LensedPlane(CombinedLens(NFWE => (
            Rs=truth.Rs, alpha_Rs=alpha_Rs, e1=e1, e2=e2,
            xcentre=truth.xcentre, ycentre=truth.ycentre)); z_lens=z_lens, cosmology=cosmo)
        host = ExtendedSource(SersicSpheric; amp=amp, Rsersic=rsr, n=n)
        agn  = PointImages((flux, agn_pos); intrinsic=true)
        fwd  = ForwardModel(lens_plane=lens,
            source_plane=LightPlane(CompositeImage(host, agn); z=z_src), grid=grid, psf=psf)
        img = try render(fwd) catch; return -Inf end
        return -0.5 * sum(((data .- img) ./ sigma_data).^2)
    end
    return logp_fn
end
logp = build_logp(data, grid, psf, cosmo, z_lens, z_src, agn_pos)

# === Narrow bounds (±30–50% around truth) ===
#           α_Rs    e1      e2     amp    Rsrs    n     flux
lower_n = [1.0,    -0.4,   -0.4,  0.2,   0.08,  0.5,  1.0]
upper_n = [3.5,     0.4,    0.4,  3.0,   0.8,   5.0,  20.0]

println("\n" * "─"^55)
println(" Multi-Start M-H: 16 starts, narrow bounds")
println(" Rs = $(truth.Rs) (fixed from e.g. mass-concentration)")
println("─"^55)
@time res = lens_mh_multistart(logp, lower_n, upper_n; n_starts=16, n=3000, seed=42)
post = chain(res; burn=1000)
println("Accepted: $(res.accepted)/3000 ($(round(res.accepted/30,digits=1))%)")

# === Results ===
names = ["alpha_Rs", "e1", "e2", "amp_host", "Rsersic", "n", "flux_agn"]
tvals = [truth.alpha_Rs, truth.e1, truth.e2, truth.amp, truth.Rsersic, truth.n, truth.flux]

println("\n" * "="^70)
println("  Parameter       Truth        Mean ± Std           Δ/σ")
println("─"^70)
for i in 1:7
    m, s = mean(post[i,:]), std(post[i,:])
    ds = (m - tvals[i]) / (s + 1e-10)
    flag = abs(ds) < 3.0 ? " ✓" : " ⚠"
    tv = round(tvals[i], digits=3)
    println("  ", rpad(names[i], 12), "  ", rpad(string(tv), 8),
            "  ", lpad(round(m, digits=4), 8), " ± ", rpad(round(s, digits=4), 8),
            "  ", lpad(round(ds, digits=1), 4), "σ", flag)
end
println("\nDone.")