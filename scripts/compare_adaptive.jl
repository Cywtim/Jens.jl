#!/usr/bin/env julia --project=.
# ═══════════════════════════════════════════════════════════════
#  compare_adaptive.jl
#
#  Side-by-side comparison: uniform vs adaptive ray-tracing.
#
#  Usage:  julia --project=. compare_adaptive.jl
# ═══════════════════════════════════════════════════════════════

using Jens, Cosmology, Statistics, Plots

# ── Import everything we need ──
using Jens.LensAdaptiveGrid:
    RefinementMap, adaptive_grid_info, render_adaptive

# ═══════════════════════════════════════════════════════════════
#  1.  Build the lens system
# ═══════════════════════════════════════════════════════════════
println("Building lens system...")

cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)

# SIE with moderate ellipticity — produces visible tangential arcs
# SIE uses e1, e2 ellipticity components (not q, varphi)
lens = Jens.LensModel.ComLens.MyLens(
    Jens.LensModel.SIE => (theta_E=1.2, e1=0.2, e2=0.08,
                           xcentre=0.0, ycentre=0.0))

# Compact Sersic source — lensed into an arc/Einstein ring
src_profile  = Jens.LightModel.SersicLight.SersicSpheric
source_model = Jens.LightModel.ExtendedSource(
    src_profile; amp=1.0, Rsersic=0.15, n=2.0,
    xcentre=0.03, ycentre=0.02)   # slightly off-centre → asymmetric arc

grid = Jens.LensGenerator.GenGrid(pix_n=200)
pixel_scale = grid.pix_size

sys = Jens.LensSystem.ForwardModel(;
    lens_plane   = Jens.LensGenerator.LensedPlane(
        lens; z_lens=0.3, cosmology=cosmo),
    source_plane = source_model,
    grid         = grid,
    z_source     = 1.5,
    psf          = Jens.LensPSF.GaussianPSF(fwhm=0.08),   # light PSF blur
)


# ═══════════════════════════════════════════════════════════════
#  2.  Compute refinement map (detection pass)
# ═══════════════════════════════════════════════════════════════
println("Computing refinement map...")

ref = RefinementMap(grid, lens; threshold=0.15, sub_n=4, z_source=1.5)
adaptive_grid_info(ref, grid)


# ═══════════════════════════════════════════════════════════════
#  3.  Render both versions
# ═══════════════════════════════════════════════════════════════
println("Rendering uniform...")
@time model_uniform = Jens.LensSystem.render(sys)

println("Rendering adaptive...")
@time model_adaptive = render_adaptive(sys, ref)

diff = model_adaptive .- model_uniform

println("")
println("─── Results ───")
println("  max  |diff|:  ", round(maximum(abs, diff); digits=6))
println("  mean |diff|:  ", round(mean(abs, diff); digits=8))
println("  refined:      $(ref.n_refined)/$(length(ref.needs_refine)) " *
        "($(round(ref.n_refined/length(ref.needs_refine)*100, digits=1))%)")
println("  ray overhead: $(round((length(ref.needs_refine) - ref.n_refined +
        ref.n_refined * ref.sub_n^2) / length(ref.needs_refine) * 100 - 100,
        digits=1))%")


# ═══════════════════════════════════════════════════════════════
#  4.  Plot — four panels
# ═══════════════════════════════════════════════════════════════
println("Plotting...")

# Prepare coordinates
xvec = grid.xg[:, 1]
yvec = grid.yg[1, :]
ext  = extrema(model_uniform)

# Difference colormap: white at zero, red positive, blue negative
diff_lim = maximum(abs, diff) * 1.05
diff_lim = max(diff_lim, 1e-10)

p1 = heatmap(xvec, yvec, model_uniform';
    aspect_ratio=:equal, title="Uniform (1 ray/pixel)",
    xlabel="x (arcsec)", ylabel="y (arcsec)",
    c=:magma, clim=ext, colorbar=true,
    right_margin=5Plots.mm)

p2 = heatmap(xvec, yvec, model_adaptive';
    aspect_ratio=:equal, title="Adaptive (4×4 near critical curve)",
    xlabel="x (arcsec)", ylabel="y (arcsec)",
    c=:magma, clim=ext, colorbar=true,
    right_margin=5Plots.mm)

p3 = heatmap(xvec, yvec, diff';
    aspect_ratio=:equal, title="Difference (adaptive − uniform)",
    xlabel="x (arcsec)", ylabel="y (arcsec)",
    c=:RdBu, clim=(-diff_lim, diff_lim), colorbar=true,
    right_margin=5Plots.mm)

p4 = heatmap(xvec, yvec, Float64.(ref.needs_refine)';
    aspect_ratio=:equal, title="Refinement mask ($(ref.n_refined) pixels)",
    xlabel="x (arcsec)", ylabel="y (arcsec)",
    c=[:black, :yellow], clim=(0, 1), colorbar=false,
    right_margin=5Plots.mm)

fig = plot(p1, p2, p3, p4;
    layout=(2, 2), size=(1200, 1100),
    plot_title="Adaptive vs Uniform Ray-Tracing  |  " *
               "SIE (θ_E=1.2, e1=0.2)  |  threshold=0.15  |  sub_n=4",
    titlefontsize=11)

outpath = joinpath(@__DIR__, "compare_adaptive.png")
savefig(fig, outpath)
println("Saved → $outpath")
println("Done.")