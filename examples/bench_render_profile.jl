#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#  Profiling: render() step-by-step timing breakdown
#  Uses @time + loop instead of BenchmarkTools (compat issue)
# ═══════════════════════════════════════════════════════════════

using Cosmology, Statistics
using Jens
using Jens.LensModel: SIS
using Jens.LensModel.ComLens: CombinedLens
using Jens.LightModel: ExtendedSource, PointImages, CompositeImage, evaluate_source
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensPSF: GaussianPSF
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Jens.LensSolver: solve_images

function time_it(f, n_warmup=3, n_run=100)
    # warmup
    for _ in 1:n_warmup; f(); end
    GC.gc()
    t0 = time_ns()
    for _ in 1:n_run; f(); end
    return (time_ns() - t0) / n_run / 1e6  # ms
end

cosmo  = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens = 0.3
z_src  = 1.5

println("="^70)
println("  render() step-by-step profiling")
println("="^70)

for (pix_n, pix_size) in [(64, 0.12), (128, 0.06), (256, 0.03)]
    label = "$(pix_n)²"
    println("\n─── Grid: $label  pix_size=$(pix_size)\" ───")

    grid = GenGrid(pix_n=pix_n, pix_size=pix_size)
    psf  = GaussianPSF(; fwhm=Float64(pix_size * 2))
    xg, yg = grid.xg, grid.yg

    lens = LensedPlane(
        CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0));
        z_lens=z_lens, cosmology=cosmo)

    host = ExtendedSource(SersicSpheric;
        amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)

    agn_images = solve_images(lens, 0.05, -0.03; z_source=z_src)
    agn_positions = [(x, y) for (x, y, _) in agn_images]
    agn = PointImages((5.0, agn_positions); intrinsic=true)

    src = CompositeImage(host, agn)
    sys = ForwardModel(
        lens_plane=lens,
        source_plane=LightPlane(src; z=z_src),
        grid=grid, psf=psf)

    # Pre-compute intermediates for sub-step benchmarks
    ax, ay = Jens.LensBase.lens_derivative(lens, xg, yg; z_source=z_src)
    bx = xg .- ax
    by = yg .- ay
    raw = evaluate_source(host, bx, by)
    half = div(pix_n, 2) * pix_size
    x_min, y_min = -half, -half
    pixel_scale = pix_size

    buf = zeros(size(xg))
    mus = [39.6, 18.2, 12.7, 8.1]
    pos_list = [(0.6, -0.36), (-0.5, 0.30), (-0.4, 0.75), (0.3, 0.9)]

    t_ld  = time_it(() -> Jens.LensBase.lens_derivative(lens, xg, yg; z_source=z_src))
    t_rt  = time_it(() -> (xg .- ax, yg .- ay))
    t_src = time_it(() -> evaluate_source(host, bx, by))
    t_psf = time_it(() -> Jens.LensPSF.conv_psf(raw, psf, pixel_scale))
    t_alloc = time_it(() -> (r = similar(xg); fill!(r, 0.0)))
    t_agn = time_it(() -> begin
        fill!(buf, 0.0)
        for (i, (tx, ty)) in enumerate(pos_list)
            px = (tx - x_min) / pixel_scale + 1
            py = (ty - y_min) / pixel_scale + 1
            Jens.LensPSF.render_point!(buf, psf, px, py, 5.0 * mus[i];
                                       pixel_scale=pixel_scale, half=7)
        end
    end)
    t_full = time_it(() -> render(sys), 3, 30)

    sum_parts = t_ld + t_rt + t_src + t_psf + t_alloc + t_agn
    overhead = t_full - sum_parts

    println("  lens_derivative        $(rpad(round(t_ld, digits=4), 8)) ms  ($(rpad(round(t_ld/t_full*100, digits=1),5))%)")
    println("  ray-trace (θ-α)        $(rpad(round(t_rt, digits=4), 8)) ms  ($(rpad(round(t_rt/t_full*100, digits=1),5))%)")
    println("  Sersic evaluate        $(rpad(round(t_src, digits=4), 8)) ms  ($(rpad(round(t_src/t_full*100, digits=1),5))%)")
    println("  PSF conv (FFT)         $(rpad(round(t_psf, digits=4), 8)) ms  ($(rpad(round(t_psf/t_full*100, digits=1),5))%)")
    println("  alloc + fill!          $(rpad(round(t_alloc, digits=4), 8)) ms  ($(rpad(round(t_alloc/t_full*100, digits=1),5))%)")
    println("  render_point! (4imgs)  $(rpad(round(t_agn, digits=4), 8)) ms  ($(rpad(round(t_agn/t_full*100, digits=1),5))%)")
    println("  ─"^35)
    println("  sum of parts           $(rpad(round(sum_parts, digits=4), 8)) ms  ($(rpad(round(sum_parts/t_full*100, digits=1),5))%)")
    println("  overhead (调度/copyto!) $(rpad(round(overhead, digits=4), 8)) ms  ($(rpad(round(overhead/t_full*100, digits=1),5))%)")
    println("  render() TOTAL         $(rpad(round(t_full, digits=3), 8)) ms  (100%)")
end

# ═══════════════════════════════════════════════════════════════
#  PSF vs no-PSF  128²
# ═══════════════════════════════════════════════════════════════
println("\n" * "="^70)
println("  PSF vs no-PSF  (128², host only)")
println("="^70)

grid128 = GenGrid(pix_n=128, pix_size=0.06)
psf128  = GaussianPSF(; fwhm=0.12)
lens128 = LensedPlane(
    CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0));
    z_lens=z_lens, cosmology=cosmo)
host128 = ExtendedSource(SersicSpheric;
    amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)

sys_psf = ForwardModel(
    lens_plane=lens128,
    source_plane=LightPlane(host128; z=z_src),
    grid=grid128, psf=psf128)
sys_nopsf = ForwardModel(
    lens_plane=lens128,
    source_plane=LightPlane(host128; z=z_src),
    grid=grid128, psf=nothing)

t_psf_full  = time_it(() -> render(sys_psf), 3, 50)
t_nopsf_full = time_it(() -> render(sys_nopsf), 3, 50)
println("  with PSF:         $(round(t_psf_full, digits=2)) ms")
println("  without PSF:      $(round(t_nopsf_full, digits=2)) ms")
println("  PSF cost:         $(round(t_psf_full - t_nopsf_full, digits=2)) ms  ($(round((t_psf_full - t_nopsf_full)/t_psf_full*100, digits=1))% of total)")