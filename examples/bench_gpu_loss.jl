#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#  Benchmark: GPU→CPU 传输 vs GPU-resident loss
# ═══════════════════════════════════════════════════════════════

using CUDA, Cosmology, Statistics, Random
using Jens
using Jens.LensModel: SIS
using Jens.LensModel.ComLens: CombinedLens
using Jens.LightModel: ExtendedSource, PointImages, CompositeImage
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensPSF: GaussianPSF
using Jens.LensGenerator: LensedPlane, LightPlane
using Jens.LensSystem: ForwardModel, render
using Jens.LensSolver: solve_images
using Jens.LensNoise: GaussNoise, add_noise, log_likelihood_gpu

function bench(f, n_warmup=5, n_run=50)
    for _ in 1:n_warmup; f(); end
    GC.gc()
    CUDA.synchronize()
    t0 = time_ns()
    for _ in 1:n_run; f(); end
    CUDA.synchronize()
    return (time_ns() - t0) / n_run / 1e6  # ms
end

Random.seed!(42)
cosmo  = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens = 0.3
z_src  = 1.5

for (pix_n, pix_size) in [(64, 0.12), (128, 0.06), (256, 0.03)]
    label = "$(pix_n)²"
    println("\n" * "="^65)
    println("  Grid: $label  ($(pix_n)×$(pix_n))")
    println("="^65)

    grid_cpu = Jens.LensGenerator.GenGrid(pix_n=pix_n, pix_size=Float64(pix_size))
    grid_gpu = gpu_grid(pix_n=pix_n, pix_size=Float32(pix_size))
    psf = GaussianPSF(; fwhm=Float64(pix_size * 2))

    lens = LensedPlane(
        CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0));
        z_lens=z_lens, cosmology=cosmo)

    host = ExtendedSource(SersicSpheric;
        amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)

    agn_img = solve_images(lens, 0.05, -0.03; z_source=z_src)
    agn_pos = [(x, y) for (x, y, _) in agn_img]
    agn = PointImages((5.0, agn_pos); intrinsic=true)
    src = CompositeImage(host, agn)

    sys = ForwardModel(
        lens_plane=lens,
        source_plane=LightPlane(src; z=z_src),
        grid=grid_gpu, psf=psf)

    sigma = 0.05

    # ── Generate data on GPU ──
    truth_gpu = render(sys)
    noise_model = GaussNoise(sigma)
    data_gpu = add_noise(truth_gpu, noise_model)
    data_cpu = Array(data_gpu)

    # ── Verify correctness ──
    # OLD: GPU→CPU every call
    old_func() = begin
        img = render(sys)
        s = sum((data_cpu .- Float64.(Array(img))).^2) / sigma^2
        -0.5 * s
    end

    # NEW: GPU-resident
    new_func() = begin
        img = render(sys)
        log_likelihood_gpu(data_gpu, img, GaussNoise(sigma))
    end

    val_old = old_func()
    val_new = new_func()
    Δ = abs(val_old - val_new)
    δ = Δ / max(abs(val_old), 1e-10) * 100

    println("  OLD result:     $(round(val_old, digits=2))")
    println("  NEW result:     $(round(val_new, digits=2))")
    println("  Rel diff:       $(round(δ, digits=6))%")
    @assert δ < 1.0 "Results diverge!"

    # ── Benchmark ──
    t_old = bench(old_func, 5, 30)
    t_new = bench(new_func, 5, 30)

    println("  OLD (GPU→CPU):  $(round(t_old, digits=3)) ms")
    println("  NEW (GPU only): $(round(t_new, digits=3)) ms")
    println("  Speedup:        $(round(t_old / t_new, digits=2))×")
    println("  Saved per step: $(round((t_old - t_new)*1000, digits=1)) μs")
end