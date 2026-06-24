#!/usr/bin/env julia
#  GPU full pipeline test
#  Usage:   julia --project=. examples/test_gpu_proper.jl

push!(LOAD_PATH, "@stdlib")

using Cosmology, Statistics, Printf
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Jens.LensPSF: GaussianPSF

# Load CUDA + retry failed extensions
using CUDA; Base.retry_load_extensions()

# Now load the GPU extension
JC = Base.get_extension(Jens, :JensCUDA)
JC === nothing && error("GPU extension not loaded! Check CUDA installation.")

function bench()
    cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
    z_lens, z_src = 0.3, 1.5

    println("="^70)
    println("  GPU vs CPU render @ 128², gaussian PSF (fwhm=0.12\")")
    println("="^70)

    # CPU
    grid_cpu = GenGrid(pix_n=128, pix_size=0.04)
    m = CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0))
    lp = LensedPlane(m; z_lens, cosmology=cosmo)
    h = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)
    psf = GaussianPSF(; fwhm=0.12)

    sys_cpu = ForwardModel(lens_plane=lp, source_plane=LightPlane(h; z=z_src),
                           grid=grid_cpu, psf=psf)

    # GPU
    grid_gpu = JC.GenGrid_GPU(pix_n=128, pix_size=0.04f0)

    # Warm-up
    img_cpu = render(sys_cpu)
    img_gpu = JC.render_lens(h, grid_gpu, lp; psf=psf, z_source=z_src)
    CUDA.synchronize()
    println("CPU sum=", round(sum(img_cpu), digits=2), " max=", round(maximum(img_cpu), digits=5))
    println("GPU sum=", round(sum(Array(img_gpu)), digits=2), " max=", round(maximum(Array(img_gpu)), digits=5))
    println("diff  =", round(maximum(abs.(img_cpu .- Array(img_gpu))), digits=8))

    # Benchmark
    tc = @elapsed for _ in 1:50; render(sys_cpu); end
    tg = @elapsed for _ in 1:50
        JC.render_lens(h, grid_gpu, lp; psf=psf, z_source=z_src)
        CUDA.synchronize()
    end

    @printf "\n  CPU: %7.2f ms   GPU: %7.2f ms   speedup: %.1f×\n" tc/50*1000 tg/50*1000 tc/tg
    println("\nDone.")
end

bench()