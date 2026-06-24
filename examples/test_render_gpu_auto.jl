#!/usr/bin/env julia
#  Test: ForwardModel(GridGPU) → render() automatic GPU dispatch
#  Usage:   julia --project=. examples/test_render_gpu_auto.jl

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
using CUDA; Base.retry_load_extensions()

JC = Base.get_extension(Jens, :JensCUDA)
JC === nothing && error("GPU ext not loaded")

function test()
    cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
    z_lens, z_src = 0.3, 1.5
    m = CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0))
    lp = LensedPlane(m; z_lens, cosmology=cosmo)
    src = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)
    psf = GaussianPSF(; fwhm=0.12)
    sp  = LightPlane(src; z=z_src)

    println("="^60)
    println("  render() auto GPU dispatch test @ 128²")
    println("="^60)

    # CPU path
    grid_cpu = GenGrid(pix_n=128, pix_size=0.04)
    sys_cpu  = ForwardModel(lens_plane=lp, source_plane=sp, grid=grid_cpu, psf=psf)
    img_cpu  = render(sys_cpu)

    # GPU path — same render() call, different grid
    grid_gpu = JC.GenGrid_GPU(pix_n=128, pix_size=0.04f0)
    sys_gpu  = ForwardModel(lens_plane=lp, source_plane=sp, grid=grid_gpu, psf=psf)
    img_gpu  = render(sys_gpu)
    CUDA.synchronize()

    @printf "  CPU sum=%.2f  max=%.5f\n" sum(img_cpu) maximum(img_cpu)
    @printf "  GPU sum=%.2f  max=%.5f\n" sum(Array(img_gpu)) maximum(Array(img_gpu))
    @printf "  diff=%.2e\n" maximum(abs.(img_cpu .- Array(img_gpu)))

    # Bench
    tc = @elapsed for _ in 1:30; render(sys_cpu); end
    tg = @elapsed for _ in 1:30
        render(sys_gpu); CUDA.synchronize()
    end
    @printf "\n  CPU %6.2f ms   GPU %6.2f ms   %.1f×\n" tc/30*1000 tg/30*1000 tc/tg
    println("\nDone.")
end

test()