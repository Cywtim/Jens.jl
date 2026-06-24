#!/usr/bin/env julia
#  GPU conv_psf test — direct convolution on CuArray
#  Usage:   julia --project=. examples/test_gpu_conv.jl

push!(LOAD_PATH, "@stdlib")

using Cosmology, Statistics, Printf, CUDA
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Jens.LensPSF: GaussianPSF, conv_psf, make_kernel

function bench_gpu()
    cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
    z_lens, z_src = 0.3, 1.5

    println("="^60)
    println("  GPU Direct Convolution: render() benchmark")
    println("="^60)

    for (n, ps) in [(64, 0.08), (128, 0.04), (256, 0.02), (512, 0.01)]
        grid = GenGrid(pix_n=n, pix_size=ps)
        psf  = GaussianPSF(; fwhm=0.12)

        m = CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0))
        lp = LensedPlane(m; z_lens, cosmology=cosmo)
        h = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)
        s = ForwardModel(lens_plane=lp, source_plane=LightPlane(h; z=z_src), grid=grid, psf=psf)

        # Warm-up (CPU)
        img_cpu = render(s)
        println("  $(n)² CPU warm: sum=", round(sum(img_cpu), digits=1))

        # GPU render
        img_gpu = render(s)
        CUDA.synchronize()
        println("  $(n)² GPU warm: sum=", round(sum(img_gpu), digits=1))

        # Bench CPU
        tc = @elapsed for _ in 1:10; render(s); end
        # Bench GPU
        tg = @elapsed for _ in 1:10
            render(s); CUDA.synchronize()
        end

        @printf "  %4d²  CPU %7.2f ms  GPU %7.2f ms  %5.1f×\n" n tc/10*1000 tg/10*1000 tc/tg
    end
    println("\nDone.")
end

bench_gpu()