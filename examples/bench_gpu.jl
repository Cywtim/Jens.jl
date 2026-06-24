#!/usr/bin/env julia
#  GPU benchmark: render timing
#  Usage:   julia --project=. --compiled-modules=no examples/bench_gpu.jl

push!(LOAD_PATH, "@stdlib")

using Cosmology, LinearAlgebra, Statistics, Printf, CUDA
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane
using Jens.LensSystem: ForwardModel, render
using Jens.LensPSF: GaussianPSF

function make_sys(n, ps)
    cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
    z_lens, z_src = 0.3, 1.5
    grid = Jens.LensGenerator.GenGrid(pix_n=n, pix_size=ps)
    m = CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0))
    lp = LensedPlane(m; z_lens, cosmology=cosmo)
    h = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)
    s = ForwardModel(lens_plane=lp, source_plane=LightPlane(h; z=z_src), grid=grid,
                     psf=GaussianPSF(; fwhm=3.0))
    return s
end

function bench()
    # Warm-up
    println("Warming up...")
    sys = make_sys(64, 0.16)
    render(sys)
    render(sys)
    println()

    grid_sizes = [32, 64, 128, 256, 512]
    pix_sizes  = [0.32, 0.16, 0.08, 0.04, 0.02]

    println("="^70)
    println("  GPU vs CPU render() [ms] — RTX 4080 SUPER")
    println("="^70)
    @printf "  %-8s %8s %8s %8s %8s\n" "grid" "pixels" "CPU" "GPU" "×"
    @printf "  %-8s %8s %8s %8s %8s\n" "----" "------" "---" "---" "--"

    for (n, ps) in zip(grid_sizes, pix_sizes)
        sys = make_sys(n, ps)
        npix = n * n

        # CPU
        tc = @elapsed for _ in 1:5; render(sys); end
        tc /= 5

        # GPU
        tg = @elapsed for _ in 1:5
            render(sys)
            CUDA.synchronize()
        end
        tg /= 5

        su = iszero(tg) ? Inf : tc / tg
        @printf "  %3d²  %8d %8.2f %8.2f %7.1f×\n" n npix tc*1000 tg*1000 su
    end
end

bench()
println("\nDone.")