#!/usr/bin/env julia
#  MCMC timing: CPU vs GPU, proper adaptive step
#  Usage:   julia --project=. examples/bench_mcmc_cpu_gpu.jl

push!(LOAD_PATH, "@stdlib")

using Cosmology, Statistics, Printf, Random
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Jens.LensPSF: GaussianPSF
using Jens.LensMH
using CUDA; Base.retry_load_extensions()
JC = Base.get_extension(Jens, :JensCUDA)

const COSMO  = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
const ZL, ZS = 0.3, 1.5
const PSF     = GaussianPSF(; fwhm=0.12)
const LOWER   = [0.2, 0.1, 0.05]
const UPPER   = [2.0, 5.0, 1.0]
const INIT    = [0.8, 1.0, 0.3]
const N_ITER  = 500

function bench_one(n_pix, pix_size, gpu, label)
    grid = gpu ? JC.GenGrid_GPU(pix_n=n_pix, pix_size=Float32(pix_size)) :
                 GenGrid(pix_n=n_pix, pix_size=pix_size)
    T = gpu ? Float32 : Float64

    # Generate data
    m = CombinedLens(SIS => (theta_E=T(0.8), xcentre=T(0.0), ycentre=T(0.0)))
    lp = LensedPlane(m; z_lens=ZL, cosmology=COSMO)
    h = ExtendedSource(SersicSpheric; amp=T(1.0), Rsersic=T(0.3), n=2.0,
                       xcentre=T(0.0), ycentre=T(0.0))
    sp = LightPlane(h; z=ZS)
    sys = ForwardModel(lens_plane=lp, source_plane=sp, grid=grid, psf=PSF)

    data_raw = render(sys)
    if gpu
        CUDA.synchronize()
        data = Float64.(Array(data_raw))
    else
        data = data_raw
    end
    sigma = max(median(abs.(data)) * 0.05, 1e-3)

    function logp_fn(p::Vector{Float64})
        theta_E, amp, Rsersic = p
        m2 = CombinedLens(SIS => (theta_E=theta_E, xcentre=0.0, ycentre=0.0))
        lp2 = LensedPlane(m2; z_lens=ZL, cosmology=COSMO)
        h2 = ExtendedSource(SersicSpheric; amp=amp, Rsersic=Rsersic, n=2.0,
                            xcentre=0.0, ycentre=0.0)
        s2 = ForwardModel(lens_plane=lp2, source_plane=LightPlane(h2; z=ZS),
                          grid=grid, psf=PSF)
        img = render(s2)
        if gpu
            # NOTE: Array() does per-stream sync — no CUDA.synchronize() needed
            img_cpu = Float64.(Array(img))
        else
            img_cpu = img
        end
        return -sum((data .- img_cpu).^2) / (2 * sigma^2)
    end

    Random.seed!(42)
    t = @elapsed result = lens_mh(logp_fn, LOWER, UPPER; n=N_ITER, init=INIT,
                                  adapt=true, seed=42)
    ms = t / N_ITER * 1000
    ar = result.accepted / N_ITER * 100
    return t, ms, ar
end

println("="^75)
println("  MCMC timing (3-param SIS, Gauss PSF fwhm=0.12\", adaptive MH)")
println("="^75)
@printf "  %-6s  %8s  %8s  %8s  %8s  %8s\n" "grid" "CPU(s)" "GPU(s)" "ratio" "CPU ms" "GPU ms"
@printf "  %-6s  %8s  %8s  %8s  %8s  %8s\n" "----" "------" "------" "-----" "------" "------"

for (n, ps) in [(64, 0.16), (128, 0.08), (256, 0.04)]
    tc, msc, arc = bench_one(n, ps, false, "cpu")
    tg, msg, arg = bench_one(n, ps, true,  "gpu")
    @printf "  %3d²  %8.1f  %8.1f  %7.1f×  %6.1f  %6.1f\n" n tc tg tc/tg msc msg
end
println("\nDone.")