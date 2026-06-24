#!/usr/bin/env julia
#  GPU batch MCMC: test Threads.@threads + CUDA streams
#  Usage:   julia --project=. -t 4 examples/test_gpu_batch.jl

push!(LOAD_PATH, "@stdlib")

using Cosmology, Statistics, Printf, Random
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane
using Jens.LensSystem: ForwardModel, render
using Jens.LensPSF: GaussianPSF
using Jens.LensMH
using CUDA; Base.retry_load_extensions()
JC = Base.get_extension(Jens, :JensCUDA)

@assert Threads.nthreads() >= 2 "Need at least 2 threads, got $(Threads.nthreads())"

function bench()
    cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
    zl, zs = 0.3, 1.5
    psf = GaussianPSF(; fwhm=0.12)
    n_pix, pix = 256, 0.04

    # Common data (CPU)
    grid_cpu = Jens.LensGenerator.GenGrid(pix_n=n_pix, pix_size=pix)
    m = CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0))
    lp = LensedPlane(m; z_lens=zl, cosmology=cosmo)
    h = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)
    s_cpu = ForwardModel(lens_plane=lp, source_plane=LightPlane(h; z=zs), grid=grid_cpu, psf=psf)
    data = render(s_cpu)
    sigma = max(median(abs.(data)) * 0.1, 1e-3)
    data .+= sigma * randn(size(data))
    println("Data: $(size(data)), σ=$(round(sigma,digits=5))")

    lower = [0.2, 0.1, 0.05]
    upper = [2.0, 5.0, 1.0]
    init  = [0.75, 0.9, 0.35]
    step0 = [0.01, 0.03, 0.01]

    # ══════════════════════════════════════════════════════════
    #  CPU single
    # ══════════════════════════════════════════════════════════
    function cpu_lpfn(p)
        m2 = CombinedLens(SIS => (theta_E=p[1], xcentre=0.0, ycentre=0.0))
        lp2 = LensedPlane(m2; z_lens=zl, cosmology=cosmo)
        h2 = ExtendedSource(SersicSpheric; amp=p[2], Rsersic=p[3], n=2.0,
                            xcentre=0.0, ycentre=0.0)
        s2 = ForwardModel(lens_plane=lp2, source_plane=LightPlane(h2; z=zs),
                          grid=grid_cpu, psf=psf)
        img = render(s2)
        return -sum((data .- img).^2) / (2 * sigma^2)
    end

    println("\n=== CPU single chain ===")
    Random.seed!(42)
    t_cpu = @elapsed r_cpu = lens_mh(cpu_lpfn, lower, upper; n=400, adapt=false,
                                     step0=step0, init=init, seed=42)
    println("  $(round(t_cpu,digits=1))s  accepted=$(r_cpu.accepted)/400")

    # ══════════════════════════════════════════════════════════
    #  GPU single
    # ══════════════════════════════════════════════════════════
    grid_gpu = JC.GenGrid_GPU(pix_n=n_pix, pix_size=Float32(pix))

    function gpu_lpfn(p)
        m2 = CombinedLens(SIS => (theta_E=p[1], xcentre=0.0, ycentre=0.0))
        lp2 = LensedPlane(m2; z_lens=zl, cosmology=cosmo)
        h2 = ExtendedSource(SersicSpheric; amp=p[2], Rsersic=p[3], n=2.0,
                            xcentre=0.0, ycentre=0.0)
        s2 = ForwardModel(lens_plane=lp2, source_plane=LightPlane(h2; z=zs),
                          grid=grid_gpu, psf=psf)
        img = render(s2)
        # NOTE: Array() does per-stream sync — no CUDA.synchronize() needed
        img_cpu = Float64.(Array(img))
        return -sum((data .- img_cpu).^2) / (2 * sigma^2)
    end

    # Warm up GPU
    gpu_lpfn(init); gpu_lpfn(init)

    println("\n=== GPU single chain ===")
    Random.seed!(42)
    t_gpu1 = @elapsed r_gpu1 = lens_mh(gpu_lpfn, lower, upper; n=400, adapt=false,
                                       step0=step0, init=init, seed=42)
    println("  $(round(t_gpu1,digits=1))s  accepted=$(r_gpu1.accepted)/400")

    # ══════════════════════════════════════════════════════════
    #  GPU batch — Threads.@threads with per-thread CUDA streams
    # ══════════════════════════════════════════════════════════

    n_chains = Threads.nthreads()

    function lens_mh_gpu_batch(logp_fn, lower, upper; n_chains=n_chains, n=400,
                                step0=nothing, init=nothing, seed=42, adapt=false)
        d = length(lower)
        step0_vec = step0 === nothing ? (upper .- lower) ./ 500 : step0
        init_vec  = init === nothing ? (lower .+ upper) ./ 2 : init

        results = Vector{MHResult}(undef, n_chains)

        Threads.@threads for c in 1:n_chains
            # Each thread gets its own CUDA stream implicitly
            results[c] = lens_mh(logp_fn, lower, upper;
                                 n=n, step0=step0_vec, adapt=adapt,
                                 init=init_vec, seed=seed + c)
        end
        return results
    end

    println("\n=== GPU batch ×$(n_chains) (Threads + CUDA streams) ===")
    Random.seed!(42)
    t_gpuN = @elapsed chains = lens_mh_gpu_batch(gpu_lpfn, lower, upper;
                                                  n_chains=n_chains, n=400,
                                                  step0=step0, init=init, seed=42)
    println("  wall=$(round(t_gpuN,digits=1))s  ideal=$(round(t_gpu1,digits=1))s  speedup=$(round(t_gpu1*n_chains/t_gpuN,digits=1))x")

    for (k, c) in enumerate(chains)
        println("  Chain $k: accepted=$(c.accepted)/400")
    end

    # ══════════════════════════════════════════════════════════
    #  Summary
    # ══════════════════════════════════════════════════════════
    println("\n" * "="^55)
    @printf "  %-20s %8s %8s\n" "Method" "Time" "vs CPU"
    @printf "  %-20s %8s %8s\n" "------" "----" "------"
    @printf "  %-20s %6.1f s %6s\n" "CPU 1 chain" t_cpu "—"
    @printf "  %-20s %6.1f s %6.1f×\n" "GPU 1 chain" t_gpu1 t_cpu/t_gpu1
    @printf "  %-20s %6.1f s %6.1f×\n" "GPU $(n_chains) chains" t_gpuN t_cpu/t_gpuN
    println("\nDone.")
end

bench()