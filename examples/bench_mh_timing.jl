#!/usr/bin/env julia
#  Benchmark: LensMH timing (fast)
#  Usage:   julia --project=. examples/bench_mh_timing.jl

push!(LOAD_PATH, "@stdlib")

using Cosmology, LinearAlgebra, Distributions, Statistics, Random, Printf
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Jens.LensPSF: GaussianPSF
using Jens.LensMH

cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens, z_src = 0.3, 1.5

psf_gauss = GaussianPSF(; fwhm=3.0)
using Jens.WFC3
psf_wfc3 = WFC3.WFC3_UVIS_PSF(lambda_eff=606e-9, oversample=5)

grid_sizes = [32, 64, 128, 256]
pix_sizes  = [0.32, 0.16, 0.08, 0.04]

function make_sys(n, ps, psf)
    grid = GenGrid(pix_n=n, pix_size=ps)
    m = CombinedLens(SIS => (theta_E=0.8, xcentre=0.0, ycentre=0.0))
    lp = LensedPlane(m; z_lens, cosmology=cosmo)
    h = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)
    s = ForwardModel(lens_plane=lp, source_plane=LightPlane(h; z=z_src), grid=grid, psf=psf)
    return s, grid
end

# Warm-up JIT
println("Warming up...")
sys_w, _ = make_sys(64, 0.16, psf_gauss)
render(sys_w); render(sys_w)
sys_w2, _ = make_sys(64, 0.16, psf_wfc3)
render(sys_w2)
println()

# ── Render timing ─────────────────────────────────────────────
println("="^60)
println("  render() single call [ms]")
println("="^60)
@printf "  %-8s %8s %8s %8s\n" "grid" "pixels" "gauss" "WFC3"
@printf "  %-8s %8s %8s %8s\n" "----" "------" "-----" "----"
for (n, ps) in zip(grid_sizes, pix_sizes)
    sys_g, _ = make_sys(n, ps, psf_gauss)
    tg = @elapsed(for _ in 1:10; render(sys_g); end) / 10 * 1000

    sys_w, _ = make_sys(n, ps, psf_wfc3)
    nrep = n >= 256 ? 1 : 3
    tw = @elapsed(for _ in 1:nrep; render(sys_w); end) / nrep * 1000

    @printf "  %3d²   %8d %8.2f %8.0f\n" n n*n tg tw
end

# ── MH step timing ────────────────────────────────────────────
println()
println("="^60)
println("  LensMH: total time [s]")
println("="^60)
@printf "  %-8s %8s %12s %12s %12s\n" "grid" "pixels" "Gauss 1k" "Gauss 5k" "WFC3 1k"
@printf "  %-8s %8s %12s %12s %12s\n" "----" "------" "--------" "--------" "--------"
for (n, ps) in zip(grid_sizes, pix_sizes)
    grid = GenGrid(pix_n=n, pix_size=ps)
    sys_g, _ = make_sys(n, ps, psf_gauss)
    data = render(sys_g)
    sigma = max(median(abs.(data)) * 0.05, 1e-3)
    data .+= sigma * randn(size(data))

    function lpfn_g(p)
        theta_E, amp, Rsersic = p
        m = CombinedLens(SIS => (theta_E=theta_E, xcentre=0.0, ycentre=0.0))
        lp2 = LensedPlane(m; z_lens, cosmology=cosmo)
        h = ExtendedSource(SersicSpheric; amp=amp, Rsersic=Rsersic, n=2.0, xcentre=0.0, ycentre=0.0)
        s = ForwardModel(lens_plane=lp2, source_plane=LightPlane(h; z=z_src), grid=grid, psf=psf_gauss)
        img = render(s; solver=:batch)
        return -sum((data .- img).^2) / (2 * sigma^2)
    end

    lower = [0.2, 0.1, 0.05]
    upper = [2.0, 5.0, 1.0]
    init  = [0.8, 1.0, 0.3]

    # Gaussian PSF: 1000 steps
    t1k = @elapsed lens_mh(lpfn_g, lower, upper; n=1000, init=init, seed=42)
    # Gaussian PSF: 5000 steps projected
    ms = t1k / 1000 * 1000
    t5k = ms * 5000 / 1000

    # WFC3 1k projected (from render ratio)
    r = (@elapsed(render(sys_g)) * 1000 < 1) ? 1.0 : 1.0  # placeholder
    sys_w, _ = make_sys(n, ps, psf_wfc3)
    tw_render = @elapsed render(sys_w)
    tg_render = @elapsed render(sys_g)
    ratio = tw_render / tg_render
    twfc3_1k = t1k * ratio

    @printf "  %3d²   %8d %9.1f s %9.0f s %9.0f s\n" n n*n t1k t5k twfc3_1k
end

println("\nDone.")