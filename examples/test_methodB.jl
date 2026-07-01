push!(LOAD_PATH, "@stdlib")
using Cosmology, Statistics, Random, Printf, NLsolve
import Cosmology as LC
using Jens, CUDA; Base.retry_load_extensions()
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: NFWE, Shear, SIS
using Jens.LensBase: lens_derivative, lens_hessian
using Jens.LightModel: ExtendedSource, PointImage, CompositeImage
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensSystem: ForwardModel, render
using Jens.LensSolver: solve_images
using Jens.LensPSF: render_point!
using Jens.LensMH

cosmo = LC.FlatLCDM(0.7, 0.3, 0.0, 0.0)
zl, zs = 0.3, 1.5
grid_gpu = gpu_grid(pix_n=128, pix_size=Float32(0.08))
psf = Jens.LensPSF.GaussianPSF(; fwhm=0.12)

# Data
lens_truth = CombinedLens(
    NFWE => (Rs=5.0, alpha_Rs=1.0, e1=0.2, e2=0.1, xcentre=0.0, ycentre=0.0),
    Shear => (gamma1=0.05, gamma2=-0.03, xcentre=0.0, ycentre=0.0))
host_truth = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0, xcentre=0.0, ycentre=0.0)
agn_truth  = PointImage(flux=50.0, beta_x=0.05, beta_y=-0.01)
sys_truth = ForwardModel(
    lens_plane=Jens.LensGenerator.LensedPlane(lens_truth; z_lens=zl, cosmology=cosmo),
    source_plane=Jens.LensGenerator.LightPlane(CompositeImage(host_truth,agn_truth); z=zs),
    grid=grid_gpu, psf=psf)
data = Float64.(Array(render(sys_truth)))
sigma_pix = max(median(abs.(data)) * 0.1, 1e-3)
data_noisy = data .+ sigma_pix .* randn(size(data))

# Scalar-safe lens helpers
defx(lp,x,y;zs)=lens_derivative(lp,[x],[y];z_source=zs)[1][1]
defy(lp,x,y;zs)=lens_derivative(lp,[x],[y];z_source=zs)[2][1]
hexx(lp,x,y;zs)=lens_hessian(lp,[x],[y];z_source=zs)[1][1]
hexy(lp,x,y;zs)=lens_hessian(lp,[x],[y];z_source=zs)[2][1]
heyy(lp,x,y;zs)=lens_hessian(lp,[x],[y];z_source=zs)[3][1]

# Warm-start solver: 1 guess per image instead of 128
function fast_solve(lp, bx, by, warm; zs)
    imgs = Tuple{Float64,Float64,Float64}[]
    for (tx0, ty0) in warm
        function f!(dx, x)
            dx[1] = x[1] - defx(lp, x[1], x[2]; zs) - bx
            dx[2] = x[2] - defy(lp, x[1], x[2]; zs) - by
        end
        sol = nlsolve(f!, [tx0, ty0]; xtol=1e-8, iterations=30)
        if NLsolve.converged(sol)
            sx, sy = sol.zero[1], sol.zero[2]
            detA = abs((1-hexx(lp,sx,sy;zs))*(1-heyy(lp,sx,sy;zs)) - hexy(lp,sx,sy;zs)^2)
            push!(imgs, (sx, sy, abs(1/max(detA, 1e-8))))
        end
    end
    return imgs
end

# Stamp AGN: render on CPU buffer, upload to GPU, add to image
function stamp_agn_gpu!(image, grid, psf, images, flux)
    half = div(grid.pix_n, 2) * Float64(grid.pix_size)
    xm, ym = -half, -half
    ps = grid.pix_size
    T = eltype(image)
    buf_cpu = zeros(T, size(image))
    for (tx, ty, mu) in images
        px = (tx - xm) / ps + 1
        py = (ty - ym) / ps + 1
        render_point!(buf_cpu, psf, px, py, flux * abs(mu); pixel_scale=ps, half=7)
    end
    image .+= CuArray(buf_cpu)
end

# logp_fn: host-only render + manual AGN stamp
warm = Ref([(0.2, -0.23)])

function logp_fast(p)
    theta_E, amp, Rsersic = p
    m = CombinedLens(SIS => (theta_E=theta_E, xcentre=0.0, ycentre=0.0))
    lp = Jens.LensGenerator.LensedPlane(m; z_lens=zl, cosmology=cosmo)

    imgs = fast_solve(lp, 0.05, -0.01, warm[]; zs=zs)
    isempty(imgs) || (warm[] = [(tx, ty) for (tx, ty, _) in imgs])

    h = ExtendedSource(SersicSpheric; amp=amp, Rsersic=Rsersic, n=2.0, xcentre=0.0, ycentre=0.0)
    sys = ForwardModel(lens_plane=lp,
        source_plane=Jens.LensGenerator.LightPlane(h; z=zs),
        grid=grid_gpu, psf=psf)
    img = render(sys)
    stamp_agn_gpu!(img, grid_gpu, psf, imgs, 50.0)

    return -sum((data_noisy .- Float64.(Array(img))).^2) / (2 * sigma_pix^2)
end

# Warmup
logp_fast([0.8, 1.0, 0.3]); logp_fast([0.8, 1.0, 0.3]); logp_fast([0.8, 1.0, 0.3])
CUDA.synchronize()

# Benchmark single call
t = @elapsed(for _ in 1:5; logp_fast([0.8, 1.0, 0.3]); end)
ms = t / 5 * 1000
println("Host + manual AGN: $(round(ms, digits=1)) ms/call  ->  $(round(1000/ms)) steps/sec")

# MCMC
lower = [0.2, 0.1, 0.05]; upper = [2.0, 5.0, 1.0]; init = [0.75, 0.9, 0.35]
warm[] = [(0.2, -0.23)]
t_mh = @elapsed result = lens_mh(logp_fast, lower, upper; n=500, adapt=true, init=init, seed=42)
m, s = chain_stats(result; burn=100)
println("500 steps: $(round(t_mh, digits=1))s ($(round(500/t_mh)) steps/sec)")
for (i, nm, tv) in zip(1:3, ["theta_E", "amp", "Rsersic"], [0.8, 1.0, 0.3])
    println("  $nm: $(round(m[i], digits=3)) +/- $(round(s[i], digits=3))  (truth=$tv)")
end
println("5000 steps: ~$(round(ms*5))s  ($(round(11/ms, digits=1))x vs Method A)")