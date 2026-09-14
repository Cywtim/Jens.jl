using Pkg; Pkg.activate(".")
using Jens, Cosmology, Statistics

obs = LensFITS.read_fits("../img/HE0435−1223.fits";
    noise = :gauss_poiss,
    noise_method = :clipped,
    noise_kwargs = (clip_sigma=3.0, max_iter=5),
    center = (976, 844), radius=32
)

println("=== Observation 检查 ===")
println("  data shape:  ", size(obs.data))
println("  pix_n:       ", obs.grid.pix_n)
println("  pix_size:    ", obs.grid.pix_size, " arcsec/pix")
println("  noise type:  ", typeof(obs.noise))
println("  noise σ_g:   ", obs.noise.sigma_gauss)
println("  noise t_exp: ", obs.noise.exp_time)
println("  exposure:    ", obs.exposure_time)
println("  psf:         ", obs.psf)

cosmo = Cosmology.FlatLCDM(0.6736, 0.3153, 0.0, 0.0)

lens = LensBase.SingleModel(LensModel.SIE;
    theta_E = 1.2, e1 = 0.15, e2 = -0.05,
    xcentre = 0.05, ycentre = -0.03
)
lp = LensGenerator.LensedPlane(lens; z_lens=0.45, cosmology=cosmo)

src = LightModel.ExtendedSource(LightModel.SersicLight.SersicSpheric;
    amp = 0.3, Rsersic = 0.2, n = 2.0,
    xcentre = 0.02, ycentre = 0.01
)
sp = LensGenerator.LightPlane(src; z=1.69)

sys = LensSystem.ForwardModel(;
    lens_plane   = lp,
    source_plane = sp,
    grid         = obs.grid,
    psf          = obs.psf,
    mask         = obs.mask
)

println("\n=== ForwardModel 构建成功 ===")

img = LensSystem.render(sys)
println("  model shape:    ", size(img))
println("  data shape:     ", size(obs.data))
println("  shape match:    ", size(img) == size(obs.data))

# GaussPoissNoise 没有 .σ 字段，用 LensNoise 路径
logp_noise = LensSystem.masked_logp(sys, obs.data, obs.noise, obs.mask)
println("  logp (LensNoise):", round(logp_noise; digits=2))

n_pix = sum(obs.mask)
chi2 = -2 * logp_noise
println("  chi2:            ", round(chi2; digits=1))
println("  n_pix:           ", n_pix)
println("  chi2/dof:        ", round(chi2 / n_pix; digits=3))

# 也测试标量 sigma 路径（用 sigma_gauss 近似）
logp_scalar = LensSystem.masked_logp(sys, obs.data, obs.noise.sigma_gauss, obs.mask)
println("  logp (sigma_g):  ", round(logp_scalar; digits=2))

# 测试 with_psf
println("\n=== with_psf 测试 ===")
psf = LensPSF.GaussianPSF(fwhm=0.09)
obs2 = LensObservation.with_psf(obs, psf)
println("  obs.psf:  ", obs.psf)
println("  obs2.psf: ", obs2.psf)
println("  obs2.noise === obs.noise: ", obs2.noise === obs.noise)

sys2 = LensSystem.ForwardModel(;
    lens_plane   = lp,
    source_plane = sp,
    grid         = obs2.grid,
    psf          = obs2.psf,
    mask         = obs2.mask
)

img2 = LensSystem.render(sys2)
logp2 = LensSystem.masked_logp(sys2, obs2.data, obs2.noise, obs2.mask)
println("  logp (with PSF): ", round(logp2; digits=2))

println("\n=== 全部通过 ===")
