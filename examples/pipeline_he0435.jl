# ═══════════════════════════════════════════════════════════════
#  pipeline_he0435.jl  —  Full FITS → ForwardModel → logP demo
#
#  Demonstrates the LensFITS pipeline on real HST/WFC3 data:
#      HE0435-1223 (quadruply-imaged quasar, F160W, 0.09"/pix)
#
#  Usage:
#      cd /path/to/Jens.jl
#      julia --project=. examples/pipeline_he0435.jl
# ═══════════════════════════════════════════════════════════════

using Pkg; Pkg.activate(".")
using Jens, Cosmology, Statistics

# ═══════════════════════════════════════════════════════════════
#  1. Load FITS → Observation
# ═══════════════════════════════════════════════════════════════

println("═══ Step 1: read_fits ═══")

obs = LensFITS.read_fits("img/HE0435−1223.fits";
    noise_method = :clipped,
    noise_kwargs = (clip_sigma=3.0, max_iter=5))

println("  Telescope:  ", LensObservation.telescope(obs))
println("  Instrument: ", LensObservation.instrument(obs))
println("  Filter:     ", LensObservation.filter_name(obs))
println("  Data shape: ", size(obs.data))
println("  Pixel scale:", round(LensObservation.pixel_scale(obs); digits=3), " arcsec/pix")
println("  FOV:        ", round(size(obs.data, 1) * LensObservation.pixel_scale(obs); digits=1), " arcsec")
println("  Flux range: ", round(minimum(obs.data); digits=3), " → ",
                                   round(maximum(obs.data); digits=1), " e⁻/s")
println("  Noise σ:    ", round(obs.noise.σ; digits=6), "  (σ-clipped)")
println("  Exposure:   ", round(obs.exposure_time; digits=0), " s")

# ═══════════════════════════════════════════════════════════════
#  2. Build lens + source model
# ═══════════════════════════════════════════════════════════════

println("\n═══ Step 2: Build model ═══")

cosmo = Cosmology.FlatLCDM(0.6736, 0.3153, 0.0, 0.0)   # Planck 2018

# Lens: Singular Isothermal Ellipsoid at z_lens ≈ 0.45
lens = LensBase.SingleModel(LensModel.SIE;
    theta_E = 1.2,
    e1      = 0.15,
    e2      = -0.05,
    xcentre = 0.05,
    ycentre = -0.03,
)
lp = LensGenerator.LensedPlane(lens; z_lens=0.45, cosmology=cosmo)

# Source: Spherical Sérsic at z_source ≈ 1.69
src = LightModel.ExtendedSource(LightModel.SersicLight.SersicSpheric;
    amp     = 0.3,
    Rsersic = 0.2,
    n       = 2.0,
    xcentre = 0.02,
    ycentre = 0.01,
)
sp = LensGenerator.LightPlane(src; z=1.69)

println("  Lens:  SIE(θ_E=1.2, e1=0.15, e2=-0.05) @ z=0.45")
println("  Src:   Sersic(r_eff=0.2, n=2) @ z=1.69")

# ═══════════════════════════════════════════════════════════════
#  3. ForwardModel → render
# ═══════════════════════════════════════════════════════════════

println("\n═══ Step 3: render ═══")

sys = LensSystem.ForwardModel(;
    lens_plane   = lp,
    source_plane = sp,
    grid         = obs.grid,     # ← from Observation
    # psf          = obs.psf,    # attach PSF here when needed
)

img = LensSystem.render(sys)
println("  Model shape: ", size(img))
println("  Model range: ", round(minimum(img); digits=4), " → ",
                                   round(maximum(img); digits=4))

# ═══════════════════════════════════════════════════════════════
#  4. Evaluate log-likelihood
# ═══════════════════════════════════════════════════════════════

println("\n═══ Step 4: masked_logp ═══")

logp = LensSystem.masked_logp(sys, obs.data, obs.noise.σ, obs.mask)
chi2 = -2 * logp
n_pix = sum(obs.mask)
residual = obs.data .- img
rms_res = sqrt(mean(residual .^ 2))

println("  χ²        = ", round(chi2; digits=1))
println("  χ² / dof  = ", round(chi2 / n_pix; digits=3), "  (n = ", n_pix, " pixels)")
println("  log P     = ", round(logp; digits=1))
println("  RMS_res   = ", round(rms_res; digits=6))
println("  RMS_res/σ = ", round(rms_res / obs.noise.σ; digits=2))

# ═══════════════════════════════════════════════════════════════
#  5. Quick stats
# ═══════════════════════════════════════════════════════════════

println("\n═══ Summary ═══")
ss_tot = sum((obs.data .- mean(obs.data)) .^ 2)
ss_res = sum(residual .^ 2)
r2 = 1 - ss_res / ss_tot
println("  Data RMS:  ", round(sqrt(mean(obs.data .^ 2)); digits=4))
println("  Model RMS: ", round(sqrt(mean(img .^ 2)); digits=4))
println("  R² ≈       ", round(r2; digits=3), "  (negative = model worse than data mean)")
println()
println("  → χ² is huge because a single SIE+Sersic cannot fit a 4-image lens.")
println("  → This is a *pipeline skeleton* — add PSF, complex lens, and MCMC to fit properly.")