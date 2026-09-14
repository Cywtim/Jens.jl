# test_simulate_lens_system.jl
# ─────────────────────────────────────────────────────────────────
#  Complete simulation pipeline:
#    1. Build a lens system (SIE + Sersic lens light + Sersic source + AGN)
#    2. Render noise-free model image
#    3. Add Gaussian+Poisson noise
#    4. Build Observation from simulated data
#    5. Evaluate log-likelihood
#    6. MCMC fit to recover parameters
# ─────────────────────────────────────────────────────────────────

using Pkg; Pkg.activate("..")
using Jens, Cosmology, Statistics
using Jens: JFloat

# ═══════════════════════════════════════════════════════════════
#  1. Setup: cosmology + grid + PSF + noise model
# ═══════════════════════════════════════════════════════════════

cosmo = Cosmology.FlatLCDM(0.6736, 0.3153, 0.0, 0.0)
grid  = LensGenerator.GenGrid(pix_n=128, pix_size=Float32(0.05))
psf   = LensPSF.GaussianPSF(fwhm=0.10)

# True parameters
θ_E_true   = 1.15
e1_true    = 0.12
e2_true    = -0.04
amp_src    = 0.35
R_src      = 0.18
n_src      = 2.5
amp_lens   = 0.80
R_lens     = 0.45
n_lens     = 4.0
e1_lens    = 0.10
e2_lens    = 0.03
flux_agn   = 25.0
z_lens     = 0.45
z_source   = 1.69

# Noise parameters
σ_gauss    = 0.005      # read noise [e-/s]
exp_time   = 5000.0     # exposure time [s]

println("=" ^ 60)
println("  Lens System Simulation Test")
println("=" ^ 60)
println("  Grid:       128×128 @ 0.05 arcsec/pix")
println("  PSF:        Gaussian FWHM = 0.10 arcsec")
println("  z_lens:     $z_lens")
println("  z_source:   $z_source")
println("  Noise:      GaussPoiss(σ_g=$σ_gauss, t_exp=$exp_time)")
println()

# ═══════════════════════════════════════════════════════════════
#  2. Build lens model
# ═══════════════════════════════════════════════════════════════

lens = LensBase.SingleModel(LensModel.SIE;
    theta_E = θ_E_true,
    e1      = e1_true,
    e2      = e2_true,
    xcentre = 0.0,
    ycentre = 0.0,
)
lp = LensGenerator.LensedPlane(lens; z_lens=z_lens, cosmology=cosmo)

# ═══════════════════════════════════════════════════════════════
#  3. Build light components
# ═══════════════════════════════════════════════════════════════

# (a) Lens galaxy light — Sersic at z_lens, NOT lensed
lens_light = LightModel.ExtendedSource(
    LightModel.SersicLight.SersicElliptical;
    amp     = amp_lens,
    Rsersic = R_lens,
    n       = n_lens,
    varphi  = 0.3,
    q       = 0.7,
    xcentre = 0.0,
    ycentre = 0.0,
)

# (b) Source host galaxy — Sersic at z_source, lensed
source_host = LightModel.ExtendedSource(
    LightModel.SersicLight.SersicSpheric;
    amp     = amp_src,
    Rsersic = R_src,
    n       = n_src,
    xcentre = 0.01,
    ycentre = 0.01,
)

# (c) AGN point source — at z_source, lensed
agn = LightModel.PointImage(;
    flux   = flux_agn,
    beta_x = 0.01,
    beta_y = 0.01,
)

# Combine source components at same redshift
source_comp = LightModel.CompositeImage(source_host, agn)

# MultiLightPlane: lens light at z_lens + source at z_source
mlp = LensGenerator.MultiLightPlane(
    lens_light   => z_lens,
    source_comp  => z_source,
)

# ═══════════════════════════════════════════════════════════════
#  4. Build ForwardModel and render noise-free image
# ═══════════════════════════════════════════════════════════════

sys_truth = LensSystem.ForwardModel(;
    lens_plane   = lp,
    source_plane = mlp,
    grid         = grid,
    psf          = psf,
)

println("Rendering truth image...")
truth = LensSystem.render(sys_truth)
println("  Truth shape:    ", size(truth))
println("  Truth range:    ", round(minimum(truth); digits=4),
    " -> ", round(maximum(truth); digits=4))
println("  Truth sum:      ", round(sum(truth); digits=2))

# ═══════════════════════════════════════════════════════════════
#  5. Add noise and build Observation
# ═══════════════════════════════════════════════════════════════

noise_model = LensNoise.GaussPoissNoise(σ_gauss, exp_time)
noisy = JFloat.(LensNoise.add_noise(truth, noise_model))

mask = trues(size(noisy))

obs = LensObservation.Observation(;
    data          = noisy,
    grid          = grid,
    noise         = noise_model,
    mask          = mask,
    psf           = psf,
    exposure_time = exp_time,
    header        = Dict{String, Any}("FILTER" => "F160W", "TELESCOP" => "HST"),
    target        = "simulation",
)

println("\nObservation built:")
println("  data shape:   ", size(obs.data))
println("  noise type:   ", typeof(obs.noise))
println("  psf type:     ", typeof(obs.psf))
println("  exposure:     ", obs.exposure_time, " s")

# ═══════════════════════════════════════════════════════════════
#  6. Evaluate log-likelihood at truth parameters
# ═══════════════════════════════════════════════════════════════

logp_truth = LensSystem.masked_logp(sys_truth, obs.data, obs.noise, obs.mask)
chi2_truth = -2 * logp_truth
n_pix      = sum(obs.mask)

println("\nLog-likelihood at truth:")
println("  log P    = ", round(logp_truth; digits=1))
println("  chi2     = ", round(chi2_truth; digits=1))
println("  chi2/dof = ", round(chi2_truth / n_pix; digits=3),
    " (expect ~1.0 for noise-only residual)")

# ═══════════════════════════════════════════════════════════════
#  7. MCMC: fit lens params (theta_E, e1, e2) + source params (amp, Rsersic)
# ═══════════════════════════════════════════════════════════════

function logp_fn(p)
    theta_E, e1, e2, amp, Rsersic = p

    # Build lens
    lens_i = LensBase.SingleModel(LensModel.SIE;
        theta_E=theta_E, e1=e1, e2=e2, xcentre=0.0, ycentre=0.0)
    lp_i = LensGenerator.LensedPlane(lens_i; z_lens=z_lens, cosmology=cosmo)

    # Build source (Sersic + AGN, keep AGN fixed)
    src_i = LightModel.ExtendedSource(
        LightModel.SersicLight.SersicSpheric;
        amp=amp, Rsersic=Rsersic, n=n_src,
        xcentre=0.01, ycentre=0.01)
    agn_i = LightModel.PointImage(; flux=flux_agn, beta_x=0.01, beta_y=0.01)
    src_comp = LightModel.CompositeImage(src_i, agn_i)

    # Lens light (fixed)
    lens_lt = LightModel.ExtendedSource(
        LightModel.SersicLight.SersicElliptical;
        amp=amp_lens, Rsersic=R_lens, n=n_lens,
        varphi=0.3, q=0.7, xcentre=0.0, ycentre=0.0)

    mlp_i = LensGenerator.MultiLightPlane(
        lens_lt  => z_lens,
        src_comp => z_source)

    sys_i = LensSystem.ForwardModel(;
        lens_plane=lp_i, source_plane=mlp_i,
        grid=obs.grid, psf=obs.psf, mask=obs.mask)

    return LensSystem.masked_logp(sys_i, obs.data, obs.noise, obs.mask)
end

lower = [0.5, -0.3, -0.3, 0.01, 0.01]
upper = [2.0,  0.3,  0.3, 2.0,  1.0]

println("\n" * "=" ^ 60)
println("  Running MCMC (5 params, 8 starts, 2000 steps)...")
println("=" ^ 60)

result = LensMH.lens_mh_multistart(logp_fn, lower, upper;
    n_starts=8, n=2000, seed=42,
    step0=[0.02, 0.01, 0.01, 0.02, 0.01])

m, s = LensMH.chain_stats(result; burn=500)

println("\nPosterior (mean +/- std):")
println("  theta_E  = ", round(m[1]; digits=3), " +/- ", round(s[1]; digits=3),
    "  (truth = ", θ_E_true, ")")
println("  e1       = ", round(m[2]; digits=3), " +/- ", round(s[2]; digits=3),
    "  (truth = ", e1_true, ")")
println("  e2       = ", round(m[3]; digits=3), " +/- ", round(s[3]; digits=3),
    "  (truth = ", e2_true, ")")
println("  amp_src  = ", round(m[4]; digits=3), " +/- ", round(s[4]; digits=3),
    "  (truth = ", amp_src, ")")
println("  Rsersic  = ", round(m[5]; digits=3), " +/- ", round(s[5]; digits=3),
    "  (truth = ", R_src, ")")
println("  accepted = ", result.accepted, " / 2000")

# ═══════════════════════════════════════════════════════════════
#  8. Render best-fit model
# ═══════════════════════════════════════════════════════════════

lens_best = LensBase.SingleModel(LensModel.SIE;
    theta_E=m[1], e1=m[2], e2=m[3], xcentre=0.0, ycentre=0.0)
lp_best = LensGenerator.LensedPlane(lens_best; z_lens=z_lens, cosmology=cosmo)

src_best = LightModel.ExtendedSource(
    LightModel.SersicLight.SersicSpheric;
    amp=m[4], Rsersic=m[5], n=n_src, xcentre=0.01, ycentre=0.01)
src_comp_best = LightModel.CompositeImage(src_best, agn)

mlp_best = LensGenerator.MultiLightPlane(
    lens_light  => z_lens,
    src_comp_best => z_source)

sys_best = LensSystem.ForwardModel(;
    lens_plane=lp_best, source_plane=mlp_best,
    grid=obs.grid, psf=obs.psf, mask=obs.mask)

img_best = LensSystem.render(sys_best)
logp_best = LensSystem.masked_logp(sys_best, obs.data, obs.noise, obs.mask)

println("\nBest-fit:")
println("  log P    = ", round(logp_best; digits=1),
    "  (truth = ", round(logp_truth; digits=1), ")")
println("  chi2/dof = ", round(-2*logp_best / n_pix; digits=3))

# ═══════════════════════════════════════════════════════════════
#  9. Summary
# ═══════════════════════════════════════════════════════════════

println("\n" * "=" ^ 60)
println("  Pipeline complete.")
println("=" ^ 60)
println("  Simulated:  SIE lens + Sersic lens light + Sersic source + AGN")
println("  Noise:      GaussPoiss(sigma=$σ_gauss, t_exp=$exp_time)")
println("  Fitted:     theta_E, e1, e2, amp_src, Rsersic (5 params)")
println("  Fixed:      n_sersic, lens light, AGN flux/position")
println()

# Check recovery
recovered = true
for (i, (name, truth_val)) in enumerate([
        ("theta_E", θ_E_true), ("e1", e1_true), ("e2", e2_true),
        ("amp_src", amp_src), ("Rsersic", R_src)])
    if abs(m[i] - truth_val) > 3 * s[i]
        println("  WARNING: $name not recovered within 3-sigma")
        recovered = false
    end
end

if recovered
    println("  All parameters recovered within 3-sigma. PASS")
else
    println("  Some parameters outside 3-sigma. Check convergence.")
end
