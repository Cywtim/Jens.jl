# test_pointimages_intrinsic_mcmc.jl
# ─────────────────────────────────────────────────────────────────
#  Strategy 2 test: PointImages(intrinsic=true)
#
#  Pipeline:
#    1. Build truth: SIE lens + Sersic source + AGN (PointImage, source-plane)
#    2. Render truth → solve_images gives image positions + magnifications
#    3. Add noise → Observation
#    4. MCMC: fit lens params + source params + intrinsic AGN flux
#       - PointImages(intrinsic=true): 1 intrinsic flux param, mu auto-computed
#       - Image positions fixed (measured from data)
#
#  Fitted params: [theta_E, e1, e2, amp_src, Rsersic, f_intr]  (6 params)
# ─────────────────────────────────────────────────────────────────

using Pkg; Pkg.activate("..")
using Jens, Cosmology, Statistics
using Jens: JFloat
using Jens.LensSolver: solve_images

# ═══════════════════════════════════════════════════════════════
#  1. Setup
# ═══════════════════════════════════════════════════════════════

cosmo = Cosmology.FlatLCDM(0.6736, 0.3153, 0.0, 0.0)
grid  = LensGenerator.GenGrid(pix_n=128, pix_size=Float32(0.05))
psf   = LensPSF.GaussianPSF(fwhm=0.10)

# True parameters
theta_E_true = 1.15
e1_true      = 0.12
e2_true      = -0.04
amp_src      = 0.35
R_src        = 0.18
n_src        = 2.5
amp_lens     = 0.80
R_lens       = 0.45
n_lens       = 4.0
flux_agn     = 25.0      # intrinsic AGN flux (source-plane)
# AGN slightly off-center but well inside the caustic — images stay
# smooth under small lens perturbations
beta_x_agn   = 0.15
beta_y_agn   = 0.10
z_lens       = 0.45
z_source     = 1.69

# Noise
sigma_gauss  = 0.005
exp_time     = 5000.0

println("=" ^ 60)
println("  PointImages(intrinsic=true) — Strategy 2 Test")
println("=" ^ 60)
println("  Grid:       128x128 @ 0.05 arcsec/pix")
println("  PSF:        Gaussian FWHM = 0.10 arcsec")
println("  z_lens:     $z_lens   z_source: $z_source")
println("  AGN:        flux=$flux_agn (intrinsic), beta=($beta_x_agn, $beta_y_agn)")
println()

# ═══════════════════════════════════════════════════════════════
#  2. Build truth system
# ═══════════════════════════════════════════════════════════════

lens = LensBase.SingleModel(LensModel.SIE;
    theta_E=theta_E_true, e1=e1_true, e2=e2_true,
    xcentre=0.0, ycentre=0.0)
lp = LensGenerator.LensedPlane(lens; z_lens=z_lens, cosmology=cosmo)

# Source host (Sersic, lensed)
source_host = LightModel.ExtendedSource(
    LightModel.SersicLight.SersicSpheric;
    amp=amp_src, Rsersic=R_src, n=n_src,
    xcentre=beta_x_agn, ycentre=beta_y_agn)

# AGN as PointImage (source-plane) — truth generation uses solve_images
agn_truth = LightModel.PointImage(;
    flux=flux_agn, beta_x=beta_x_agn, beta_y=beta_y_agn)

source_comp = LightModel.CompositeImage(source_host, agn_truth)

# Lens light (not lensed)
lens_light = LightModel.ExtendedSource(
    LightModel.SersicLight.SersicElliptical;
    amp=amp_lens, Rsersic=R_lens, n=n_lens,
    varphi=0.3, q=0.7, xcentre=0.0, ycentre=0.0)

mlp = LensGenerator.MultiLightPlane(
    lens_light  => z_lens,
    source_comp => z_source)

sys_truth = LensSystem.ForwardModel(;
    lens_plane=lp, source_plane=mlp, grid=grid, psf=psf)

# ═══════════════════════════════════════════════════════════════
#  3. Render truth and extract image positions
# ═══════════════════════════════════════════════════════════════

println("Rendering truth image...")
truth = LensSystem.render(sys_truth)
println("  Truth shape:    ", size(truth))
println("  Truth range:    ", round(minimum(truth); digits=4),
    " -> ", round(maximum(truth); digits=4))
println("  Truth sum:      ", round(sum(truth); digits=2))

# Solve lens equation for the AGN to get image positions + magnifications
println("\nSolving lens equation for AGN source position...")
images = solve_images(lp, beta_x_agn, beta_y_agn; z_source=z_source)
println("  Found ", length(images), " images:")
for (i, (tx, ty, mu)) in enumerate(images)
    println("    Image $i:  (", round(tx; digits=4), ", ", round(ty; digits=4), "),",
             "  mu = ", round(mu; digits=2),
             "  -> observed flux = ", round(flux_agn * abs(mu); digits=2))
end

# Extract image positions (these are "measured" observables in real data)
image_positions = [(JFloat(tx), JFloat(ty)) for (tx, ty, _) in images]

# ═══════════════════════════════════════════════════════════════
#  4. Add noise → Observation
# ═══════════════════════════════════════════════════════════════

noise_model = LensNoise.GaussPoissNoise(sigma_gauss, exp_time)
noisy = JFloat.(LensNoise.add_noise(truth, noise_model))
mask = trues(size(noisy))

obs = LensObservation.Observation(;
    data=noisy, grid=grid, noise=noise_model, mask=mask,
    psf=psf, exposure_time=exp_time,
    header=Dict{String,Any}("FILTER"=>"F160W", "TELESCOP"=>"HST"),
    target="pointimages_intrinsic_sim")

println("\nObservation built:")
println("  data shape:   ", size(obs.data))
println("  noise type:   ", typeof(obs.noise))

# ═══════════════════════════════════════════════════════════════
#  5. Log-likelihood at truth
# ═══════════════════════════════════════════════════════════════

logp_truth = LensSystem.masked_logp(sys_truth, obs.data, obs.noise, obs.mask)
chi2_truth = -2 * logp_truth
n_pix = sum(obs.mask)

println("\nLog-likelihood at truth:")
println("  log P    = ", round(logp_truth; digits=1))
println("  chi2     = ", round(chi2_truth; digits=1))
println("  chi2/dof = ", round(chi2_truth / n_pix; digits=3),
    " (expect ~1.0)")

# ═══════════════════════════════════════════════════════════════
#  6. MCMC — Strategy 2: PointImages(intrinsic=true)
#     Fitted: [theta_E, e1, e2, amp_src, Rsersic, f_intr]
#     f_intr = intrinsic AGN flux (1 param for all images)
#     mu computed automatically from lens_hessian at each step
# ═══════════════════════════════════════════════════════════════

function logp_fn(p)
    theta_E, e1, e2, amp, Rsersic, f_intr = p

    # Guard: intrinsic flux must be positive
    f_intr <= 0 && return -Inf

    # Build lens
    lens_i = LensBase.SingleModel(LensModel.SIE;
        theta_E=theta_E, e1=e1, e2=e2, xcentre=0.0, ycentre=0.0)
    lp_i = LensGenerator.LensedPlane(lens_i; z_lens=z_lens, cosmology=cosmo)

    # Build source (Sersic host)
    src_i = LightModel.ExtendedSource(
        LightModel.SersicLight.SersicSpheric;
        amp=amp, Rsersic=Rsersic, n=n_src,
        xcentre=beta_x_agn, ycentre=beta_y_agn)

    # AGN as PointImages(intrinsic=true):
    #   - 1 intrinsic flux, image positions fixed (measured)
    #   - mu auto-computed from lens_hessian at each MCMC step
    agn_i = LightModel.PointImages(
        (f_intr, image_positions);
        intrinsic=true
    )

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

    lp_val = LensSystem.masked_logp(sys_i, obs.data, obs.noise, obs.mask)
    # Guard against Inf/NaN from divergent magnification
    (isnan(lp_val) || isinf(lp_val)) && return -Inf
    return lp_val
end

# Parameter bounds — tightened around truth to help multistart
#  theta_E  e1      e2      amp_src  Rsersic  f_intr
lower = [0.8,  -0.15,  -0.15,  0.1,     0.05,    10.0]
upper = [1.5,   0.25,   0.15,  0.8,     0.4,     50.0]

# Step sizes: small for lens params, moderate for source, moderate for flux
step0 = [0.02, 0.01, 0.01, 0.02, 0.01, 1.0]

# In practice, MCMC is run AFTER a global optimizer (PSO/gradient) finds
# the basin.  Here we simulate that by starting near truth with small
# perturbation — this tests the intrinsic=true mechanism itself.
init_guess = [theta_E_true, e1_true, e2_true, amp_src, R_src, flux_agn]

# Diagnostic: verify mu at truth via logp_fn's code path
println("\n  Diagnostic: mu at truth parameters via lens_hessian")
lens_diag = LensBase.SingleModel(LensModel.SIE;
    theta_E=theta_E_true, e1=e1_true, e2=e2_true, xcentre=0.0, ycentre=0.0)
lp_diag = LensGenerator.LensedPlane(lens_diag; z_lens=z_lens, cosmology=cosmo)
# Must use Float64 — lens_hessian is 2nd derivative, Float32 is catastrophically wrong
fxx_d, fxy_d, fyy_d = LensSystem.lens_hessian(lp_diag,
    Float64.([p[1] for p in image_positions]),
    Float64.([p[2] for p in image_positions]); z_source=z_source)
for (i, ((tx, ty), mu_truth)) in enumerate(zip(image_positions, [mu for (_,_,mu) in images]))
    detA = (1 - fxx_d[i]) * (1 - fyy_d[i]) - fxy_d[i]^2
    mu_hessian = 1 / abs(detA)
    println("    Image $i: mu(solve_images)=", round(abs(mu_truth); digits=3),
        "  mu(lens_hessian)=", round(mu_hessian; digits=3),
        "  detA=", round(detA; digits=5))
end
println("  logp_fn at truth = ", round(logp_fn(init_guess); digits=1),
    "  (masked_logp truth = ", round(logp_truth; digits=1), ")")

println("\n" * "=" ^ 60)
println("  Running MCMC (6 params, 5000 steps)")
println("  Strategy 2: PointImages(intrinsic=true)")
println("    f_intr = 1 intrinsic flux param for all images")
println("    mu computed from lens_hessian at each step")
println("  Init near truth (simulating post-optimizer start)")
println("=" ^ 60)

result = LensMH.lens_mh(logp_fn, lower, upper;
    n=5000, seed=42, step0=step0, init=init_guess, adapt=true)

m, s = LensMH.chain_stats(result; burn=500)

println("\nPosterior (mean +/- std):")
println("  theta_E  = ", round(m[1]; digits=3), " +/- ", round(s[1]; digits=3),
    "  (truth = ", theta_E_true, ")")
println("  e1       = ", round(m[2]; digits=3), " +/- ", round(s[2]; digits=3),
    "  (truth = ", e1_true, ")")
println("  e2       = ", round(m[3]; digits=3), " +/- ", round(s[3]; digits=3),
    "  (truth = ", e2_true, ")")
println("  amp_src  = ", round(m[4]; digits=3), " +/- ", round(s[4]; digits=3),
    "  (truth = ", amp_src, ")")
println("  Rsersic  = ", round(m[5]; digits=3), " +/- ", round(s[5]; digits=3),
    "  (truth = ", R_src, ")")
println("  f_intr   = ", round(m[6]; digits=3), " +/- ", round(s[6]; digits=3),
    "  (truth = ", flux_agn, ")")
println("  accepted = ", result.accepted, " / 5000")

# ═══════════════════════════════════════════════════════════════
#  7. Verify magnification consistency
# ═══════════════════════════════════════════════════════════════

println("\n" * "-" ^ 40)
println("  Magnification check (best-fit lens model):")

lens_best = LensBase.SingleModel(LensModel.SIE;
    theta_E=m[1], e1=m[2], e2=m[3], xcentre=0.0, ycentre=0.0)
lp_best = LensGenerator.LensedPlane(lens_best; z_lens=z_lens, cosmology=cosmo)

# Recompute mu at image positions with best-fit lens
# Must use Float64 for lens_hessian (2nd derivative precision)
fxx, fxy, fyy = LensSystem.lens_hessian(lp_best,
    Float64.([p[1] for p in image_positions]),
    Float64.([p[2] for p in image_positions]);
    z_source=z_source)

println("  Image |  mu(truth)  mu(best-fit)  obs_flux(truth)  obs_flux(best)")
for (i, ((tx, ty), mu_truth)) in enumerate(zip(image_positions, [mu for (_,_,mu) in images]))
    # Must use Float64 for lens_hessian (2nd derivative precision)
    detA = (1 - fxx[i]) * (1 - fyy[i]) - fxy[i]^2
    mu_best = 1 / abs(detA)
    obs_truth = flux_agn * abs(mu_truth)
    obs_best  = m[6] * abs(mu_best)
    println("    $i   |  ", round(abs(mu_truth); digits=2),
        "      ", round(mu_best; digits=2),
        "       ", round(obs_truth; digits=2),
        "          ", round(obs_best; digits=2))
end

# ═══════════════════════════════════════════════════════════════
#  8. Best-fit model logp
# ═══════════════════════════════════════════════════════════════

src_best = LightModel.ExtendedSource(
    LightModel.SersicLight.SersicSpheric;
    amp=m[4], Rsersic=m[5], n=n_src, xcentre=beta_x_agn, ycentre=beta_y_agn)
agn_best = LightModel.PointImages(
    (m[6], image_positions); intrinsic=true)
src_comp_best = LightModel.CompositeImage(src_best, agn_best)

mlp_best = LensGenerator.MultiLightPlane(
    lens_light  => z_lens,
    src_comp_best => z_source)

sys_best = LensSystem.ForwardModel(;
    lens_plane=lp_best, source_plane=mlp_best,
    grid=obs.grid, psf=obs.psf, mask=obs.mask)

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
println("  Strategy:   PointImages(intrinsic=true)")
println("  Simulated:  SIE + Sersic lens light + Sersic source + AGN")
println("  AGN images: ", length(image_positions), " (positions fixed, from solve_images)")
println("  Fitted:     theta_E, e1, e2, amp_src, Rsersic, f_intr (6 params)")
println("  Fixed:      n_sersic, lens light, AGN positions")
println()

# Recovery check
let
    truth_vals = [theta_E_true, e1_true, e2_true, amp_src, R_src, flux_agn]
    names = ["theta_E", "e1", "e2", "amp_src", "Rsersic", "f_intr"]
    all_recovered = true
    for i in 1:6
        if abs(m[i] - truth_vals[i]) > 3 * max(s[i], 1e-6)
            println("  WARNING: $(names[i]) not recovered within 3-sigma")
            all_recovered = false
        end
    end

    if all_recovered
        println("  All parameters recovered within 3-sigma. PASS")
    else
        println("  Some parameters outside 3-sigma. Check convergence.")
    end
end
