#!/usr/bin/env julia --project=.
# ═══════════════════════════════════════════════════════════════
#  demo_shapelet_mcmc.jl
#
#  End-to-end MCMC: SIS lens + shapelet source reconstruction.
#
#  Demonstrates:
#    - Mock data generation (shapelet-generated truth)
#    - build_design / solve_coeffs inside MCMC logp
#    - lens_mh_multistart chain + diagnostics
#    - Posterior and residual visualisation
# ═══════════════════════════════════════════════════════════════

using Jens, Cosmology, Statistics, LinearAlgebra, Random, Printf, Plots
using Jens.LensShapelet: ShapeletBasis, n_basis, build_design,
    solve_coeffs, shapelet_model, build_reg_matrix
using Jens.LensMH: lens_mh_multistart, lens_mh, chain, chain_stats

Random.seed!(42)

# ═══════════════════════════════════════════════════════════════
#  1.  Truth model — SIS + shapelet source
#     Using shapelet-generated truth so the model is well-specified.
#     (Previous GaussianSphere truth was a model mismatch — shapelets
#      can't represent a Sersic cusp → χ² ~ 10⁶ → MCMC froze.)
# ═══════════════════════════════════════════════════════════════
println("=== Generating mock data (shapelet truth) ===\n")

cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
grid  = Jens.LensGenerator.GenGrid(pix_n=64)   # 64² = 4096 pix; masked ~1000

# Truth parameters
THETA_E_TRUE = 0.8
BETA_TRUE    = 0.10
SRC_X_TRUE   = 0.03
SRC_Y_TRUE   = 0.02
NMAX_TRUE    = 6                                # 28 basis functions
SIGMA_NOISE  = 0.003

n_basis_true = (NMAX_TRUE + 1) * (NMAX_TRUE + 2) ÷ 2

# Build truth lens + design matrix
lens_truth = Jens.LensBase.SingleModel(
    Jens.LensModel.SIS.SIS;
    theta_E=THETA_E_TRUE, xcentre=0.0, ycentre=0.0)

# Dummy source (unused for rendering — we render via shapelet coefficients)
sys_truth = Jens.LensSystem.ForwardModel(;
    lens_plane   = Jens.LensGenerator.LensedPlane(
        lens_truth; z_lens=0.3, cosmology=cosmo),
    source_plane = Jens.LightModel.ExtendedSource(
        Jens.LightModel.GaussianLight.GaussianSphere;
        amp=1.0, sigma=0.01),
    grid         = grid,
    z_source     = 1.5,
)

mask = Jens.LensMask.circular_mask(grid, 2.0)
mask_idx = findall(vec(mask))
n_pix = length(mask_idx)

basis_true = ShapeletBasis(NMAX_TRUE, BETA_TRUE)

# Placeholder data (needed for build_design API — gets overwritten)
tmp_data = zeros(size(grid.xg))

A_masked_true, A_conv_true, _ = build_design(
    sys_truth, basis_true, tmp_data, mask;
    xc_src=SRC_X_TRUE, yc_src=SRC_Y_TRUE)

# Generate random shapelet coefficients (sparse-ish)
c_truth = randn(n_basis_true) .* 0.3
c_truth[1] = 1.0                                  # DC component
# Set a few low-order modes
c_truth[2] = 0.3                                  # n1=1,n2=0 → dipole x
c_truth[3] = -0.2                                 # n1=0,n2=1 → dipole y

# Render truth image
data_true = reshape(A_conv_true * c_truth, size(grid.xg))
data_true_max = maximum(abs, data_true)
data_true ./= data_true_max                        # normalise to ~1

# Re-render with normalised coefficients
c_truth ./= data_true_max
data_true = reshape(A_conv_true * c_truth, size(grid.xg))

# Add noise
data = data_true .+ SIGMA_NOISE .* randn(size(data_true))
snr = maximum(data[mask]) / SIGMA_NOISE

println("  θ_E truth  = $THETA_E_TRUE")
println("  β truth    = $BETA_TRUE")
println("  src centre = ($SRC_X_TRUE, $SRC_Y_TRUE)")
println("  noise σ    = $SIGMA_NOISE")
println("  peak S/N   = $(round(snr, digits=1))")
println("  masked pix = $n_pix")
println("  n_basis    = $n_basis_true")
println()

# ═══════════════════════════════════════════════════════════════
#  2.  MCMC setup
# ═══════════════════════════════════════════════════════════════

# For the MCMC fit we use the SAME n_max as truth so the model is
# perfectly specified.  The MCMC samples only 4 parameters:
#   [theta_E, beta, xc_src, yc_src]

SHAPELET_NMAX_MCMC = NMAX_TRUE    # must = truth for well-specified model
N_MCMC = (SHAPELET_NMAX_MCMC + 1) * (SHAPELET_NMAX_MCMC + 2) ÷ 2

println("=== MCMC setup ===")
println("  MCMC n_max  = $SHAPELET_NMAX_MCMC ($N_MCMC coefficients)")
println("  Design size = $n_pix × $N_MCMC")
println()

# ═══════════════════════════════════════════════════════════════
#  3.  Log-probability function
# ═══════════════════════════════════════════════════════════════

"""
    safe_shapelet_logp(params)

MCMC samples: params = [theta_E, beta, xc_src, yc_src]

1. Build ForwardModel with current lens params
2. Compute design matrix A via build_design
3. Solve linear system c = (AᵀA + λI)⁻¹ Aᵀd
4. Return log-prob = -χ²/2

Uses invokelatest to avoid Julia 1.12 world-age issues with
dynamically-compiled closures.
"""
function safe_shapelet_logp(params)
    return Base.invokelatest(params) do p
        theta_E = p[1]
        beta    = p[2]
        xc_src  = p[3]
        yc_src  = p[4]

        # Hard bounds (prior)
        if theta_E < 0.1 || theta_E > 2.5 ||
           beta < 0.01 || beta > 0.50 ||
           abs(xc_src) > 0.3 || abs(yc_src) > 0.3
            return -Inf
        end

        lens = Jens.LensBase.SingleModel(
            Jens.LensModel.SIS.SIS;
            theta_E=theta_E, xcentre=0.0, ycentre=0.0)

        sys = Jens.LensSystem.ForwardModel(;
            lens_plane   = Jens.LensGenerator.LensedPlane(
                lens; z_lens=0.3, cosmology=cosmo),
            source_plane = Jens.LightModel.ExtendedSource(
                Jens.LightModel.GaussianLight.GaussianSphere;
                amp=1.0, sigma=0.01),
            grid         = grid,
            z_source     = 1.5,
        )

        b = ShapeletBasis(SHAPELET_NMAX_MCMC, beta)

        # build_design needs data for mask handling (not used otherwise)
        A_masked, _, _ = build_design(sys, b, data, mask;
                                       xc_src=xc_src, yc_src=yc_src)

        _, chi2 = solve_coeffs(A_masked, data, mask_idx, SIGMA_NOISE;
                                lambda=0.01)

        return -chi2 / 2
    end
end

# ═══════════════════════════════════════════════════════════════
#  4.  Run MCMC
# ═══════════════════════════════════════════════════════════════
println("=== Running MCMC ===\n")

lower = [0.3,   0.03,    -0.15,  -0.15]
upper = [2.0,   0.30,     0.15,   0.15]
#         θ_E    β       xc_src   yc_src

n_params = length(lower)

# Multi-start with generous warmup to find the posterior and adapt steps
result = lens_mh_multistart(safe_shapelet_logp, lower, upper;
                             n_starts=24, n_warmup=1500, n=2000, seed=42)

samples = chain(result; burn=500)
means, stds = chain_stats(result; burn=500)

acc_rate = result.accepted / 2000 * 100
println("  Acceptance rate: $(round(acc_rate, digits=1))%")
println()

println("─── MCMC Results ───")
labels = ["θ_E", "β", "xc_src", "yc_src"]
truths = [THETA_E_TRUE, BETA_TRUE, SRC_X_TRUE, SRC_Y_TRUE]
println(rpad("  Parameter", 12), rpad("Truth", 10), rpad("Posterior", 24), "Δ")
for i in 1:n_params
    delta = means[i] - truths[i]
    sigma_delta = delta / stds[i]
    println(@sprintf("  %-10s  %6.3f   %6.3f ± %6.3f   %+.3f  (%+.1fσ)",
                     labels[i], truths[i], means[i], stds[i], delta, sigma_delta))
end

# ═══════════════════════════════════════════════════════════════
#  5.  Reconstruct best-fit model
# ═══════════════════════════════════════════════════════════════
println("\n=== Reconstructing best-fit model ===\n")

best_theta_E = means[1]
best_beta    = means[2]
best_xc      = means[3]
best_yc      = means[4]

lens_best = Jens.LensBase.SingleModel(
    Jens.LensModel.SIS.SIS;
    theta_E=best_theta_E, xcentre=0.0, ycentre=0.0)

sys_best = Jens.LensSystem.ForwardModel(;
    lens_plane   = Jens.LensGenerator.LensedPlane(
        lens_best; z_lens=0.3, cosmology=cosmo),
    source_plane = Jens.LightModel.ExtendedSource(
        Jens.LightModel.GaussianLight.GaussianSphere;
        amp=1.0, sigma=0.01),
    grid         = grid,
    z_source     = 1.5,
)

basis_best = ShapeletBasis(SHAPELET_NMAX_MCMC, best_beta)
A_masked_best, A_conv_best, _ = build_design(
    sys_best, basis_best, data, mask;
    xc_src=best_xc, yc_src=best_yc)
c_best, chi2_best = solve_coeffs(A_masked_best, data, mask_idx, SIGMA_NOISE;
                                  lambda=0.01)
model_best = shapelet_model(A_conv_best, c_best, size(grid.xg, 1), size(grid.xg, 2))

residual = (data .- model_best) .* mask
chi2_red = chi2_best / (n_pix - N_MCMC)
max_res = maximum(abs, residual[mask])

println("  reduced χ² = ", round(chi2_red, digits=3))
println("  max residual = ", round(max_res, digits=5),
        "  ($(round(max_res/SIGMA_NOISE, digits=1))σ)")
println()

# ═══════════════════════════════════════════════════════════════
#  6.  Plot
# ═══════════════════════════════════════════════════════════════
println("=== Plotting ===\n")

xvec = grid.xg[:, 1]
yvec = grid.yg[1, :]
clim_data = (-0.05, maximum(data[mask]) * 1.1)

p1 = heatmap(xvec, yvec, data';
    aspect_ratio=:equal, title="Mock (S/N≈$(round(snr,digits=1)))",
    c=:magma, xlabel="x [arcsec]", ylabel="y [arcsec]",
    clim=clim_data, colorbar=true)

p2 = heatmap(xvec, yvec, model_best';
    aspect_ratio=:equal, title="Shapelet (n_max=$NMAX_TRUE)",
    c=:magma, xlabel="x [arcsec]", ylabel="y [arcsec]",
    clim=clim_data, colorbar=true)

p3 = heatmap(xvec, yvec, residual';
    aspect_ratio=:equal, title="Residual (σ=$SIGMA_NOISE)",
    c=:RdBu, xlabel="x [arcsec]", ylabel="y [arcsec]",
    clim=(-4*SIGMA_NOISE, 4*SIGMA_NOISE), colorbar=true)

fig = plot(p1, p2, p3; layout=(1, 3), size=(1500, 460),
    plot_title="Shapelet MCMC  |  " *
    "θ_E=$(round(best_theta_E,digits=3))  " *
    "β=$(round(best_beta,digits=3))  " *
    "χ²_ν=$(round(chi2_red,digits=2))")

outpath = joinpath(@__DIR__, "demo_shapelet_mcmc.png")
savefig(fig, outpath)
println("Saved → $outpath")
println("Done.")