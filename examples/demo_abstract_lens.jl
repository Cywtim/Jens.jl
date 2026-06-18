#!/usr/bin/env julia
# =============================================================================
#  AbstractLens Demo — Full Pipeline
#
#  Shows all four lens_* functions, critical curves, caustics,
#  magnification, ray shooting, and Fermat potential — using:
#    - LensModule (raw Module → AbstractLens wrapper)
#    - CombinedLens (multi-component)
#    - LensedPlane (cosmological D_ls/D_s scaling)
#    - WithTidal (LOS tidal matrix)
#    - MultiLensedPlane (multi-plane)
# =============================================================================

using Jens
using Plots; default(fmt=:png, dpi=150)

# Grab all top-level exports in one place
const LB  = Jens.LensBase
const LG  = Jens.LensGenerator
const LLM = Jens.LensModel
const LU  = Jens.LensUtils
using Cosmology

# ---------------------------------------------------------------------------
#  1. Setup — coordinate grid
# ---------------------------------------------------------------------------
r_max = 2.5       # arcsec
N     = 201
xg    = range(-r_max, r_max; length=N)
yg    = range(-r_max, r_max; length=N)
XG    = [x for x in xg, _ in yg]
YG    = [y for _ in xg, y in yg]

println("Grid: $(size(XG))  ×  [±$r_max arcsec]")

# ---------------------------------------------------------------------------
#  2. LENS-MODEL BASICS  (SIS via LensModule)
# ---------------------------------------------------------------------------
θE       = 1.2
sis_raw  = LLM.SIS   # raw Module
sis_lm   = LB.LensModule(sis_raw)    # wrap → AbstractLens

kw_sis = (; theta_E=θE, xcentre=0.0, ycentre=0.0)

# --- lensing potential ψ(θ) ---
psi      = LB.lens_potential(sis_lm, XG, YG; kw_sis...)

# --- deflection α(θ) ---
αx, αy   = LB.lens_derivative(sis_lm, XG, YG; kw_sis...)

# --- Hessian f_xx, f_xy, f_yy ---
fxx, fxy, fyy = LB.lens_hessian(sis_lm, XG, YG; kw_sis...)

# --- magnification μ(θ) ---
detJ = @. (1.0 - fxx) * (1.0 - fyy) - fxy^2
μ    = @. 1.0 / detJ

println("ψ  range: [$(minimum(psi)), $(maximum(psi))]")
println("α  range: [$(minimum(αx)), $(maximum(αx))]")
println("μ  range: [$(minimum(μ)), $(maximum(μ))]")

# --- quick plot ---
p1 = heatmap(xg, yg, psi', title="Potential ψ(θ)",           aspect_ratio=1)
p2 = heatmap(xg, yg, sqrt.(αx.^2 .+ αy.^2)', title="|α(θ)|", aspect_ratio=1)
p3 = heatmap(xg, yg, μ',  title="Magnification μ(θ)",         aspect_ratio=1, clims=(-10,10))
display(plot(p1, p2, p3; layout=(1,3), size=(1200,350)))

# ---------------------------------------------------------------------------
#  3. CRITICAL CURVE  &  CAUSTIC  (SIS)
# ---------------------------------------------------------------------------
# Uniform-grid approach
ccx, ccy   = LB.LensCriticalCurve(; LensModel=sis_lm,
               LensKwargs=Dict(:theta_E=>θE), hperr=0.01,
               r_max=2.5, r_bins=2000, theta_bins=2000)
csx, csy   = LB.LensCaustic(;     LensModel=sis_lm,
               LensKwargs=Dict(:theta_E=>θE), hperr=0.01,
               r_max=2.5, r_bins=2000, theta_bins=2000)
println("Critical curve: $(length(ccx)) points, Caustic: $(length(csx)) points")

# Adaptive-grid (quadtree) — much fewer Hessian evals for same accuracy
ccx_a, ccy_a = LB.LensAdaptiveCriticalCurve(; LensModel=sis_lm,
               LensKwargs=Dict(:theta_E=>θE),
               xlim=(-2.5,2.5), ylim=(-2.5,2.5),
               initial_nx=16, initial_ny=16, max_depth=6)
csx_a, csy_a = LB.LensAdaptiveCaustic(;     LensModel=sis_lm,
               LensKwargs=Dict(:theta_E=>θE),
               xlim=(-2.5,2.5), ylim=(-2.5,2.5),
               initial_nx=16, initial_ny=16, max_depth=6)
println("Adaptive critical curve: $(length(ccx_a)) pts, Caustic: $(length(csx_a)) pts")

# Plot
scatter(ccx, ccy;   ms=0.5, mc=:red,  label="CC (uniform)", msw=0)
scatter!(csx, csy;  ms=0.5, mc=:cyan, label="Caustic (uniform)", msw=0)
scatter!(ccx_a, ccy_a; ms=1.5, mc=:orange, label="CC (adaptive)", msw=0, shape=:x)
scatter!(csx_a, csy_a; ms=1.5, mc=:green,  label="Caustic (adaptive)", msw=0, shape=:x)
title!("SIS Critical Curve & Caustic"); xlabel!("θ₁ [arcsec]"); ylabel!("θ₂")
display(current())

# ---------------------------------------------------------------------------
#  4. RAY SHOOTING  (lens plane, source plane)
# ---------------------------------------------------------------------------
# LensPlane:   β = θ - α(θ)
βx, βy = LB.LensPlane(XG, YG; LensModel=sis_lm,
                      LensKwargs=Dict(:theta_E=>θE))

# RayShooting: source-plane light for a Gaussian source
gaussian_source(xs, ys; sigma=0.1) = @. exp(-(xs^2 + ys^2) / (2sigma^2)) / (2π*sigma^2)
light = LB.LensRayShooting(XG, YG;
         LensModel=sis_lm,
         LensKwargs=Dict(:theta_E=>θE),
         SourceProfile=gaussian_source,
         SourceKwargs=Dict(:sigma=>0.1))

p4 = heatmap(xg, yg, βx', title="Lens Plane β₁")
p5 = heatmap(xg, yg, βy', title="Lens Plane β₂")
p6 = heatmap(xg, yg, light', title="Ray-shooted light")
display(plot(p4, p5, p6; layout=(1,3), size=(1200,350)))

# ---------------------------------------------------------------------------
#  5. FERMAT POTENTIAL
# ---------------------------------------------------------------------------
β_source  = [0.3, -0.2]   # source position
fermat     = LB.LensFermat(XG, YG, β_source;
               LensModel=sis_lm,
               LensKwargs=Dict(:theta_E=>θE))

heatmap(xg, yg, fermat', title="Fermat Φ(θ) — source at $(β_source)",
        aspect_ratio=1); display(current())

# ---------------------------------------------------------------------------
#  6. COMBINED LENS  (SIS + external Shear + NFW)
# ---------------------------------------------------------------------------
cl = LLM.ComLens.CombinedLens(
    LLM.SIS => (theta_E=1.0, xcentre=0.0, ycentre=0.0),
    LLM.Shear => (gamma1=0.05, gamma2=0.03, xcentre=0.0, ycentre=0.0),
)

ψ_cl      = LB.lens_potential(cl, XG, YG)
αx_c, αy_c = LB.lens_derivative(cl, XG, YG)
ccx_c, ccy_c = LB.LensAdaptiveCriticalCurve(;
    LensModel=cl, LensKwargs=Dict(),
    xlim=(-2.5,2.5), ylim=(-2.5,2.5))
csx_c, csy_c = LB.LensAdaptiveCaustic(;
    LensModel=cl, LensKwargs=Dict(),
    xlim=(-2.5,2.5), ylim=(-2.5,2.5))

println("CombinedLens — ψ range: [$(minimum(ψ_cl)), $(maximum(ψ_cl))]")
scatter(ccx_c, ccy_c; ms=1, mc=:red, label="CC", msw=0)
scatter!(csx_c, csy_c; ms=1, mc=:cyan, label="Caustic", msw=0)
title!("SIS+Shear CombinedLens"); display(current())


# ---------------------------------------------------------------------------
#  7. COSMOLOGICAL SCALING  (LensedPlane)
# ---------------------------------------------------------------------------
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
zL    = 0.3
zS    = 1.5

lp = LG.LensedPlane(cl; z_lens=zL, cosmology=cosmo)

# D_ls/D_s ratio
ratio = Jens.LensCosmo.lens_distance_ratio(cosmo, zL, zS)
println("D_ls/D_s = $(round(ratio; digits=3))  @ zL=$zL, zS=$zS")

# Check: deflection with LensedPlane is scaled by ratio
αx_lp, αy_lp = LB.lens_derivative(lp, XG, YG; z_source=zS)
println("α scaled: ratio × α_raw = $(ratio) × $(maximum(abs.(αx_c))) ≈ $(maximum(abs.(αx_lp)))")

ccx_lp, ccy_lp = LB.LensAdaptiveCriticalCurve(;
    LensModel=lp, LensKwargs=Dict(), z_source=zS,
    xlim=(-2.5,2.5), ylim=(-2.5,2.5))
csx_lp, csy_lp = LB.LensAdaptiveCaustic(;
    LensModel=lp, LensKwargs=Dict(), z_source=zS,
    xlim=(-2.5,2.5), ylim=(-2.5,2.5))

scatter(ccx_lp, ccy_lp; ms=1, mc=:red, label="CC (cosmo)", msw=0)
scatter!(csx_lp, csy_lp; ms=1, mc=:cyan, label="Caustic (cosmo)", msw=0)
title!("LensedPlane — zL=$zL, zS=$zS"); display(current())

# ---------------------------------------------------------------------------
#  8. LOS TIDAL EFFECTS  (WithTidal)
# ---------------------------------------------------------------------------
tidal = Jens.LensLOS.ExternalTidal(0.05, 0.02, -0.01)
wt    = Jens.LensLOS.WithTidal(lp, tidal)

αx_wt, αy_wt = LB.lens_derivative(wt, XG, YG; z_source=zS)   # unchanged by design
ψ_wt          = LB.lens_potential(wt, XG, YG; z_source=zS)         # adds LOS quadrupole

# Hessian includes T_ext · A_lens
fxx_wt, fxy_wt, fyy_wt = LB.lens_hessian(wt, XG, YG; z_source=zS)
detJ_wt = @. (1 - fxx_wt)*(1 - fyy_wt) - fxy_wt^2
μ_wt    = @. 1.0 / detJ_wt

ccx_wt, ccy_wt = LB.LensAdaptiveCriticalCurve(;
    LensModel=wt, LensKwargs=Dict(), z_source=zS,
    xlim=(-2.5,2.5), ylim=(-2.5,2.5))
csx_wt, csy_wt = LB.LensAdaptiveCaustic(;
    LensModel=wt, LensKwargs=Dict(), z_source=zS,
    xlim=(-2.5,2.5), ylim=(-2.5,2.5))

p7 = heatmap(xg, yg, μ',  title="μ (no LOS)", clims=(-10,10))
p8 = heatmap(xg, yg, μ_wt',  title="μ (with tidal)", clims=(-10,10))
display(plot(p7, p8; layout=(1,2), size=(800,350)))

scatter(ccx_wt, ccy_wt; ms=1, mc=:red, label="CC (tidal)", msw=0)
scatter!(csx_wt, csy_wt; ms=1, mc=:cyan, label="Caustic (tidal)", msw=0)
title!("WithTidal — SIS+Shear @ zL=$zL, κ_ext=0.05"); display(current())

# ---------------------------------------------------------------------------
#  9. MULTI-PLANE LENSING  (MultiLensedPlane)
# ---------------------------------------------------------------------------
# Multi-plane: each plane gets the SAME kwargs dict.
# Limitation: planes with different param names (SIS vs Shear) can't share.
# Workaround: use two SIS planes at different redshifts.
ml = LG.MultiLensedPlane((
    (LLM.SIS, 0.3),
    (LLM.SIS, 0.8),
); z_source=1.5, cosmology=cosmo)

# Same SIS params → both planes take theta_E, xcentre, ycentre
kw_mp = (; theta_E=0.6, xcentre=0.0, ycentre=0.0)
αx_ml, αy_ml = LB.lens_derivative(ml, XG, YG; kw_mp...)

println("MultiLensedPlane — 2× SIS @ z=0.3, 0.8")
println("α multi-plane range: [$(minimum(αx_ml)), $(maximum(αx_ml))]")

# ---------------------------------------------------------------------------
#  10. SUMMARY — TYPE HIERARCHY
# ---------------------------------------------------------------------------
println("""

══════════════════════════════════════════════════════
  AbstractLens hierarchy snapshot
══════════════════════════════════════════════════════
  typeof(sis_lm)  = $(typeof(sis_lm))          # LensModule{SIS}
  typeof(cl)      = $(typeof(cl))              # CombinedLens{...}
  typeof(lp)      = $(typeof(lp))              # LensedPlane{CombinedLens, ...}
  typeof(wt)      = $(typeof(wt))              # WithTidal{LensedPlane, ...}
  typeof(ml)      = $(typeof(ml))              # MultiLensedPlane{...}

  sis_lm <: AbstractLens = $(sis_lm isa Jens.LensBase.AbstractLens)
  cl     <: AbstractLens = $(cl     isa Jens.LensBase.AbstractLens)
  lp     <: AbstractLens = $(lp     isa Jens.LensBase.AbstractLens)
  wt     <: AbstractLens = $(wt     isa Jens.LensBase.AbstractLens)
  ml     <: AbstractLens = $(ml     isa Jens.LensBase.AbstractLens)
══════════════════════════════════════════════════════
""")
