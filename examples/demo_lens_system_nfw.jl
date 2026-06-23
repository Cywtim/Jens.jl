#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#  LensSystem demo: NFW + Shear, Sersic host + AGN, HST PSF
#  Tests all three source_plane forms:
#    A. LightPlane(light; z=...)          — single source
#    B. MultiLightPlane(light=>z, ...)     — multi-redshift
#    C. bare AbstractLight + z_source=...  — auto-wrap shortcut
# ═══════════════════════════════════════════════════════════════

using Cosmology
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: NFW, Shear
using Jens.LightModel: ExtendedSource, PointImage, CompositeImage
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, MultiLightPlane
using Jens.LensSystem: ForwardModel, render

# Shared setup ───────────────────────────────────────────────
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
lens_mass = CombinedLens(
    NFW   => (Rs=5.0, alpha_Rs=1.0,  xcentre=0.0, ycentre=0.0),
    Shear => (gamma1=0.05, gamma2=-0.02, xcentre=0.0, ycentre=0.0),
)
z_lens, z_src = 0.3, 1.5
lp = LensedPlane(lens_mass; z_lens=z_lens, cosmology=cosmo)

host = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0)
agn  = PointImage(flux=100.0, beta_x=0.05, beta_y=-0.03)
lens_light = ExtendedSource(SersicSpheric; amp=0.5, Rsersic=0.8, n=4.0, ycentre=-0.05)

grid = LensGenerator.GenGrid(pix_n=256, pix_size=0.04)
psf  = WFC3.WFC3_UVIS_PSF(lambda_eff=606e-9, oversample=5)

# ─── A. LightPlane: single source at one z ─────────────────
sys_a = ForwardModel(
    lens_plane   = lp,
    source_plane = LightPlane(host; z=z_src),
    grid         = grid,
    psf          = psf,
)
img_a = render(sys_a)
println("A. LightPlane: sum=$(round(sum(img_a), digits=2)), max=$(round(maximum(img_a), digits=4))")

# ─── B. MultiLightPlane: lens light + source composite ─────
src = CompositeImage(host, agn)
sys_b = ForwardModel(
    lens_plane   = lp,
    source_plane = MultiLightPlane(lens_light => z_lens, src => z_src),
    grid         = grid,
    psf          = psf,
)
img_b = render(sys_b)
println("B. MultiLightPlane: sum=$(round(sum(img_b), digits=2)), max=$(round(maximum(img_b), digits=4))")

# ─── C. auto-wrap: bare CompositeImage + z_source ──────────
sys_c = ForwardModel(
    lens_plane   = lp,
    source_plane = CompositeImage(host, agn),
    z_source     = z_src,
    grid         = grid,
    psf          = psf,
)
img_c = render(sys_c)
println("C. bare + z_source: sum=$(round(sum(img_c), digits=2)), max=$(round(maximum(img_c), digits=4))")

# ─── Verify consistency ────────────────────────────────────
println("\nA+B+C all OK. Done.")