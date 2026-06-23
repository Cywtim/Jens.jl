#!/usr/bin/env julia
using Cosmology
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: NFW, Shear, SIS
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensSystem: ForwardModel
using Jens.LensBase: LensCriticalCurve, LensCaustic, LensAdaptiveCriticalCurve, LensAdaptiveCaustic

cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens, z_src = 0.3, 1.5

# ── Lens model ────────────────────────────────
lens_mass = CombinedLens(
    SIS   => (theta_E=0.8, xcentre=0.0, ycentre=0.0),
    Shear => (gamma1=0.05, gamma2=-0.02, xcentre=0.0, ycentre=0.0),
)
lp = LensedPlane(lens_mass; z_lens=z_lens, cosmology=cosmo)

src = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0)
grid = GenGrid(pix_n=128, pix_size=0.04)

sys = ForwardModel(
    lens_plane   = lp,
    source_plane = LightPlane(src; z=z_src),
    grid         = grid,
)

# ── Critical curve (image plane) ──────────────
println("Computing critical curve...")
@time ccx, ccy = LensCriticalCurve(;
    LensModel   = sys,
    LensKwargs  = NamedTuple(),    # 空，参数已 baked 在 lens_plane 里
    r_max       = 2.0,
    r_bins      = 2000,
    theta_bins  = 2000,
)

# ── Caustic (source plane) ────────────────────
println("Computing caustic...")
@time csx, csy = LensCaustic(;
    LensModel   = sys,
    LensKwargs  = NamedTuple(),
    r_max       = 2.0,
    r_bins      = 2000,
    theta_bins  = 2000,
)

println("Critical curve: $(length(ccx)) points")
println("Caustic:        $(length(csy)) points")

# ── Adaptive (higher resolution, recommended) ─
println("\nAdaptive critical curve...")
@time ccx_a, ccy_a = LensAdaptiveCriticalCurve(;
    LensModel   = sys,
    LensKwargs  = NamedTuple(),
    max_depth   = 8,
    hp_threshold = 1e-5,
)
println("Caustic (mapped from adaptive CC)...")
@time csx_a, csy_a = LensAdaptiveCaustic(;
    LensModel   = sys,
    LensKwargs  = NamedTuple(),
    max_depth   = 8,
    hp_threshold = 1e-5,
)

println("Adaptive CC: $(length(ccx_a)) points")
println("Adaptive caustic: $(length(csy_a)) points")
println("\nDone.")