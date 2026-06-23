#!/usr/bin/env julia
using Cosmology
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: NFW, Shear
using Jens.LightModel: ExtendedSource, PointImage, CompositeImage
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, MultiLightPlane
using Jens.LensSystem: ForwardModel, render

cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
lens_mass = CombinedLens(
    NFW   => (Rs=5.0, alpha_Rs=1.0,  xcentre=0.0, ycentre=0.0),
    Shear => (gamma1=0.05, gamma2=-0.02, xcentre=0.0, ycentre=0.0),
)
lp = LensedPlane(lens_mass; z_lens=0.3, cosmology=cosmo)
z_src = 1.5

agn = PointImage(flux=100.0, beta_x=0.05, beta_y=-0.03)
host = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0)
src = CompositeImage(host, agn)

grid = LensGenerator.GenGrid(pix_n=256, pix_size=0.04)
psf = WFC3.WFC3_UVIS_PSF(lambda_eff=606e-9, oversample=5)

sys = ForwardModel(
    lens_plane   = lp,
    source_plane = LightPlane(src; z=z_src),
    grid         = grid,
    psf          = psf,
)

# Warmup
render(sys; solver=:nlsolve)
render(sys; solver=:batch)

println("solver=:nlsolve")
@time img_nlsolve = render(sys; solver=:nlsolve)
println("  sum=$(round(sum(img_nlsolve),digits=2)), max=$(round(maximum(img_nlsolve),digits=4))")

println("\nsolver=:batch")
@time img_batch = render(sys; solver=:batch)
println("  sum=$(round(sum(img_batch),digits=2)), max=$(round(maximum(img_batch),digits=4))")

println("\ndifference: max=$(round(maximum(abs.(img_nlsolve .- img_batch)), digits=8))")
println("Done.")