#!/usr/bin/env julia
using Cosmology, Random, Statistics
using Jens
using Jens.LensTuring: lens_fit_model_simple
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS
using Jens.LightModel: ExtendedSource
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render
using Turing: NUTS, sample
using ADTypes: AutoFiniteDiff
Random.seed!(42)

cosmo = Cosmology.FlatLCDM(0.7,0.3,0.,0.)
zl, zs = 0.3, 1.5
grid = GenGrid(pix_n=32, pix_size=0.09)
psf  = LensPSF.GaussianPSF(fwhm=0.08)

# Truth
lm = CombinedLens(SIS=>(theta_E=0.8, xcentre=0., ycentre=0.))
lp = LensedPlane(lm; z_lens=zl, cosmology=cosmo)
host = ExtendedSource(SersicSpheric; amp=1.0, Rsersic=0.3, n=2.0)
sys = ForwardModel(lens_plane=lp, source_plane=LightPlane(host; z=zs), grid=grid, psf=psf)
data = max.(render(sys; solver=:batch) .+ 0.02 .* randn(size(grid.xg)), 0.0)

println("Running NUTS...")
model = lens_fit_model_simple(data, grid, psf, cosmo, zl, zs)
chain = sample(model, NUTS(adtype=AutoFiniteDiff()), 300; nadapts=100, progress=true)

println("\nResults:")
for name in ["theta_E", "amp", "Rsersic"]
    v = chain[name]; m, s = mean(v), std(v)
    println("  $name = $(round(m,digits=4)) ± $(round(s,digits=4))")
end
println("Done.")