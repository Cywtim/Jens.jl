#!/usr/bin/env julia
# ═══════════════════════════════════════════════════════════════
#  MCMC Demo: NFW+Shear + Sersic host + AGN (7 free params)
#
#  Lens:   NFW (Rs, alpha_Rs) + Shear (gamma1, gamma2)
#  Light:  Sersic host (amp, Rsersic) + AGN (flux, beta_x, beta_y)
#  Priors: informed by HST-like observations
# ═══════════════════════════════════════════════════════════════

using Cosmology, Random, Statistics, Distributions, LinearAlgebra
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: NFW, Shear
using Jens.LightModel: ExtendedSource, PointImage, CompositeImage
using Jens.LightModel.SersicLight: SersicSpheric
using Jens.LensGenerator: LensedPlane, LightPlane, GenGrid
using Jens.LensSystem: ForwardModel, render

Random.seed!(42)

cosmo  = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_lens, z_src = 0.3, 1.5
sigma_data = 0.025

grid = GenGrid(pix_n=48, pix_size=0.06)
psf  = LensPSF.GaussianPSF(fwhm=0.1)

# ═══════ Truth ═══════
#   NFW:   Rs=5.0   alpha_Rs=1.0
#   Shear: gamma1=0.05  gamma2=-0.02
#   Host:  amp=1.0  Rsersic=0.3  (n=2 fixed)
#   AGN:   flux=100  beta=(0.05, -0.03)
truth = (5.0, 1.0, 0.05, -0.02,  1.0, 0.3,  100.0, 0.05, -0.03)

function make_sys(Rs, aRs, g1, g2, amp, Rser, fagn, bx, by)
    lm = CombinedLens(
        NFW   => (Rs=Rs, alpha_Rs=aRs, xcentre=0.0, ycentre=0.0),
        Shear => (gamma1=g1, gamma2=g2, xcentre=0.0, ycentre=0.0))
    lp = LensedPlane(lm; z_lens=z_lens, cosmology=cosmo)
    host = ExtendedSource(SersicSpheric; amp=amp, Rsersic=Rser, n=2.0)
    agn  = PointImage(flux=fagn, beta_x=bx, beta_y=by)
    src  = CompositeImage(host, agn)
    return ForwardModel(lens_plane=lp, source_plane=LightPlane(src; z=z_src),
                        grid=grid, psf=psf)
end

img_truth = render(make_sys(truth...); solver=:batch)
data = max.(img_truth .+ sigma_data .* randn(size(grid.xg)), 0.0)
println("Grid: $(grid.pix_n)×$(grid.pix_n), S/N≈$(round(maximum(data)/sigma_data,digits=0))")

# ═══════ Priors ═══════
#   Rs       ~ LogNormal(log5, 0.5)      (weak)
#   alpha_Rs ~ LogNormal(log1, 0.3)      (moderate)
#   gamma    ~ Normal(0, 0.08)           (external shear)
#   amp_host ~ LogNormal(log1, 0.3)      (HST photometry)
#   Rsersic  ~ LogNormal(log0.3, 0.25)   (GALFIT)
#   flux_agn ~ LogNormal(log100, 0.5)    (photometry)
#   beta     ~ Normal(truth, 0.03)       (astrometry)

prior_Rs      = LogNormal(log(5.0), 0.5)
prior_aRs     = LogNormal(log(1.0), 0.3)
prior_gamma   = Normal(0.0, 0.08)
prior_amp     = LogNormal(log(1.0), 0.3)
prior_Rser    = LogNormal(log(0.3), 0.25)
prior_fagn    = LogNormal(log(100.0), 0.5)
prior_bx      = Normal(0.05, 0.03)
prior_by      = Normal(-0.03, 0.03)

function log_post(theta)
    Rs, aRs, g1, g2, amp, Rser, fagn, bx, by = theta
    # Hard bounds
    if !(0.5<Rs<15 && 0.1<aRs<3 && -0.3<g1<0.3 && -0.3<g2<0.3 &&
         0.1<amp<5 && 0.05<Rser<1 && 1<fagn<500 && -0.3<bx<0.3 && -0.3<by<0.3)
        return -Inf
    end
    lp = logpdf(prior_Rs, Rs)   + logpdf(prior_aRs, aRs) +
         logpdf(prior_gamma, g1) + logpdf(prior_gamma, g2) +
         logpdf(prior_amp, amp)  + logpdf(prior_Rser, Rser) +
         logpdf(prior_fagn, fagn) + logpdf(prior_bx, bx) + logpdf(prior_by, by)
    img = render(make_sys(Rs, aRs, g1, g2, amp, Rser, fagn, bx, by); solver=:batch)
    chi2 = sum(((data .- img) ./ sigma_data).^2)
    return lp - 0.5 * chi2
end

# ═══════ MCMC ═══════
function mh(log_post, theta0, n, sigma_p)
    d = length(theta0); chain = zeros(n, d); na = 0
    th = copy(theta0); lp = log_post(th)
    for i in 1:n
        th_p = th .+ sigma_p .* randn(d)
        lp_p = log_post(th_p)
        if log(rand()) < lp_p - lp; th=th_p; lp=lp_p; na+=1 end
        chain[i,:] .= th
    end
    return chain, na/n
end

theta0  = [6.0, 1.2, 0.03, -0.01, 1.2, 0.35, 80.0, 0.06, -0.02]
sigma_p = [0.03,0.006, 0.0012,0.0012, 0.008,0.003, 1.0, 0.0008,0.0008]
n_iter, n_burn = 10000, 3000

println("\nM-H $(n_iter) iter, $(length(theta0)) params...")
@time chain, acc = mh(log_post, theta0, n_iter, sigma_p)
cp = chain[n_burn+1:end, :]

println("\n═══ Results (acc=$(round(acc*100,digits=1))%) ═══")
for (j,(name,tv)) in enumerate(zip(
    ["Rs","alpha_Rs","gamma1","gamma2","amp_host","Rsersic","flux_agn","beta_x","beta_y"],
    [truth...]))
    v = cp[:,j]; m, s = mean(v), std(v)
    println("  $(rpad(name,10)) $(rpad(string(tv),8)) → $(round(m,digits=4)) ± $(round(s,digits=4))  ($(round((m-tv)/s,digits=1))σ)")
end
println("Done.")