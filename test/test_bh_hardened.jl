# Validate hardened Barnes-Hut:
#  1. quadtree_bh_tree + quadtree_deflection_bh vs exact nsub=1 (accuracy)
#  2. QuadTreeLens(method=:bh) full ForwardModel render — timing vs NIE
#  3. ROI-optimised tree + BH together
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensSystem: ForwardModel, render
using Jens.LensGenerator: GenGrid, LensedPlane, LightPlane
using Jens.LensBase: SingleModel
using Jens.LightModel: ExtendedSource
using Jens.LightModel.GaussianLight: GaussianSphere
using Jens.LensCosmo: Cosmology
import Jens.LensModel: NIEkappa, NIE
using Jens.LensMeshRefine: n_leaves

const b,s,q,varphi = 1.2,0.15,0.75,pi/6
function nie_kappa(xs,ys)
    fxx,fxy,fyy = NIEkappa.LensHessian(xs,ys;b=b,s=s,q=q,varphi=varphi)
    return (fxx.+fyy)./2
end
EXT=3.0; nx=111
c=collect(LinRange(-EXT,EXT,nx)); xs=Float64[]; ys=Float64[]
for y in c, x in c; push!(xs,x); push!(ys,y); end
kt=nie_kappa(xs,ys)

function build(roi, cap)
    mf=MassField(-EXT,EXT,-EXT,EXT;max_level=8,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
    rs=collect(kt)
    for it in 1:30
        accumulate_residuals!(mf,xs,ys,rs); update_field!(mf)
        refine_mass!(mf;tau=5e-4,part=0.4,roi=roi,cap_outside=cap)
        rs .= kt .- field_value(mf,xs,ys)
    end
    return mf
end

mf = build(1.2, 4)
L = n_leaves(mf.tree)
println("ROI quadtree: leaves=$L")

# ═══ accuracy: BH vs exact (nsub=1) on arc window ═══
gd=200; ge=collect(range(-1.0,1.0;length=gd))
Xg=repeat(reshape(ge,:,1),1,gd); Yg=repeat(reshape(ge,1,:),gd,1)
ax_t,ay_t=NIE.LensDerivative(Xg,Yg;theta_E=1.2,s_scale=0.15,e1=0.1,e2=0.0)
mag=sqrt.(ax_t.^2 .+ ay_t.^2); m_arc=mag.>0.2

println("\n=== 1) BH vs exact (nsub=1), arc window ===")
t0=time(); ax_ex,ay_ex=quadtree_deflection(mf,vec(Xg),vec(Yg);nsub=1); t_ex=time()-t0
root=quadtree_bh_tree(mf)
for th in (0.3,0.5,0.8)
    t0=time(); axb,ayb=quadtree_deflection_bh(root,vec(Xg),vec(Yg);theta=th); t_bh=time()-t0
    err=sqrt.((axb.-ax_ex).^2 .+ (ayb.-ay_ex).^2)./(sqrt.(ax_ex.^2 .+ ay_ex.^2) .+ 1e-6)
    println("  θ=$th: BH=$(round(t_bh*1000;digits=1))ms  exact=$(round(t_ex*1000;digits=0))ms  " *
            "speedup=$(round(t_ex/t_bh;digits=1))×  |α|err med=$(round(100*median(vec(err));digits=3))%  p90=$(round(100*quantile(vec(err),0.9);digits=3))%")
end

# ═══ 2) full render: QuadTreeLens exact vs BH vs NIE ═══
src = ExtendedSource(GaussianSphere; amp=1.0, sigma=0.05, xcentre=0.18, ycentre=0.10)
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
grid = GenGrid(pix_n=200, pix_size=0.018)
lens_nie = SingleModel(NIE; theta_E=1.2, s_scale=0.15, e1=0.1, e2=0.0)
sys_nie = ForwardModel(lens_plane=LensedPlane(lens_nie; z_lens=0.5, cosmology=cosmo),
                       source_plane=LightPlane(src; z=1.5), grid=grid)
render(sys_nie); render(sys_nie)
t0=time(); img_nie=render(sys_nie); t_nie=time()-t0

lens_ex = QuadTreeLens(mf)                         # exact (default)
lens_bh = QuadTreeLens(mf; method=:bh, theta=0.5)
sys_ex = ForwardModel(lens_plane=LensedPlane(lens_ex; z_lens=0.5, cosmology=cosmo),
                      source_plane=LightPlane(src; z=1.5), grid=grid)
sys_bh = ForwardModel(lens_plane=LensedPlane(lens_bh; z_lens=0.5, cosmology=cosmo),
                      source_plane=LightPlane(src; z=1.5), grid=grid)

println("\n=== 2) 完整 render：QuadTreeLens(ROI树) exact vs BH vs NIE ===")
render(sys_ex); t0=time(); img_ex=render(sys_ex); t1=time()-t0
render(sys_bh); t0=time(); img_bh=render(sys_bh); t2=time()-t0
println("  NIE render          : $(round(t_nie*1000;digits=1)) ms")
println("  quadtree exact render: $(round(t1*1000;digits=0)) ms  ($(round(t1/t_nie;digits=0))× NIE)")
println("  quadtree BH render  : $(round(t2*1000;digits=1)) ms  ($(round(t2/t_nie;digits=1))× NIE)")

# image-level agreement: BH image vs exact image (same lens, near-nominal)
mimg = img_ex .> 0.05*maximum(img_ex)
relimg = abs.(img_bh .- img_ex) ./ (img_ex .+ 1e-6)
println("\n  BH image vs exact image (bright px): med=$(round(100*median(vec(relimg[mimg]));digits=2))%")
println("  BH image vs NIE (true): bright-px residual med=$(round(100*median(vec(abs.(img_bh .- img_nie)./(img_nie .+ 1e-6))[vec(img_nie .> 0.05*maximum(img_nie))]);digits=1))%")
