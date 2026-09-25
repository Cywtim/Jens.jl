# Scheme A quick validation: adaptive RELATIVE refinement threshold.
#   Stock:   tau = const (5e-4) → 2509 leaves, near-uniform fill.
#   SchemeA: tau = max(base, q × quantile(all-leaf resid, p)) each pass
#            → only structurally-outstanding leaves get split.
# Does NOT modify LensMassRecon; only changes how tau is passed.
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
import Jens.LensModel: NIEkappa, NIE
const b,s,q,varphi = 1.2,0.15,0.75,pi/6
function nie_kappa(xs,ys)
    fxx,fxy,fyy = NIEkappa.LensHessian(xs,ys;b=b,s=s,q=q,varphi=varphi)
    return (fxx.+fyy)./2
end
EXT=3.0; nx=111
c=collect(LinRange(-EXT,EXT,nx)); xs=Float64[]; ys=Float64[]
for y in c, x in c
    push!(xs,x); push!(ys,y)
end
kt=nie_kappa(xs,ys)

# evaluation window (image plane / arc region)
gd=200; ge=collect(range(-1.0,1.0;length=gd))
Xg=repeat(reshape(ge,:,1),1,gd); Yg=repeat(reshape(ge,1,:),gd,1)
xv=vec(Xg); yv=vec(Yg)
ax_t,ay_t = NIE.LensDerivative(Xg,Yg;theta_E=1.2,s_scale=0.15,e1=0.1,e2=0.0)
mag_t = sqrt.(ax_t.^2 .+ ay_t.^2)

function run_refine(; tau0=5e-4, p=0.0, qmult=1.0, maxlv=8, iters=25)
    mf=MassField(-EXT,EXT,-EXT,EXT;max_level=maxlv,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
    rs=collect(kt)
    for it in 1:iters
        accumulate_residuals!(mf,xs,ys,rs)
        update_field!(mf)
        # adaptive tau: quantile of leaf resid among LEAVES THAT COULD refine
        if p > 0
            res=[lf.resid for lf in mf.tree.leaves if lf.level < mf.tree.max_level && lf.n_obs >= mf.min_obs]
            tau = isempty(res) ? tau0 : max(tau0, qmult * quantile(res, p))
        else
            tau = tau0
        end
        refine_mass!(mf; tau=tau, part=0.4)
        rs .= kt .- field_value(mf,xs,ys)
    end
    L=Jens.LensMeshRefine.n_leaves(mf.tree)
    lvs=[lf.level for lf in mf.tree.leaves]
    # deflection in arc window
    axq,ayq = quadtree_deflection(mf, Xg, Yg; nsub=1)
    scl = sum(ax_t.*axq .+ ay_t.*ayq)/sum(axq.^2 .+ ayq.^2)
    err = sqrt.((ax_t .- scl.*axq).^2 .+ (ay_t .- scl.*ayq).^2) ./ (mag_t .+ 1e-6)
    # error restricted to bright-arc pixels (|α|>0.2) — what matters for imaging
    m_arc = mag_t .> 0.2
    rms = sqrt(mean(rs.^2))
    return (L=L, dist=join(["[$(k)]$(count(==(k),lvs))" for k in sort(unique(lvs))]," "),
            rms=rms, med=100*median(vec(err)), med_arc=100*median(vec(err[m_arc])),
            p90=100*quantile(vec(err),0.9), scl=scl)
end

println("=== 方案 A: 相对分位数 tau vs 固定 tau ===")
base = run_refine()
println(rpad("固定 tau=5e-4 (现状)",26), " 叶=$(base.L)  rms=$(round(base.rms;sigdigits=3))  ",
        "arc-window |α|err: med=$(round(base.med;sigdigits=3))%  med_arc=$(round(base.med_arc;sigdigits=3))%  p90=$(round(base.p90;sigdigits=3))%")
println("   level分布: $(base.dist)")
println()
for (p, qm) in ((0.5,2.0), (0.5,4.0), (0.75,2.0), (0.9,1.5), (0.9,3.0))
    r = run_refine(p=p, qmult=qm)
    println(rpad("A: p=$(p) q=$(qm)",26), " 叶=$(r.L)  rms=$(round(r.rms;sigdigits=3))  ",
            "arc-window |α|err: med=$(round(r.med;sigdigits=3))%  med_arc=$(round(r.med_arc;sigdigits=3))%  p90=$(round(r.p90;sigdigits=3))%")
    println("   level分布: $(r.dist)")
end
