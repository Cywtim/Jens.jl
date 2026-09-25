# Scheme C validation: SPATIAL goal-oriented refinement.
# tau depends on leaf DISTANCE to the ROI (arc window): near leaves use
# strict tau (split), far leaves use loose tau (don't waste budget).
# Reimplements refine_mass!'s Dörfler MARK/CUT/SPLIT with a spatial tau,
# purely in-script (LensMassRecon untouched).
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: QuadLeaf, split_leaf, n_leaves
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

# ROI: arc window [-1.2, 1.2] square (covers the Einstein arc region)
ROI = 1.2
# spatial tau: strict (tau0) inside ROI, looser outside by (d/ROI)^α
function spatial_tau(lf; tau0=5e-4, tau_far=8e-3, alpha=2.0)
    cx=(lf.xmin+lf.xmax)/2; cy=(lf.ymin+lf.ymax)/2
    # distance from leaf centre to ROI square (0 if inside)
    dx = max(0.0, abs(cx)-ROI); dy = max(0.0, abs(cy)-ROI)
    d = sqrt(dx*dx+dy*dy)
    if d <= 0.0
        return tau0
    else
        return tau0 + (tau_far-tau0)*min(1.0,(d/ROI)^alpha)
    end
end

# spatial refine (mirror of refine_mass! with lf-dependent tau)
function refine_spatial!(mf; part=0.4)
    leaves=mf.tree.leaves
    candidates=Int[]
    for (k,lf) in enumerate(leaves)
        t = spatial_tau(lf)
        ok = lf.level < mf.tree.max_level && lf.n_obs >= mf.min_obs && lf.resid > t
        ok && push!(candidates,k)
    end
    sort!(candidates; by=k->mf.tree.leaves[k].resid, rev=true)
    isempty(candidates) && return mf
    total=sum(lf.resid for k in candidates for lf in (leaves[k],))
    ns=0; acc=0.0
    if total>0
        for k in candidates
            acc+=leaves[k].resid; ns+=1
            acc>=part*total && break
        end
    end
    to_split=Set(candidates[1:ns])
    new_leaves=QuadLeaf[]
    for (k,lf) in enumerate(leaves)
        k in to_split ? append!(new_leaves, split_leaf(lf)) : push!(new_leaves,lf)
    end
    mf.tree.leaves=new_leaves
    resize!(mf.mom, 10*length(new_leaves)); fill!(mf.mom,0)
    return mf
end

function run(; spatial::Bool, tau0=5e-4, maxlv=8)
    mf=MassField(-EXT,EXT,-EXT,EXT;max_level=maxlv,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
    rs=collect(kt)
    for it in 1:30
        accumulate_residuals!(mf,xs,ys,rs); update_field!(mf)
        if spatial
            refine_spatial!(mf)
        else
            refine_mass!(mf;tau=tau0,part=0.4)
        end
        rs .= kt .- field_value(mf,xs,ys)
    end
    return mf
end

gd=200; ge=collect(range(-1.0,1.0;length=gd))
Xg=repeat(reshape(ge,:,1),1,gd); Yg=repeat(reshape(ge,1,:),gd,1)
ax_t,ay_t=NIE.LensDerivative(Xg,Yg;theta_E=1.2,s_scale=0.15,e1=0.1,e2=0.0)
mag_t=sqrt.(ax_t.^2 .+ ay_t.^2); m_arc=mag_t.>0.2

function report(name, mf)
    L=n_leaves(mf.tree)
    lvs=[lf.level for lf in mf.tree.leaves]
    # fraction of leaves inside ROI
    in_roi=count(lf -> max(abs((lf.xmin+lf.xmax)/2),abs((lf.ymin+lf.ymax)/2))<=ROI, mf.tree.leaves)
    axq,ayq=quadtree_deflection(mf,Xg,Yg;nsub=1)
    scl=sum(ax_t.*axq .+ ay_t.*ayq)/sum(axq.^2 .+ ayq.^2)
    err=sqrt.((ax_t .- scl.*axq).^2 .+ (ay_t .- scl.*ayq).^2)./(mag_t .+ 1e-6)
    println(rpad(name,32), " 叶=", rpad(L,6), " 弧区叶=", rpad(in_roi,7), "($(round(100*in_roi/L;digits=1))%)  ",
            "αerr 全窗med=", rpad("$(round(100*median(vec(err));sigdigits=3))%",10),
            "弧区med=", rpad("$(round(100*median(vec(err[m_arc]));sigdigits=3))%",9),
            "p90=", "$(round(100*quantile(vec(err),0.9);sigdigits=3))%")
    println("        level分布: ", join(["[$(j)]$(count(==(j),lvs))" for j in sort(unique(lvs))]," "))
    return L
end

println("=== 方案 C: 空间目标导向细化 vs 基准 ===")
mf0 = run(spatial=false)        # baseline: uniform tau=5e-4
println("基准 (均匀 tau=5e-4):")
report("", mf0)
mfA = run(spatial=false, tau0=4e-3)   # naive global bigger tau (旧方案A极端)
println("\n全局 tau=4e-3 (方案A简化):")
report("", mfA)
println("\n方案 C (空间感知: ROI 内 tau=5e-4, 外 → 8e-3):")
mfC = run(spatial=true)
report("", mfC)
