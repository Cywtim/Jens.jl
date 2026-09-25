# Hybrid: refine deeply ONLY inside ROI; outside ROI cap at max_level=4.
#   inside  [-ROI,ROI]² :  tau=5e-4, max depth = maxlv
#   outside            :  tau=5e-4, max depth = 4   (coarse but present)
# In-script refinement (LensMassRecon untouched).
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

ROI=1.2
OUTSIDE_MAXLV=4

function refine_hybrid!(mf; tau=5e-4, part=0.4, maxlv=8)
    leaves=mf.tree.leaves
    candidates=Int[]
    for (k,lf) in enumerate(leaves)
        cx=(lf.xmin+lf.xmax)/2; cy=(lf.ymin+lf.ymax)/2
        in_roi = abs(cx)<=ROI && abs(cy)<=ROI
        cap = in_roi ? maxlv : OUTSIDE_MAXLV
        ok = lf.level < cap && lf.n_obs >= mf.min_obs && lf.resid > tau
        ok && push!(candidates,k)
    end
    sort!(candidates; by=k->leaves[k].resid, rev=true)
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
    resize!(mf.mom,10*length(new_leaves)); fill!(mf.mom,0)
    return mf
end

function run(; maxlv=8, tau=5e-4, OUTSIDE=OUTSIDE_MAXLV, iters=30)
    global OUTSIDE_MAXLV = OUTSIDE
    mf=MassField(-EXT,EXT,-EXT,EXT;max_level=maxlv,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
    rs=collect(kt)
    for it in 1:iters
        accumulate_residuals!(mf,xs,ys,rs); update_field!(mf)
        refine_hybrid!(mf;tau=tau,maxlv=maxlv)
        rs .= kt .- field_value(mf,xs,ys)
    end
    return mf
end

gd=200; ge=collect(range(-1.0,1.0;length=gd))
Xg=repeat(reshape(ge,:,1),1,gd); Yg=repeat(reshape(ge,1,:),gd,1)
ax_t,ay_t=NIE.LensDerivative(Xg,Yg;theta_E=1.2,s_scale=0.15,e1=0.1,e2=0.0)
mag_t=sqrt.(ax_t.^2 .+ ay_t.^2); m_arc=mag_t.>0.2

function report(name,mf)
    L=n_leaves(mf.tree); lvs=[lf.level for lf in mf.tree.leaves]
    in_roi=count(lf -> max(abs((lf.xmin+lf.xmax)/2),abs((lf.ymin+lf.ymax)/2))<=ROI, mf.tree.leaves)
    axq,ayq=quadtree_deflection(mf,Xg,Yg;nsub=1)
    scl=sum(ax_t.*axq .+ ay_t.*ayq)/sum(axq.^2 .+ ayq.^2)
    err=sqrt.((ax_t .- scl.*axq).^2 .+ (ay_t .- scl.*ayq).^2)./(mag_t .+ 1e-6)
    println(rpad(name,44), " 叶=",rpad(L,6), " 弧区叶=",rpad(in_roi,6),"($(round(100*in_roi/L;digits=1))%)  ",
            "αerr窗med=",rpad("$(round(100*median(vec(err));sigdigits=3))%",9),
            "弧med=",rpad("$(round(100*median(vec(err[m_arc]));sigdigits=3))%",9),
            "p90=$(round(100*quantile(vec(err),0.9);sigdigits=3))%")
    println("          level分布: ", join(["[$(j)]$(count(==(j),lvs))" for j in sort(unique(lvs))]," "))
end

println("=== 混合：ROI 内深细分，ROI 外 max_level 封顶 ===")
for (name, OUT, maxlv) in (("基准 均匀(无封顶)", 8, 8),
                           ("全局 tau=4e-3 (方案A)", 8, 8),
                           ("混合 外=4", 4, 8),
                           ("混合 外=3", 3, 8),
                           ("混合 外=2", 2, 8))
    mf = run(OUTSIDE=OUT, maxlv=maxlv)
    report(name, mf)
end
