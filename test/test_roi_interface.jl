# Validate the new refine_mass! roi/cap_outside interface:
#  1. default (no roi) → identical to old behaviour
#  2. roi=1.2, cap_outside=4 → fewer leaves, no arc-window error loss
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: n_leaves
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

function build(; kwargs...)
    mf=MassField(-EXT,EXT,-EXT,EXT;max_level=8,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
    rs=collect(kt)
    for it in 1:30
        accumulate_residuals!(mf,xs,ys,rs); update_field!(mf)
        refine_mass!(mf;tau=5e-4,part=0.4,kwargs...)
        rs .= kt .- field_value(mf,xs,ys)
    end
    return mf
end

# eval grid whole [-3,3]
gd=240; ge=collect(range(-3.0,3.0;length=gd))
Xg=repeat(reshape(ge,:,1),1,gd); Yg=repeat(reshape(ge,1,:),gd,1)
ax_t,ay_t=NIE.LensDerivative(Xg,Yg;theta_E=1.2,s_scale=0.15,e1=0.1,e2=0.0)
mag=sqrt.(ax_t.^2 .+ ay_t.^2); R=sqrt.(Xg.^2 .+ Yg.^2)
m_arc=R .<= 1.2; m_outer = R .> 1.2

function report(name,mf)
    L=n_leaves(mf.tree)
    lvs=[lf.level for lf in mf.tree.leaves]
    axq,ayq=quadtree_deflection(mf,Xg,Yg;nsub=1)
    scl=sum(ax_t.*axq .+ ay_t.*ayq)/sum(axq.^2 .+ ayq.^2)
    e=sqrt.((ax_t .- scl.*axq).^2 .+ (ay_t .- scl.*ayq).^2)./(mag .+ 1e-6)
    println(rpad(name,34), " 叶=",rpad(L,6),
            " 全窗med=",rpad("$(round(100*median(vec(e));sigdigits=4))%",11),
            "弧区=",rpad("$(round(100*median(vec(e[m_arc]));sigdigits=4))%",11),
            "外圈=",rpad("$(round(100*median(vec(e[m_outer]));sigdigits=4))%",11),
            "最大level=", maximum(lvs))
end

println("=== 1) 默认行为不变性 ===")
mfA = build()
mfB = build(roi=nothing)
same = all(lfA.xmin==lfB.xmin && lfA.xmax==lfB.xmax && lfA.ymin==lfB.ymin && lfA.ymax==lfB.ymax
           for (lfA,lfB) in zip(mfA.tree.leaves, mfB.tree.leaves))
println("  build()(默认) vs build(roi=nothing):  叶数 ", n_leaves(mfA.tree), " == ", n_leaves(mfB.tree),
        "  树几何相同: ", same)
report("  基准(默认)", mfA)

println("\n=== 2) roi=1.2, cap_outside=4 ===")
mfC = build(roi=1.2, cap_outside=4)
report("  roi=1.2 cap=4", mfC)
println("  叶数比: $(round(100*n_leaves(mfC.tree)/n_leaves(mfA.tree);digits=1))% 保留")

println("\n=== 3) cap_outside 变体 ===")
for cap in (3, 2)
    mfD = build(roi=1.2, cap_outside=cap)
    report("  roi=1.2 cap=$cap", mfD)
end
