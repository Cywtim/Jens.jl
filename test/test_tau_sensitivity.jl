# Sensitivity of refinement budget to tau: what do we GIVE UP by
# refining less (larger tau)?  Scans tau = 5e-4 × 2^k, records
# leaves / reconstruction rms / arc-window deflection error.
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

gd=200; ge=collect(range(-1.0,1.0;length=gd))
Xg=repeat(reshape(ge,:,1),1,gd); Yg=repeat(reshape(ge,1,:),gd,1)
ax_t,ay_t = NIE.LensDerivative(Xg,Yg;theta_E=1.2,s_scale=0.15,e1=0.1,e2=0.0)
mag_t = sqrt.(ax_t.^2 .+ ay_t.^2)
m_arc = mag_t .> 0.2

println(rpad("tau",10), rpad("叶数",8), rpad("rms",10), rpad("|α|err全窗 med",18),
        rpad("弧区 med",12), rpad("p90",8), rpad("level分布",28))
for k in 0:8
    tau = 5e-4 * 2.0^k
    mf=MassField(-EXT,EXT,-EXT,EXT;max_level=8,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
    rs=collect(kt)
    for it in 1:25
        accumulate_residuals!(mf,xs,ys,rs); update_field!(mf)
        refine_mass!(mf;tau=tau,part=0.4); rs .= kt .- field_value(mf,xs,ys)
    end
    L=Jens.LensMeshRefine.n_leaves(mf.tree)
    lvs=[lf.level for lf in mf.tree.leaves]
    axq,ayq = quadtree_deflection(mf, Xg, Yg; nsub=1)
    scl = sum(ax_t.*axq .+ ay_t.*ayq)/sum(axq.^2 .+ ayq.^2)
    err = sqrt.((ax_t .- scl.*axq).^2 .+ (ay_t .- scl.*ayq).^2) ./ (mag_t .+ 1e-6)
    println(rpad("$(round(tau;sigdigits=3))",10), rpad(L,8),
            rpad("$(round(sqrt(mean(rs.^2));sigdigits=3))",10),
            rpad("$(round(100*median(vec(err));sigdigits=3))%",18),
            rpad("$(round(100*median(vec(err[m_arc]));sigdigits=3))%",12),
            rpad("$(round(100*quantile(vec(err),0.9);sigdigits=3))%",8),
            join(["[$(j)]$(count(==(j),lvs))" for j in sort(unique(lvs))]," "))
end
