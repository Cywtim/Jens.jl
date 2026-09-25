# Uniform 4-way benchmark on a max_level=7 tree (the level reached by
# adaptive refinement for this NIE field):
#   NIE render | Scheme1 render (exact deflection) | Scheme2 BH deflection | Scheme3 κ eval
# Same 200² grid, same Gaussian source, same tree.
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMassRecon: _rect_deflection
using Jens.LensSystem: ForwardModel, render
using Jens.LensGenerator: GenGrid, LensedPlane, LightPlane
using Jens.LensBase: SingleModel
using Jens.LightModel: ExtendedSource
using Jens.LightModel.GaussianLight: GaussianSphere
using Jens.LensCosmo: Cosmology
import Jens.LensModel: NIEkappa, NIE
using Jens.LensMeshRefine: QuadLeaf

const b,s,q,varphi = 1.2,0.15,0.75,pi/6
function nie_kappa(xs,ys)
    fxx,fxy,fyy = NIEkappa.LensHessian(xs,ys;b=b,s=s,q=q,varphi=varphi)
    return (fxx.+fyy)./2
end

const MAXLV = 7
EXT=3.0; nx=111
c=collect(LinRange(-EXT,EXT,nx)); xs=Float64[]; ys=Float64[]
for y in c, x in c
    push!(xs,x); push!(ys,y)
end
kt=nie_kappa(xs,ys)
mf=MassField(-EXT,EXT,-EXT,EXT;max_level=MAXLV,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
rs=collect(kt)
for it in 1:25
    accumulate_residuals!(mf,xs,ys,rs); update_field!(mf)
    refine_mass!(mf;tau=5e-4,part=0.4); rs .= kt .- field_value(mf,xs,ys)
end
L=Jens.LensMeshRefine.n_leaves(mf.tree)
lvs=[lf.level for lf in mf.tree.leaves]
println("max_level=$MAXLV 树: 叶=$L  实际level最大=$(maximum(lvs))  分布=",
        join(["[$(k)]$(count(==(k),lvs))" for k in sort(unique(lvs))]," "))
rms=sqrt(mean(rs.^2)); println("重建 rms=$(round(rms;sigdigits=3))")

# ═══════ Barnes-Hut (from benchmark) ═══════
struct QN
    xmin::Float64; xmax::Float64; ymin::Float64; ymax::Float64
    cx::Float64; cy::Float64; M::Float64; level::Int; isleaf::Bool
    children::Vector{QN}; leaf::Union{Nothing, QuadLeaf}
end
function buildtree(mf::MassField)
    leaves = mf.tree.leaves
    function match_leaf(xmin,xmax,ymin,ymax)
        ax = (xmax-xmin)*(ymax-ymin)
        for lf in leaves
            (lf.xmax-lf.xmin)*(lf.ymax-lf.ymin)==ax && lf.xmin>=xmin-1e-12 && lf.xmax<=xmax+1e-12 && lf.ymin>=ymin-1e-12 && lf.ymax<=ymax+1e-12 && return lf
        end
        return nothing
    end
    function rec(xmin,xmax,ymin,ymax,lv)
        lf=match_leaf(xmin,xmax,ymin,ymax)
        if lf!==nothing
            cx=(lf.xmin+lf.xmax)/2; cy=(lf.ymin+lf.ymax)/2; w=lf.xmax-lf.xmin; h=lf.ymax-lf.ymin
            return QN(xmin,xmax,ymin,ymax,cx,cy,Float64(lf.value)*w*h,lv,true,QN[],lf)
        end
        xmid=(xmin+xmax)/2; ymid=(ymin+ymax)/2
        kids=QN[rec(xmin,xmid,ymin,ymid,lv+1),rec(xmid,xmax,ymin,ymid,lv+1),rec(xmin,xmid,ymid,ymax,lv+1),rec(xmid,xmax,ymid,ymax,lv+1)]
        M=sum(k.M for k in kids); cx=sum(k.M*k.cx for k in kids)/max(M,1e-30); cy=sum(k.M*k.cy for k in kids)/max(M,1e-30)
        return QN(xmin,xmax,ymin,ymax,cx,cy,M,lv,false,kids,nothing)
    end
    return rec(mf.tree.xmin,mf.tree.xmax,mf.tree.ymin,mf.tree.ymax,0)
end
@inline function _mp(q::QN,x::Real,y::Real)
    Rx=x-q.cx; Ry=y-q.cy; R2=Rx*Rx+Ry*Ry
    R2<1e-20 && return 0.0,0.0
    invR2=1/R2
    return (q.M/pi)*Rx*invR2,(q.M/pi)*Ry*invR2
end
function bh(root::QN,xv,yv;theta=0.5)
    axs=zeros(length(xv)); ays=zeros(length(xv))
    for i in eachindex(xv)
        x=xv[i];y=yv[i];ax=0.0;ay=0.0
        st=[root]
        while !isempty(st)
            q=pop!(st); dx=x-q.cx; dy=y-q.cy; s=q.xmax-q.xmin
            if q.isleaf
                q.leaf===nothing && continue
                dax,day=_rect_deflection(x,y,q.leaf.xmin,q.leaf.xmax,q.leaf.ymin,q.leaf.ymax)
                ax+=Float64(q.leaf.value)*dax; ay+=Float64(q.leaf.value)*day
            elseif s/sqrt(dx*dx+dy*dy)<theta || q.level>=10
                dax,day=_mp(q,x,y); ax+=dax; ay+=day
            else
                append!(st,q.children)
            end
        end
        axs[i]=ax; ays[i]=ay
    end
    return axs,ays
end
root = buildtree(mf)

# ═══════ shared 200² setup ═══════
src = ExtendedSource(GaussianSphere; amp=1.0, sigma=0.05, xcentre=0.18, ycentre=0.10)
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
grid = GenGrid(pix_n=200, pix_size=0.018)
xg, yg = grid.xg, grid.yg
xv2=vec(xg); yv2=vec(yg)
lens_nie = SingleModel(NIE; theta_E=1.2, s_scale=0.15, e1=0.1, e2=0.0)
lens_qt  = QuadTreeLens(mf)
sys_nie = ForwardModel(lens_plane=LensedPlane(lens_nie; z_lens=0.5, cosmology=cosmo),
                       source_plane=LightPlane(src; z=1.5), grid=grid)
sys_qt  = ForwardModel(lens_plane=LensedPlane(lens_qt; z_lens=0.5, cosmology=cosmo),
                       source_plane=LightPlane(src; z=1.5), grid=grid)
render(sys_nie); render(sys_qt); bh(root,xv2,yv2); field_value(mf,xv2,yv2)

println("\n=== level 7 树 · 200² · 一次 forward evaluation ===")
t0=time(); img_nie=render(sys_nie); t_nie=time()-t0
println(rpad("NIE render (解析基准)",46), rpad("$(round(t_nie*1000;digits=1)) ms",12), "1×")

t0=time(); img_qt=render(sys_qt); t1=time()-t0
println(rpad("方案1 render (精确 deflection)",46), rpad("$(round(t1*1000;digits=1)) ms",12), "$(round(t1/t_nie;digits=0))×")

for th in (0.5, 0.8)
    t0=time(); axs,ays=bh(root,xv2,yv2;theta=th); t2=time()-t0
    # complete render estimate: deflection + source eval (source eval ~ms, negligible)
    println(rpad("方案2 deflection (BH θ=$th)",46), rpad("$(round(t2*1000;digits=1)) ms",12), "$(round(t2/t_nie;digits=0))×")
end

t0=time(); kv=field_value(mf,xv2,yv2); t3=time()-t0
kt_obs=nie_kappa(xv2,yv2)
t0=time(); chi2=sum((kv.-kt_obs).^2); t3b=time()-t0
println(rpad("方案3 κ 评估 (场值)",46), rpad("$(round(t3*1000;digits=1)) ms",12), "$(round(t3/t_nie;digits=0))×")
println(rpad("       (含χ²: $(round(t3b;sigdigits=2)) s, χ²=$(round(chi2;sigdigits=3)))",50))

# also exact timing as reference
t0=time(); quadtree_deflection(mf,xv2,yv2;nsub=1); t_ex=time()-t0
println("\n对照: 精确 deflection(200²) = $(round(t_ex*1000;digits=1)) ms  = $(round(t_ex/t_nie;digits=0))× NIE")
