# Uniform forward-evaluation benchmark:
#   NIE render            — analytic lens (baseline ref)
#   Scheme 1 render       — QuadTreeLens via ForwardModel (exact deflection)
#   Scheme 2 render       — QuadTreeLens with Barnes-Hut deflection
#   Scheme 3 evaluation   — κ-plane fitness eval (no deflection)
# All on the same 200² grid / same source / same MassField.
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

# ══════════ rebuild NIE → P1 quadtree ══════════
EXT=3.0; nx=111
c=collect(LinRange(-EXT,EXT,nx)); xs=Float64[]; ys=Float64[]
for y in c, x in c
    push!(xs,x); push!(ys,y)
end
kt=nie_kappa(xs,ys)
mf=MassField(-EXT,EXT,-EXT,EXT;max_level=8,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
rs=collect(kt)
for it in 1:25
    accumulate_residuals!(mf,xs,ys,rs); update_field!(mf)
    refine_mass!(mf;tau=5e-4,part=0.4); rs .= kt .- field_value(mf,xs,ys)
end
L=Jens.LensMeshRefine.n_leaves(mf.tree)
println("leaves=$L")

# ══════════ Barnes-Hut (prototype, inlined) ══════════
struct QN
    xmin::Float64; xmax::Float64; ymin::Float64; ymax::Float64
    cx::Float64; cy::Float64
    M::Float64
    level::Int
    isleaf::Bool
    children::Vector{QN}
    leaf::Union{Nothing, QuadLeaf}
end
function buildtree(mf::MassField)
    leaves = mf.tree.leaves
    function match_leaf(xmin,xmax,ymin,ymax)
        ax = (xmax-xmin)*(ymax-ymin)
        for lf in leaves
            if (lf.xmax-lf.xmin)*(lf.ymax-lf.ymin) == ax &&
               lf.xmin >= xmin-1e-12 && lf.xmax <= xmax+1e-12 &&
               lf.ymin >= ymin-1e-12 && lf.ymax <= ymax+1e-12
                return lf
            end
        end
        return nothing
    end
    function rec(xmin,xmax,ymin,ymax,lv)
        lf = match_leaf(xmin,xmax,ymin,ymax)
        if lf !== nothing
            cx=(lf.xmin+lf.xmax)/2; cy=(lf.ymin+lf.ymax)/2
            w=lf.xmax-lf.xmin; h=lf.ymax-lf.ymin
            M = Float64(lf.value)*w*h
            return QN(xmin,xmax,ymin,ymax,cx,cy,M,lv,true,QN[],lf)
        end
        xmid=(xmin+xmax)/2; ymid=(ymin+ymax)/2
        kids = QN[rec(xmin,xmid,ymin,ymid,lv+1),
                  rec(xmid,xmax,ymin,ymid,lv+1),
                  rec(xmin,xmid,ymid,ymax,lv+1),
                  rec(xmid,xmax,ymid,ymax,lv+1)]
        M = sum(k.M for k in kids)
        cx = sum(k.M*k.cx for k in kids)/max(M,1e-30)
        cy = sum(k.M*k.cy for k in kids)/max(M,1e-30)
        return QN(xmin,xmax,ymin,ymax,cx,cy,M,lv,false,kids,nothing)
    end
    return rec(mf.tree.xmin, mf.tree.xmax, mf.tree.ymin, mf.tree.ymax, 0)
end
@inline function _mpole(q::QN, x::Real, y::Real)
    Rx = x-q.cx; Ry = y-q.cy
    R2 = Rx*Rx+Ry*Ry
    R2 < 1e-20 && return 0.0, 0.0
    invR2 = 1/R2
    return (q.M/pi)*Rx*invR2, (q.M/pi)*Ry*invR2
end
function bh_deflection(root::QN, xv, yv; theta::Float64=0.5)
    axs = zeros(length(xv)); ays = zeros(length(xv))
    for i in eachindex(xv)
        x = xv[i]; y = yv[i]
        ax=0.0; ay=0.0
        stack = [root]
        while !isempty(stack)
            q = pop!(stack)
            dx = x-q.cx; dy = y-q.cy
            s = q.xmax-q.xmin
            if q.isleaf
                q.leaf === nothing && continue
                dax, day = _rect_deflection(x,y,q.leaf.xmin,q.leaf.xmax,
                                            q.leaf.ymin,q.leaf.ymax)
                ax += Float64(q.leaf.value)*dax; ay += Float64(q.leaf.value)*day
            elseif s/sqrt(dx*dx+dy*dy) < theta || q.level >= 10
                dax, day = _mpole(q,x,y); ax += dax; ay += day
            else
                append!(stack, q.children)
            end
        end
        axs[i]=ax; ays[i]=ay
    end
    return axs, ays
end

root = buildtree(mf)

# ══════════ shared setup ══════════
src = ExtendedSource(GaussianSphere; amp=1.0, sigma=0.05,
                     xcentre=0.18, ycentre=0.10)
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
grid = GenGrid(pix_n=200, pix_size=0.018)
xg, yg = grid.xg, grid.yg
xv2 = vec(xg); yv2 = vec(yg)

lens_nie = SingleModel(NIE; theta_E=1.2, s_scale=0.15, e1=0.1, e2=0.0)
lens_qt  = QuadTreeLens(mf)
sys_nie = ForwardModel(lens_plane=LensedPlane(lens_nie; z_lens=0.5, cosmology=cosmo),
                       source_plane=LightPlane(src; z=1.5), grid=grid)
sys_qt  = ForwardModel(lens_plane=LensedPlane(lens_qt; z_lens=0.5, cosmology=cosmo),
                       source_plane=LightPlane(src; z=1.5), grid=grid)

# warm up
render(sys_nie); render(sys_qt)
bh_deflection(root, xv2, yv2; theta=0.5)
field_value(mf, xv2, yv2)

println("\n=== 一次 forward evaluation 时间（200²=40000 像素, 2509 叶）===")
# NIE render (analytic baseline)
t0=time(); img_nie = render(sys_nie); t_nie = time()-t0
println("NIE  render            : $(round(t_nie*1000;digits=1)) ms   (解析透镜基准)")

# Scheme 1: QuadTreeLens render via ForwardModel (exact deflection)
t0=time(); img_qt = render(sys_qt); t1 = time()-t0
println("方案1 render (精确 deflection): $(round(t1*1000;digits=1)) ms  [$(round(t1/t_nie;digits=1))× NIE]")

# Scheme 2: BH deflection render (one deflection + source eval, no full render pipeline)
t0=time()
axs, ays = bh_deflection(root, xv2, yv2; theta=0.5)
t_bh = time()-t0
println("方案2 deflection (BH θ=0.5): $(round(t_bh*1000;digits=1)) ms  [$(round(t_bh/t_nie;digits=2))× NIE]")

# Scheme 3: κ-plane evaluation (no deflection at all)
t0=time(); kv = field_value(mf, xv2, yv2); t_k = time()-t0
println("方案3 κ 评估 (场值+χ²): $(round(t_k*1000;digits=1)) ms  [$(round(t_k/t_nie;digits=3))× NIE]")
kt_obs = nie_kappa(xv2, yv2)                    # "observed" κ (true NIE)
t0=time(); chi2 = sum((kv .- kt_obs).^2); t_k2 = time()-t0
println("        （含 χ² 比较: $(round(t_k2*1000;digits=3)) ms，χ²=$(round(chi2;sigdigits=3))）")

# also BH θ=0.8
t0=time(); bh_deflection(root, xv2, yv2; theta=0.8); t_bh8 = time()-t0
println("方案2 deflection (BH θ=0.8): $(round(t_bh8*1000;digits=1)) ms")

# scheme1 render breakdown: deflection alone is the dominant cost
t0=time(); quadtree_deflection(mf, xv2, yv2; nsub=1); t_exd = time()-t0
println("\n对照: 精确 deflection (nsub=1) = $(round(t_exd*1000;digits=1)) ms")
println("NIE render 之前: $(round(t_nie*1000;digits=1)) ms")
