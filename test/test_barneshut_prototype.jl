# Barnes-Hut prototype for quadtree deflection speedup.
# Builds an internal-node tree from the flat leaf list (no changes to
# the existing QuadTree), then per-pixel: descend, use multipole for
# far nodes (MAC), exact _rect_deflection for near leaves.
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMassRecon: _rect_deflection, _leaf_value, _leaf_cx, _leaf_cy
using Jens.LensMeshRefine: QuadLeaf
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
mf=MassField(-EXT,EXT,-EXT,EXT;max_level=8,eta=0.5,damp_pow=0.0,min_obs=1,p_order=1)
rs=collect(kt)
for it in 1:25
    accumulate_residuals!(mf,xs,ys,rs); update_field!(mf)
    refine_mass!(mf;tau=5e-4,part=0.4); rs .= kt .- field_value(mf,xs,ys)
end
L=Jens.LensMeshRefine.n_leaves(mf.tree)
println("leaves=$L")

# ── build internal-node tree (mutable, holds multipole mass) ──
struct QN
    xmin::Float64; xmax::Float64; ymin::Float64; ymax::Float64
    cx::Float64; cy::Float64      # centre (≈ mass centroid for multipole)
    M::Float64                    # monomer κ·area of subtree
    level::Int
    isleaf::Bool
    children::Vector{QN}
    leaf::Union{Nothing, QuadLeaf}
end

function buildtree(mf::MassField; p_order=mf.p_order)
    leaves = mf.tree.leaves
    # quadtree is spatially complete: leaves tile the root bbox without
    # overlap.  Identify a box as a leaf when its (area) matches a QuadLeaf;
    # otherwise recurse into 4 children.
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
            kv = Float64(lf.value)
            M = kv * w * h
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

# monopole+quadrupole multipole kernel (no transcendentals)
@inline function _mpole_contrib(q::QN, x::Float64, y::Float64)
    Rx = x - q.cx; Ry = y - q.cy
    R2 = Rx*Rx + Ry*Ry
    R2 < 1e-20 && return 0.0, 0.0
    invR2 = 1/R2
    ax = (q.M/pi) * Rx * invR2
    ay = (q.M/pi) * Ry * invR2
    return ax, ay
end

# Barnes-Hut traversal for one pixel
function bh_one(root::QN, x::Float64, y::Float64; theta::Float64=0.5)
    ax=0.0; ay=0.0
    stack = QN[root]
    while !isempty(stack)
        q = pop!(stack)
        dx = x - q.cx; dy = y - q.cy
        s = q.xmax - q.xmin                    # node size
        if q.isleaf
            if q.leaf === nothing
                continue
            end
            # exact contribution of this leaf (matches quadtree_deflection
            # nsub=1: CONSTANT κ = leaf-centre value × exact rect integral)
            dax, day = _rect_deflection(x, y, q.leaf.xmin, q.leaf.xmax,
                                        q.leaf.ymin, q.leaf.ymax)
            kc = Float64(q.leaf.value)
            ax += kc*dax; ay += kc*day
        elseif s/sqrt(dx*dx+dy*dy) < theta || q.level >= 10
            # far: multipole
            dax, day = _mpole_contrib(q, x, y)
            ax += dax; ay += day
        else
            append!(stack, q.children)
        end
    end
    return ax, ay
end

# build tree
root = buildtree(mf)
println("tree built, root M=", round(root.M;sigdigits=4))

# 200² evaluation grid
gd=120
ge=collect(range(-1.0,1.0;length=gd))
Xg=repeat(reshape(ge,:,1),1,gd); Yg=repeat(reshape(ge,1,:),gd,1)
xv=vec(Xg); yv=vec(Yg)

# exact reference (nsub=1)
t0=time()
ax_ex, ay_ex = quadtree_deflection(mf, xv, yv; nsub=1)
t_ex = time()-t0
println("\nexact nsub=1: $(round(t_ex;sigdigits=3)) s")

# Barnes-Hut
for theta in (0.3, 0.5, 0.8)
    t0=time()
    axs = zeros(length(xv)); ays = zeros(length(xv))
    for i in eachindex(xv)
        axs[i], ays[i] = bh_one(root, xv[i], yv[i]; theta=theta)
    end
    t_bh = time()-t0
    # error vs exact
    err = sqrt.((axs.-ax_ex).^2 .+ (ays.-ay_ex).^2) ./ (sqrt.(ax_ex.^2 .+ ay_ex.^2) .+ 1.0e-6)
    println("BH θ=$theta: $(round(t_bh;sigdigits=3)) s   speedup=$(round(t_ex/t_bh;sigdigits=3))×   " *
            "|α| rel err: med=$(round(100*median(vec(err));digits=2))%  p90=$(round(100*quantile(vec(err),0.9);digits=2))%")
end
