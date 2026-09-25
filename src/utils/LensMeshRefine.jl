# ═══════════════════════════════════════════════════════════════
#  LensMeshRefine — adaptive quad-tree grid for lens-plane fields
#
#  Generic 2D quad-tree over a rectangular region, used by BOTH
#  the mass side (LensMassRecon: κ / potential-correction Dψ) and
#  the source side (LensSourceRecon: pixelised source).
#
#  Design notes (see discussion in the quadtree branch):
#    * We store ONLY the current leaf set in a flat `Vector`,
#      not a pointer-based tree.  Refinement = appending 4 child
#      leaves per split (see `refine!`).  This keeps the structure
#      GPU-friendly (Morton/Z-order sort gives coalesced access).
#    * Split decision is data-driven: a leaf splits iff
#        level < max_level  AND  n_obs ≥ min_obs  AND  resid > τ(level)
#      where `resid` is a generic criterion (image residual, mass
#      gradient, or goal-oriented gain ΔE_D) set by the caller.
#    * `refine!` is a single level-up pass; call repeatedly for a
#      full L0 → L_max coarse-to-fine growth.
# ═══════════════════════════════════════════════════════════════

module LensMeshRefine

import Jens: JFloat

export QuadLeaf, QuadTree
export split_leaf, refine!, n_leaves
export leaf_centers, leaf_areas, to_flat, bbox, fill_field


# ═══════════════════════════════════════════════════════════════
#  QuadLeaf — one flat leaf (a block carrying a scalar value)
# ═══════════════════════════════════════════════════════════════

"""
    QuadLeaf(xmin, xmax, ymin, ymax, level; value=0.0, gx=0.0, gy=0.0,
             hxx=0.0, hxy=0.0, hyy=0.0, resid=Inf, n_obs=0)

One leaf of the quad-tree: an axis-aligned box at refinement depth
`level`, carrying a scalar `value` (e.g. κ) plus a **P1/P2 Taylor
expansion** about the leaf centre (x_c, y_c):
    κ(x, y) ≈ value + gx·(x−x_c) + gy·(y−y_c)            (P1)
            + hxx·(x−x_c)² + hxy·(x−x_c)(y−y_c) + hyy·(y−y_c)²   (P2)
Also carries `resid` (subdivision criterion) and `n_obs` (number of
constraining rays/images inside this box).
"""
mutable struct QuadLeaf
    xmin::JFloat; xmax::JFloat
    ymin::JFloat; ymax::JFloat
    level::Int
    value::JFloat       # field value at leaf centre
    gx::JFloat          # ∂field/∂x (P1 slope about centre)
    gy::JFloat          # ∂field/∂y (P1 slope about centre)
    hxx::JFloat         # P2 curvature terms
    hxy::JFloat
    hyy::JFloat
    resid::JFloat       # subdivision criterion (set by caller)
    n_obs::Int          # number of constraining observations in the box
end

function QuadLeaf(xmin::Real, xmax::Real, ymin::Real, ymax::Real,
                  level::Int; value::Real=0.0, gx::Real=0.0, gy::Real=0.0,
                  hxx::Real=0.0, hxy::Real=0.0, hyy::Real=0.0,
                  resid::Real=Inf, n_obs::Int=0)
    return QuadLeaf(JFloat(xmin), JFloat(xmax), JFloat(ymin), JFloat(ymax),
                    level, JFloat(value), JFloat(gx), JFloat(gy),
                    JFloat(hxx), JFloat(hxy), JFloat(hyy),
                    JFloat(resid), Int(n_obs))
end

leaf_center_x(lf::QuadLeaf) = (lf.xmin + lf.xmax) / 2
leaf_center_y(lf::QuadLeaf) = (lf.ymin + lf.ymax) / 2

"""
    cx, cy = leaf_centers(t)

Vector of leaf-centre coordinates (one per leaf), useful for
gathering samples on the field.
"""
function leaf_centers(t)
    cx = JFloat[leaf_center_x(lf) for lf in t.leaves]
    cy = JFloat[leaf_center_y(lf) for lf in t.leaves]
    return cx, cy
end

"""
    a = leaf_areas(t)

Area of every leaf (useful for quadrature / mass from κ).
"""
function leaf_areas(t)
    return [(lf.xmax - lf.xmin) * (lf.ymax - lf.ymin) for lf in t.leaves]
end


# ═══════════════════════════════════════════════════════════════
#  QuadTree — flat leaf container + bbox + max depth
# ═══════════════════════════════════════════════════════════════

"""
    QuadTree(xmin, xmax, ymin, ymax; max_level=6)

A quad-tree represented as a flat `Vector{QuadLeaf}` (initialised
with a single root leaf).  `max_level` caps refinement depth.
"""
mutable struct QuadTree
    xmin::JFloat; xmax::JFloat
    ymin::JFloat; ymax::JFloat
    max_level::Int
    leaves::Vector{QuadLeaf}
end
function QuadTree(xmin::Real, xmax::Real, ymin::Real, ymax::Real;
                  max_level::Int=6)
    root = QuadLeaf(xmin, xmax, ymin, ymax, 0)
    return QuadTree(JFloat(xmin), JFloat(xmax), JFloat(ymin), JFloat(ymax),
                    Int(max_level), [root])
end

n_leaves(t::QuadTree) = length(t.leaves)

bbox(t::QuadTree) = (t.xmin, t.xmax, t.ymin, t.ymax)


# ═══════════════════════════════════════════════════════════════
#  Splitting / refinement
# ═══════════════════════════════════════════════════════════════

"""
    children = split_leaf(lf)

Split one leaf into its 4 children (NW, NE, SW, SE), each at
`level+1`, inheriting `value` of the parent.  Returns a Vector of
4 new leaves.
"""
function split_leaf(lf::QuadLeaf)
    xmid = (lf.xmin + lf.xmax) / 2
    ymid = (lf.ymin + lf.ymax) / 2
    lv = lf.level + 1
    return QuadLeaf[
        QuadLeaf(lf.xmin, xmid, lf.ymin, ymid, lv; value=lf.value, gx=lf.gx, gy=lf.gy,
                 hxx=lf.hxx, hxy=lf.hxy, hyy=lf.hyy, resid=Inf, n_obs=lf.n_obs),  # NW
        QuadLeaf(xmid, lf.xmax, lf.ymin, ymid, lv; value=lf.value, gx=lf.gx, gy=lf.gy,
                 hxx=lf.hxx, hxy=lf.hxy, hyy=lf.hyy, resid=Inf, n_obs=lf.n_obs),  # NE
        QuadLeaf(lf.xmin, xmid, ymid, lf.ymax, lv; value=lf.value, gx=lf.gx, gy=lf.gy,
                 hxx=lf.hxx, hxy=lf.hxy, hyy=lf.hyy, resid=Inf, n_obs=lf.n_obs),  # SW
        QuadLeaf(xmid, lf.xmax, ymid, lf.ymax, lv; value=lf.value, gx=lf.gx, gy=lf.gy,
                 hxx=lf.hxx, hxy=lf.hxy, hyy=lf.hyy, resid=Inf, n_obs=lf.n_obs),  # SE
    ]
end

"""
    refine!(tree; tau=1e-3, min_obs=0)

One full refinement pass: for every leaf, split it if
`level < max_level` AND `n_obs ≥ min_obs` AND `resid > tau`.
Returns the (possibly grown) tree.  Call repeatedly to grow
coarse → fine.  Threshold `tau` may be a scalar or a callable
`tau(level)`.
"""
function refine!(tree::QuadTree; tau=1e-3, min_obs::Int=0)
    tauf = tau isa Function ? tau : (_ -> tau)
    new_leaves = QuadLeaf[]
    sizehint!(new_leaves, length(tree.leaves) + 4 * length(tree.leaves))
    for lf in tree.leaves
        split = (lf.level < tree.max_level) &&
                (lf.n_obs >= min_obs) &&
                (lf.resid > tauf(lf.level))
        if split
            append!(new_leaves, split_leaf(lf))
        else
            push!(new_leaves, lf)
        end
    end
    tree.leaves = new_leaves
    return tree
end


# ═══════════════════════════════════════════════════════════════
#  Flat-array helpers (GPU/interpolation friendly)
# ═══════════════════════════════════════════════════════════════

"""
    leaves = to_flat(tree)

Return the flat `Vector{QuadLeaf}` (this IS the tree storage).
"""
to_flat(t::QuadTree) = t.leaves

"""
    f(x, y) interpolated / gathered from the leaf field.

`fill_field(tree; f)`: evaluate `f(leaf)` for every leaf and return
a `Vector` of values in leaf order.
"""
fill_field(t::QuadTree; f = lf -> lf.value) = [f(lf) for lf in t.leaves]

end # module LensMeshRefine
