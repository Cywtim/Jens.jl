# ═══════════════════════════════════════════════════════════════
#  LensMassRecon — quad-tree mass-field reconstruction
#
#  Builds a mass field κ(θ) (or potential correction Dψ) on an
#  adaptive quad-tree, driven by residual samples rather than a
#  fixed analytic profile.
#
#  ── Part of the quad-tree lens stack ─────────────────────────────
#  This is the LENS ENGINE of the stack.  Bottom mesh lives in
#  `LensMeshRefine` (QuadLeaf/QuadTree, pure adaptive grid); the
#  image-plane fitness layer lives in `QuadTreeFit`
#  (`quadtree_image_logp`).  Typical flow: LensMeshRefine provides
#  the grid → this module reconstructs κ / α / ψ / H →
#  QuadTreeFit plugs it into a log-posterior.
#  ──────────────────────────────────────────────────────────────
#
#  Workflow (per refinement level):
#    1. ACCUMULATE  — scatter residual samples (xᵢ, yᵢ, wᵢ, rᵢ)
#       into the leaf that contains them (box search on the flat
#       leaf array).   Each leaf accumulates Σw, Σw·r.
#    2. UPDATE      — leaf residual → leaf κ correction via a
#       damped update:  κ_leaf ← κ_leaf + η · (Σwr/Σw)   (tiny-η
#       stabilisation; caller drives more sophisticated estimators).
#    3. REGULARIZE  — level-dependent damping (deeper levels are
#       damped harder: deeper cells have less signal per DOF).
#    4. REFINE      — (goal-oriented / residual) subdivision via
#       `LensMeshRefine.refine!`; supports `roi`/`cap_outside` for
#       confining detail to a region of interest.
#
#  Physical frame: this module does NOT call a specific analytic
#  lens model — it reconstructs a *correction/bias field* the caller
#  combines with a base model (e.g. EPL).  This keeps it pure and
#  testable, mirroring the `RegularDpsiMesh` idea from PyAutoLens
#  (a coarse regular grid) but with quad-tree adaptivity.
# ═══════════════════════════════════════════════════════════════

module LensMassRecon

import Jens: JFloat
using LinearAlgebra: det, I
import ..LensBase: AbstractLens, lens_derivative, lens_hessian,
    lens_potential, lens_check
using Jens.LensMeshRefine:
    QuadTree, QuadLeaf, split_leaf, refine!, n_leaves,
    leaf_centers, leaf_areas, to_flat

export MassField
export accumulate_residuals!, update_field!, refine_mass!
export mass_kappa, mass_linear, p1_value, field_value
export mass_gradient, mass_total
export mean_residual, max_abs_residual, converged_global
export quadtree_deflection, quadtree_deflection_multipole
export quadtree_potential, quadtree_hessian
export quadtree_bh_tree, quadtree_deflection_bh
export QuadTreeLens


# ═══════════════════════════════════════════════════════════════
#  Polynomial-order selectable leaf model (P0 / P1 / P2)
#
#  Each leaf carries a Taylor expansion about its centre (x_c, y_c):
#     P0:  κ ≈ value
#     P1:  κ ≈ value + gx·dx + gy·dy
#     P2:  κ ≈ value + gx·dx + gy·dy + hxx·dx² + hxy·dx·dy + hyy·dy²
#  `p_order` (0/1/2) selects the model; accumulation and update use
#  only the moments the chosen order needs (moment subset indexing
#  below).  Deeper leaves are damped harder (λ(ℓ) = (1/2)^(ℓ·damp_pow)).
# ═══════════════════════════════════════════════════════════════

# moment row indices in the flat per-leaf moment block.
# Layout is CHOSEN so that each order uses a contiguous prefix:
#    P0 rows 1:3    = w, r, rr
#    P1 rows 1:10   = P0 + x, y, xx, xy, yy, rx, ry
#    P2 rows 1:22   = P1 + third/fourth moments + rxx, rxy, ryy
const _M_W   = 1                                    # Σw
const _M_R   = 2                                    # Σw·r
const _M_RR  = 3                                    # Σw·r²
const _M_X   = 4; const _M_Y   = 5                  # Σw·dx,  Σw·dy
const _M_XX  = 6; const _M_XY = 7; const _M_YY = 8  # Σw·dx², Σw·dxdy, Σw·dy²
const _M_RX  = 9; const _M_RY = 10                  # Σw·r·dx, Σw·r·dy
const _M_XXX = 11; const _M_XXY = 12; const _M_XYY = 13; const _M_YYY = 14   # 3rd
const _M_XXXX=15; const _M_XXXY=16; const _M_XXYY=17; const _M_XYYY=18; const _M_YYYY=19  # 4th
const _M_RXX = 20; const _M_RXY = 21; const _M_RYY = 22  # Σw·r·dx², r·dxdy, r·dy²

# number of moments needed per order (contiguous rows 1:n)
_nmom(po::Int) = po == 0 ? 3 : po == 1 ? 10 : 22

# ═══════════════════════════════════════════════════════════════
#  MassField — quad-tree + accumulated statistics
# ═══════════════════════════════════════════════════════════════

"""
    MassField(xmin, xmax, ymin, ymax; max_level=6, eta=0.5,
              damp_pow=1.0, min_obs=3, p_order=1)

Quad-tree mass field being reconstructed, with a selectable local
polynomial order `p_order ∈ {0,1,2}`:
- `0` = constant leaves (`value` only)
- `1` = P1 linear leaves (`value, gx, gy`)
- `2` = P2 quadratic leaves (`value, gx, gy, hxx, hxy, hyy`)

During `accumulate_residuals!` we gather the local weighted LSQ
moments (`_nmom(p_order)` per leaf); `update_field!` solves the
m×m normal equations (m = n_coefficients of `p_order`), with graceful
degradation to a lower order when a leaf is too small or degenerate
(P2→P1→constant).

# Fields
- `tree::QuadTree`      — adaptive grid (leaf value/gx/gy/hxx/hxy/hyy)
- `eta::Float64`        — damped-update step size
- `damp_pow::Float64`   — level-dependent damping: λ(ℓ) = (1/2)^(ℓ·damp_pow)
- `min_obs::Int`        — refinement needs ≥ this many samples in a leaf
- `p_order::Int`        — 0/1/2 leaf-model order
- `mom::Vector{JFloat}` — flat per-leaf moments `[n_mom × n_leaves]`
"""
# mutable: `rev` is a live counter bumped by update_field!/refine_mass!,
# used to auto-rebuild cached acceleration structures (Barnes-Hut tree).
mutable struct MassField
    tree::QuadTree
    eta::JFloat
    damp_pow::JFloat
    min_obs::Int
    p_order::Int
    mom::Vector{JFloat}   # length n_mom × n_leaves (row-major blocks)
    rev::UInt             # ++ on every field-mutating pass (update/refine)
end

function MassField(xmin::Real, xmax::Real, ymin::Real, ymax::Real;
                   max_level::Int=6, eta::Real=0.5, damp_pow::Real=1.0,
                   min_obs::Int=3, p_order::Int=1)
    tree = QuadTree(xmin, xmax, ymin, ymax; max_level=max_level)
    n = length(tree.leaves) * _nmom(p_order)
    return MassField(tree, JFloat(eta), JFloat(damp_pow), Int(min_obs),
                     Int(p_order), zeros(JFloat, n), UInt(0))
end

_leaves(mf::MassField) = mf.tree.leaves

_leaf_cx(lf::QuadLeaf) = (lf.xmin + lf.xmax) / 2
_leaf_cy(lf::QuadLeaf) = (lf.ymin + lf.ymax) / 2

# per-leaf moment block accessors
_nmom_rows(po::Int) = _nmom(po)
@inline _moff(mf::MassField, k::Int) = (k - 1) * _nmom_rows(mf.p_order)
@inline _mget(mf::MassField, k::Int, row::Int) =
    mf.mom[_moff(mf, k) + row]
@inline _madd!(mf::MassField, k::Int, row::Int, v::Real) =
    (mf.mom[_moff(mf, k) + row] += v; nothing)


# ═══════════════════════════════════════════════════════════════
#  Step 1 — accumulate residual samples into containing leaves
# ═══════════════════════════════════════════════════════════════

"""
    accumulate_residuals!(mf, xs, ys, rs; ws=nothing)

Scatter each residual sample `(xs[i], ys[i]) -> rs[i]` into the leaf
containing it (box containment over the flat leaf array).  Optional
weights `ws` default to 1.  Fills the per-leaf LSQ moments needed by
`mf.p_order` (3 for P0, 10 for P1, 22 for P2) and sets each leaf's
`n_obs`.  dx, dy are measured from the leaf centre.  Ready for
`update_field!`.
"""
function accumulate_residuals!(mf::MassField, xs, ys, rs; ws=nothing)
    leaves = _leaves(mf)
    nm = _nmom_rows(mf.p_order)
    resize!(mf.mom, nm * length(leaves)); fill!(mf.mom, 0)
    for lf in leaves
        lf.n_obs = 0
        lf.resid = 0.0
    end
    po = mf.p_order

    for i in eachindex(xs)
        x = xs[i]; y = ys[i]; r = rs[i]
        w = ws === nothing ? one(JFloat) : JFloat(ws[i])
        for (k, lf) in enumerate(leaves)
            if lf.xmin <= x <= lf.xmax && lf.ymin <= y <= lf.ymax
                dx = JFloat(x - _leaf_cx(lf))
                dy = JFloat(y - _leaf_cy(lf))
                mf.mom[_moff(mf, k) + _M_W] += w
                mf.mom[_moff(mf, k) + _M_RR] += w * r * r
                mf.mom[_moff(mf, k) + _M_R] += w * r
                if po >= 1
                    mf.mom[_moff(mf, k) + _M_X]  += w * dx
                    mf.mom[_moff(mf, k) + _M_Y]  += w * dy
                    mf.mom[_moff(mf, k) + _M_XX] += w * dx * dx
                    mf.mom[_moff(mf, k) + _M_XY] += w * dx * dy
                    mf.mom[_moff(mf, k) + _M_YY] += w * dy * dy
                    mf.mom[_moff(mf, k) + _M_RX] += w * r * dx
                    mf.mom[_moff(mf, k) + _M_RY] += w * r * dy
                end
                if po >= 2
                    dx2 = dx * dx; dy2 = dy * dy
                    mf.mom[_moff(mf, k) + _M_XXX]  += w * dx2 * dx
                    mf.mom[_moff(mf, k) + _M_XXY]  += w * dx2 * dy
                    mf.mom[_moff(mf, k) + _M_XYY]  += w * dx * dy2
                    mf.mom[_moff(mf, k) + _M_YYY]  += w * dy2 * dy
                    mf.mom[_moff(mf, k) + _M_XXXX] += w * dx2 * dx2
                    mf.mom[_moff(mf, k) + _M_XXXY] += w * dx2 * dx * dy
                    mf.mom[_moff(mf, k) + _M_XXYY] += w * dx2 * dy2
                    mf.mom[_moff(mf, k) + _M_XYYY] += w * dx * dy2 * dy
                    mf.mom[_moff(mf, k) + _M_YYYY] += w * dy2 * dy2
                    mf.mom[_moff(mf, k) + _M_RXX]  += w * r * dx2
                    mf.mom[_moff(mf, k) + _M_RXY]  += w * r * dx * dy
                    mf.mom[_moff(mf, k) + _M_RYY]  += w * r * dy2
                end
                lf.n_obs += 1
                break
            end
        end
    end
    return mf
end


# ═══════════════════════════════════════════════════════════════
#  Step 2 — update leaf κ values (damped, level-regularized)
# ═══════════════════════════════════════════════════════════════

"""
    update_field!(mf)

Convert accumulated LSQ moments into leaf updates using the leaf
model selected by `mf.p_order`:
- P0: constant  `κ ← κ + ηλ·(Σwr/Σw)`
- P1: linear    `Δκ = Δv + gx·dx + gy·dy`  (solve 3×3 normal equations)
- P2: quadratic `Δκ = Δv + gx·dx + gy·dy + hxx·dx² + hxy·dxdy + hyy·dy²`
                (solve 6×6 normal equations)

Deeper leaves are damped more (λ(ℓ) = (1/2)^(ℓ·damp_pow)) to prevent
over-fitting to undersampled high-frequency cells.  Leaves with too
few samples or degenerate geometry **degrade gracefully to the next
lower order** (P2→P1→constant).  Leaf `resid` = **post-fit RMS**: the
residual left after the fitted polynomial is removed — the
subdivision criterion (a smooth monotone field is captured by slope/
curvature, so it stops spuriously splitting; genuine curvature still
triggers refinement).  Returns `mf`.
"""
function update_field!(mf::MassField)
    leaves = _leaves(mf)
    T = JFloat
    po = mf.p_order
    for (k, lf) in enumerate(leaves)
        sw = _mget(mf, k, _M_W)
        if sw <= 0
            lf.resid = 0.0
            continue
        end
        lam = T(0.5)^(lf.level * mf.damp_pow)
        step = mf.eta * lam
        srr = _mget(mf, k, _M_RR)
        br  = _mget(mf, k, _M_R)

        # ── P2: 6×6 fit (requires 6+ samples, well-posed) ──
        if po >= 2 && lf.n_obs >= 6
            c = _solve_p2(mf, k)
            if c !== nothing
                lf.value += step * c[1]
                lf.gx    += step * c[2]
                lf.gy    += step * c[3]
                lf.hxx   += step * c[4]
                lf.hxy   += step * c[5]
                lf.hyy   += step * c[6]
                rms2 = (srr - dot6(c, mf, k)) / sw
                lf.resid = sqrt(max(rms2, 0))
                continue
            end
        end

        # ── P1: 3×3 fit (3+ samples, well-posed) ──
        if po >= 1 && lf.n_obs >= 3
            c = _solve_p1(mf, k)
            if c !== nothing
                lf.value += step * c[1]
                lf.gx    += step * c[2]
                lf.gy    += step * c[3]
                rms2 = (srr - (c[1]*br + c[2]*_mget(mf,k,_M_RX) +
                               c[3]*_mget(mf,k,_M_RY))) / sw
                lf.resid = sqrt(max(rms2, 0))
                continue
            end
        end

        # ── constant fit (P0 or degradation) ──
        meanr = br / sw
        lf.value += step * meanr
        rms2 = (srr - br * meanr) / sw
        lf.resid = sqrt(max(rms2, 0))
    end
    mf.rev += 1          # field values changed → cached BH tree stale
    return mf
end

dot6(c, mf, k) = (c[1]*_mget(mf,k,_M_R)  + c[2]*_mget(mf,k,_M_RX) +
                   c[3]*_mget(mf,k,_M_RY) + c[4]*_mget(mf,k,_M_RXX) +
                   c[5]*_mget(mf,k,_M_RXY) + c[6]*_mget(mf,k,_M_RYY))

# P1 normal equations: (Δv, gx, gy).  Returns nothing on degeneracy.
function _solve_p1(mf::MassField, k::Int)
    T = JFloat
    sw = _mget(mf,k,_M_W); sx = _mget(mf,k,_M_X); sy = _mget(mf,k,_M_Y)
    sxx = _mget(mf,k,_M_XX); sxy = _mget(mf,k,_M_XY); syy = _mget(mf,k,_M_YY)
    br = _mget(mf,k,_M_R); brx = _mget(mf,k,_M_RX); bry = _mget(mf,k,_M_RY)
    A = T[sw sx sy; sx sxx sxy; sy sxy syy]
    detA = A[1,1]*(A[2,2]*A[3,3] - A[2,3]*A[3,2]) -
           A[1,2]*(A[2,1]*A[3,3] - A[2,3]*A[3,1]) +
           A[1,3]*(A[2,1]*A[3,2] - A[2,2]*A[3,1])
    abs(detA) < 1e-30 * (sw^3) && return nothing
    return A \ T[br; brx; bry]
end

# P2 normal equations over basis (1, dx, dy, dx², dxdy, dy²).
# The system is solved in a LEAF-SCALED ("whitened") basis so that all
# moments are O(1) and the degeneracy test is independent of leaf size:
# let u = dx/hx, v = dy/hy with hx,hy the leaf half-widths; then the
# scaled Gram H_ij = E[u^a v^b] is near the identity scale, and raw
# coefficients are recovered by undoing the diagonal scaling.  Returns
# nothing on degeneracy (caller falls back to P1 / constant).
function _solve_p2(mf::MassField, k::Int)
    T = JFloat
    sw = _mget(mf,k,_M_W)
    hx = 0.5 * (_leaves(mf)[k].xmax - _leaves(mf)[k].xmin)
    hy = 0.5 * (_leaves(mf)[k].ymax - _leaves(mf)[k].ymin)
    (hx > 0 && hy > 0) || return nothing
    inv_sw = 1 / sw
    # scaled moments: E[u^a v^b] = (Σw u^a v^b)/sw  with u=dx/hx, v=dy/hy
    # raw moments: s_ab = Σw dx^a dy^b  ⇒ scaled = s_ab / (sw · hx^a · hy^b)
    m(r) = _mget(mf, k, r)
    sx = m(_M_X)/(sw*hx);     sy = m(_M_Y)/(sw*hy)
    sxx = m(_M_XX)/(sw*hx*hx); sxy = m(_M_XY)/(sw*hx*hy); syy = m(_M_YY)/(sw*hy*hy)
    sxxx = m(_M_XXX)/(sw*hx^3); sxxy = m(_M_XXY)/(sw*hx*hx*hy)
    sxyy = m(_M_XYY)/(sw*hx*hy*hy); syyy = m(_M_YYY)/(sw*hy^3)
    sxxxx = m(_M_XXXX)/(sw*hx^4); sxxxy = m(_M_XXXY)/(sw*hx^3*hy)
    sxxyy = m(_M_XXYY)/(sw*hx*hx*hy*hy); sxyyy = m(_M_XYYY)/(sw*hx*hy^3)
    syyyy = m(_M_YYYY)/(sw*hy^4)
    br  = m(_M_R)*inv_sw
    brx = m(_M_RX)/(sw*hx);    bry = m(_M_RY)/(sw*hy)
    brxx = m(_M_RXX)/(sw*hx*hx); brxy = m(_M_RXY)/(sw*hx*hy); bryy = m(_M_RYY)/(sw*hy*hy)
    # scaled Gram (basis 1,u,v,u²,uv,v²) — coefficient order: c = (Δv, gu, gv, guu, guv, gvv)
    G = T[1      sx    sy    sxx   sxy   syy;
          sx    sxx   sxy   sxxx  sxxy  sxyy;
          sy    sxy   syy   sxxy  sxyy  syyy;
          sxx   sxxx  sxxy  sxxxx sxxxy sxxyy;
          sxy   sxxy  sxyy  sxxxy sxxyy sxyyy;
          syy   sxyy  syyy  sxxyy sxyyy syyyy]
    h = T[br; brx; bry; brxx; brxy; bryy]
    # degeneracy: whitened Gram has unit-ish scale, so an absolute det
    # threshold is now leaf-size independent
    abs(det(G)) < 1e-10 && return nothing
    c = (G + T(1e-9) * I) \ h      # light ridge stabilisation
    any(!isfinite, c) && return nothing
    # undo scaling: raw coef = scaled coef / (hx^a hy^b)
    return T[c[1],
             c[2]/hx, c[3]/hy,
             c[4]/hx^2, c[5]/(hx*hy), c[6]/hy^2]
end


# ═══════════════════════════════════════════════════════════════
#  Step 3 — subdivide (residual / goal-driven pass)
# ═══════════════════════════════════════════════════════════════

"""
    refine_mass!(mf; tau=1e-3, part=0.5, tauf=nothing,
                 roi=nothing, cap_outside=typemax(Int))

One refinement pass over the mass quad-tree using a Dörfler-style
mark-and-split strategy (standard adaptive-FEM MARK step):

  MARK  — consider leaves with `level<max_level`, `n_obs≥min_obs`
          and `resid>tau` (or `tauf(level)`).
  ORDER — sort those by residual descending.
  CUT   — take the smallest prefix whose cumulative residual reaches
          `part` × total residual of the candidates.
  SPLIT — split exactly that prefix.

This spends refinement budget where the residual (structure) is
largest, instead of blindly splitting every high-residual leaf
(which fills the tree to `max_level`).  Accumulators are resized to
the new leaf set.

# Region-of-interest (optional)
- `roi::Real` or `nothing`: if a positive number, the ROI is the
  square `[-roi, roi]²`; leaves OUTSIDE it are depth-capped at
  `cap_outside` (default: uncapped = original behaviour).  Leaves
  inside the ROI refine to `mf.tree.max_level` as usual.  This
  spends detail where it matters (e.g. near the critical curve /
  arc window) and stops wasting budget on far-field detail, with
  NO change when `roi=nothing`.
"""
function refine_mass!(mf::MassField; tau::Real=1e-3, part::Real=0.5,
                      tauf=nothing, roi=nothing,
                      cap_outside::Int=typemax(Int))
    leaves = _leaves(mf)
    f = tauf === nothing ? (l -> tau) : tauf
    has_roi = roi !== nothing && roi > 0

    # MARK: candidates
    candidates = Int[]
    for (k, lf) in enumerate(leaves)
        ok = lf.level < mf.tree.max_level &&
             lf.n_obs >= mf.min_obs &&
             lf.resid > f(lf.level)
        if ok && has_roi
            cx = (lf.xmin + lf.xmax) / 2
            cy = (lf.ymin + lf.ymax) / 2
            in_roi = abs(cx) <= roi && abs(cy) <= roi
            ok = in_roi || lf.level < cap_outside
        end
        ok && push!(candidates, k)
    end

    # ORDER by residual desc
    sort!(candidates; by=k -> _leaves(mf)[k].resid, rev=true)

    # nothing to refine → tree converged; bail out before empty-sum crash
    isempty(candidates) && return mf

    # CUT: smallest prefix reaching part × total
    total = sum(lf.resid for k in candidates for lf in (leaves[k],))
    ns = 0
    acc = 0.0
    if total > 0
        for k in candidates
            acc += leaves[k].resid
            ns += 1
            acc >= part * total && break
        end
    end

    # SPLIT chosen leaves
    to_split = Set(candidates[1:ns])
    new_leaves = QuadLeaf[]
    sizehint!(new_leaves, length(leaves) + 4 * ns)
    for (k, lf) in enumerate(leaves)
        if k in to_split
            append!(new_leaves, split_leaf(lf))
        else
            push!(new_leaves, lf)
        end
    end
    mf.tree.leaves = new_leaves

    n = n_leaves(mf.tree)
    resize!(mf.mom, _nmom_rows(mf.p_order) * n); fill!(mf.mom, 0)
    mf.rev += 1          # leaf topology changed → cached BH tree stale
    return mf
end


# ═══════════════════════════════════════════════════════════════
#  Diagnostics / output
# ═══════════════════════════════════════════════════════════════

"""
    kv = mass_kappa(mf)

Per-leaf κ values (in leaf order).
"""
mass_kappa(mf::MassField) = [lf.value for lf in _leaves(mf)]

"""
    g = mass_gradient(mf)

Simple leaf-to-leaf gradient proxy: max absolute κ difference
between a leaf and its topological neighbours (by centre distance,
using the largest neighbouring cells).  A crude but informative
"smoothness / subdivision-worthy" score; 0 when no neighbour within
2× the leaf width.
"""
function mass_gradient(mf::MassField)
    leaves = _leaves(mf)
    cx, cy = leaf_centers(mf.tree)
    g = zeros(JFloat, length(leaves))
    for k in eachindex(leaves)
        lf = leaves[k]
        w = lf.xmax - lf.xmin
        best = 0.0
        for j in eachindex(leaves)
            j == k && continue
            dx = cx[j] - cx[k]; dy = cy[j] - cy[k]
            d = sqrt(dx * dx + dy * dy)
            if d <= 2 * w     # neighbouring (within 2 cell-widths)
                gk = abs(leaves[j].value - lf.value)
                best = max(best, gk)
            end
        end
        g[k] = best
    end
    return g
end

"""
    M = mass_total(mf)

Total projected "mass" Σ κ·area over all leaves.
"""
function mass_total(mf::MassField)
    areas = leaf_areas(mf.tree)
    kv = mass_kappa(mf)
    return sum(a * k for (a, k) in zip(areas, kv))
end

"""
    mean = mean_residual(mf)
    mx   = max_abs_residual(mf)

Summary statistics of the current accumulated leaf residual field.
"""
function mean_residual(mf::MassField)
    isempty(mf.mom) && return JFloat(0)
    nm = _nmom_rows(mf.p_order)
    n = length(mf.mom) ÷ nm
    sw = 0.0; br = 0.0
    @inbounds for k in 1:n
        sw += mf.mom[(k-1)*nm + _M_W]
        br += mf.mom[(k-1)*nm + _M_R]
    end
    return br / max(sw, eps(JFloat))
end
function max_abs_residual(mf::MassField)
    isempty(mf.mom) && return JFloat(0)
    nm = _nmom_rows(mf.p_order)
    n = length(mf.mom) ÷ nm
    mx = 0.0
    @inbounds for k in 1:n
        sw = mf.mom[(k-1)*nm + _M_W]
        br = mf.mom[(k-1)*nm + _M_R]
        mx = max(mx, abs(br) / max(sw, eps(JFloat)))
    end
    return JFloat(mx)
end

"""
    ok = converged_global(mf; tol=1e-6)

Global convergence flag: true when the largest remaining leaf
mean-residual is below `tol` (i.e. the mass field has "absorbed"
the supplied residuals).
"""
converged_global(mf::MassField; tol::Real=1e-6) =
    max_abs_residual(mf) < tol


# ═══════════════════════════════════════════════════════════════
#  Field evaluation (P0 / P1 / P2 by mf.p_order)
# ═══════════════════════════════════════════════════════════════

"""
    v, gx, gy [, hxx, hxy, hyy] = mass_linear(mf)

Per-leaf Taylor coefficients: `value` (κ at leaf centre), `gx`, `gy`
(P1 slopes), and — when present in `mf` (P2) — `hxx, hxy, hyy` (P2
curvature).  Each a Vector in leaf order.
"""
function mass_linear(mf::MassField)
    leaves = _leaves(mf)
    if mf.p_order >= 2
        return ([lf.value for lf in leaves], [lf.gx for lf in leaves],
                [lf.gy for lf in leaves], [lf.hxx for lf in leaves],
                [lf.hxy for lf in leaves], [lf.hyy for lf in leaves])
    end
    return ([lf.value for lf in leaves], [lf.gx for lf in leaves],
            [lf.gy for lf in leaves])
end

"Evaluate the leaf's polynomial model at (dx, dy) about its centre."
@inline function _leaf_value(lf::QuadLeaf, dx::Real, dy::Real, po::Int)
    v = lf.value + lf.gx * dx + lf.gy * dy
    if po >= 2
        v += lf.hxx * dx * dx + lf.hxy * dx * dy + lf.hyy * dy * dy
    end
    return v
end

"""
    k = field_value(mf, x, y)

Evaluate the reconstructed field at point(s) `(x, y)` using the leaf
polynomial model selected by `mf.p_order`:
    P0: κ ≈ value
    P1: κ ≈ value + gx·(x−x_c) + gy·(y−y_c)
    P2: κ ≈ 〃 + hxx·(x−x_c)² + hxy·(x−x_c)(y−y_c) + hyy·(y−y_c)²
at the containing leaf.  (The subdivision driver must use THIS — the
full model — to compute residuals; leaf-centre `value` alone is NOT
the model once P1/P2 slopes are active.)
"""
function field_value(mf::MassField, x, y)
    leaves = _leaves(mf)
    po = mf.p_order
    out = zeros(JFloat, size(x))
    for i in eachindex(x)
        for lf in leaves
            if lf.xmin <= x[i] <= lf.xmax && lf.ymin <= y[i] <= lf.ymax
                out[i] = _leaf_value(lf, x[i] - _leaf_cx(lf), y[i] - _leaf_cy(lf), po)
                break
            end
        end
    end
    return out
end

"""
    k = p1_value(mf, x, y)

Evaluate the reconstructed field at point(s) `(x, y)` using the **P1**
(linear) part of the leaf model:
    κ ≈ value + gx·(x−x_c) + gy·(y−y_c)   at the containing leaf.
For `p_order ∈ {0,1}` this equals `field_value`; for P2 it deliberately
omits the quadratic terms (kept for backward compatibility with the pre-
P2 reconstruction code paths).
"""
function p1_value(mf::MassField, x, y)
    leaves = _leaves(mf)
    out = zeros(JFloat, size(x))
    for i in eachindex(x)
        for lf in leaves
            if lf.xmin <= x[i] <= lf.xmax && lf.ymin <= y[i] <= lf.ymax
                out[i] = lf.value + lf.gx * (x[i] - _leaf_cx(lf)) +
                         lf.gy * (y[i] - _leaf_cy(lf))
                break
            end
        end
    end
    return out
end


# ═══════════════════════════════════════════════════════════════
#  Deflection — connect the quad-tree mass field to image-plane
#  rendering:  α(θ) from κ(θ)  (Poisson problem ∇²ψ = 2κ, α = ∇ψ)
#
#  Standard lensing relation:
#      α(θ) = (1/π) ∫ κ(θ') (θ − θ')/|θ − θ'|² d²θ'
#
#  Each leaf is a uniform-κ rectangle.  For a uniform rectangle the
#  kernel integral has an EXACT closed form (antiderivative
#      G(u,v) = v·ln(u²+v²) − 2v + 2u·atan2(v, u),
#  with ∂²G/∂u∂v = 2u/(u²+v²)), so the deflection is a sum over the
#  four box corners with no quadrature error and no boundary
#  singularity.  Summing over leaves gives α at any image point.
#
#  This is the bridge that lets the reconstructed mass field be
#  ray-traced and compared against observed images (image-plane
#  residual maps) rather than only against κ itself.
# ═══════════════════════════════════════════════════════════════

# antiderivative G(u,v): ∂²G/∂u∂v = 2u/(u²+v²)
#
# IMPORTANT branch convention: the potential antiderivative must use
# the PRINCIPAL-ARCTAN branch (`atan(v/u)` ∈ (−π/2, π/2)).  Using
# `atan2(v, u)` (∈ (−π, π]) shifts arms with u<0 by ±π, and the
# four-corner sum does NOT cancel that constant — it appears as a
# spurious constant offset when the observation point crosses the
# rectangle's vertical edges.  We therefore convert to the principal
# branch explicitly.
@inline function _G(u::Real, v::Real)
    r2 = u * u + v * v
    r2 < 1e-20 && return 0.0          # singular corner: regularized
    th = atan(v, u)                    # atan2 branch, ∈ (−π, π]
    if u < 0                            # restore principal atan(v/u) branch
        th = th > 0 ? th - pi : th + pi
    end
    return v * log(r2) - 2v + 2 * u * th
end

# Exact deflection of a uniform-κ rectangle [xlo,xhi]×[ylo,yhi] at (x,y),
# INDEPENDENT of κ (caller multiplies); returns (ax, ay) / (2π) factor built in.
function _rect_deflection(x::Real, y::Real, xlo, xhi, ylo, yhi)
    # shifted corner coordinates u = x − x', v = y − y'
    u1 = x - xhi; u2 = x - xlo
    v1 = y - yhi; v2 = y - ylo
    # α_x = (G(u2,v2) − G(u1,v2) − G(u2,v1) + G(u1,v1)) / (2π)
    ax = (_G(u2, v2) - _G(u1, v2) - _G(u2, v1) + _G(u1, v1)) / (2 * pi)
    # α_y = (G(v2,u2) − G(v1,u2) − G(v2,u1) + G(v1,u1)) / (2π)   [G symmetric swap]
    ay = (_G(v2, u2) - _G(v1, u2) - _G(v2, u1) + _G(v1, u1)) / (2 * pi)
    return ax, ay
end

"""
    ax, ay = quadtree_deflection(mf, x, y; nsub=2)

Ray-plane deflection α(θ) from the quad-tree mass field `mf` at point(s)
`(x, y)`.  Each leaf's **P1 (linear)** field is approximated by
`nsub × nsub` uniform sub-rectangles whose constant κ equals the
linear value at the sub-rectangle centre; each sub-rectangle uses the
EXACT uniform-rectangle antiderivative:
    α(θ) = (1/π) Σ_leaf Σ_sub κ_sub ∫∫_sub (θ−θ')/|θ−θ'|² d²θ'.
With `nsub=2` the linear field is reproduced to O(h²) (exact for a
true linear field in the far field; empirically converged as nsub↑).
No quadrature error on the kernel itself.  Returns two arrays
broadcastable to the shape of `x`, `y`.
(This is the bridge to image-plane rendering.)
"""
function quadtree_deflection(mf::MassField, x, y; nsub::Int=2)
    leaves = _leaves(mf)
    ax = zeros(JFloat, size(x)); ay = zeros(JFloat, size(y))
    for lf in leaves
        # sub-divide leaf into nsub×nsub uniform cells
        hx = (lf.xmax - lf.xmin) / nsub
        hy = (lf.ymax - lf.ymin) / nsub
        for js in 1:nsub, is in 1:nsub
            sx0 = lf.xmin + (is - 1) * hx
            sy0 = lf.ymin + (js - 1) * hy
            sx1 = sx0 + hx;  sy1 = sy0 + hy
            xc = (sx0 + sx1) / 2; yc = (sy0 + sy1) / 2
            # polynomial value at sub-cell centre (respects p_order)
            kc = JFloat(_leaf_value(lf, xc - _leaf_cx(lf), yc - _leaf_cy(lf),
                                    mf.p_order))
            kc == 0 && continue
            for i in eachindex(x)
                dax, day = _rect_deflection(x[i], y[i], sx0, sx1, sy0, sy1)
                ax[i] += kc * dax
                ay[i] += kc * day
            end
        end
    end
    return ax, ay
end

# ═══════════════════════════════════════════════════════════════
#  Barnes–Hut accelerated deflection (O(N log L) instead of O(N·L))
#
#  quadtree_deflection is O(n_leaves × n_points): every leaf contributes
#  to every point.  For image-plane rendering this is the bottleneck
#  (level-7 tree + 200² grid → 10-30 s/render).
#
#  Barnes–Hut replaces the flat leaf scan with a spatial tree:
#    • near leaves are integrated EXACTLY with _rect_deflection,
#    • far nodes are collapsed into a monopole multipole (mass M at
#      centre of mass), whose contribution is evaluated in O(1).
#  The opening criterion (MAC) s/r < θ decides near vs far; θ smaller
#  = more exact but slower, θ~0.5-0.8 is sub-percent accurate at
#  30-100× speedup (measured on the P1 NIE test case: θ=0.5 → 0.04%,
#  θ=0.8 → 0.24% median |α| error).
#
#  IMPORTANT consistency note: the exact leaf term uses the LEAF-CENTRE
#  constant κ × exact rectangle integral, i.e. exactly what
#  quadtree_deflection(mf, x, y; nsub=1) computes.  This guarantees
#  BH(θ→0) → exact(nsub=1), and the accuracy numbers above are vs that
#  same baseline.
# ═══════════════════════════════════════════════════════════════

# internal tree node (built once from the flat leaf list)
struct BHNode
    xmin::JFloat; xmax::JFloat; ymin::JFloat; ymax::JFloat
    cx::JFloat; cy::JFloat          # centre of mass of subtree
    M::JFloat                       # subtree total κ·area
    level::Int
    isleaf::Bool
    children::Vector{BHNode}
    leaf::Union{Nothing, QuadLeaf}
end

"""
    root = quadtree_bh_tree(mf)

Build the internal Barnes–Hut tree of the mass field's leaves (flat
leaf list → recursive subdivision).  The tree is a pure acceleration
structure; `mf` itself is untouched.  Leaves carry M = κ_c·area and
centre of mass at the area centroid; P1/P2 gradients are *not* folded
into higher multipoles (monopole only) — adequate at θ ≤ 0.8 for the
tested NIE fields.
"""
function quadtree_bh_tree(mf::MassField)
    leaves = _leaves(mf)
    # identify the unique leaf exactly filling a given box (complete tiling)
    function _match_leaf(xmin, xmax, ymin, ymax)
        ax = (xmax - xmin) * (ymax - ymin)
        for lf in leaves
            if (lf.xmax - lf.xmin) * (lf.ymax - lf.ymin) == ax &&
               lf.xmin >= xmin - 1e-12 && lf.xmax <= xmax + 1e-12 &&
               lf.ymin >= ymin - 1e-12 && lf.ymax <= ymax + 1e-12
                return lf
            end
        end
        return nothing
    end
    function _rec(xmin, xmax, ymin, ymax, lv)
        lf = _match_leaf(xmin, xmax, ymin, ymax)
        if lf !== nothing
            cx = JFloat((lf.xmin + lf.xmax) / 2)
            cy = JFloat((lf.ymin + lf.ymax) / 2)
            w = JFloat(lf.xmax - lf.xmin)
            h = JFloat(lf.ymax - lf.ymin)
            M = JFloat(lf.value) * w * h
            return BHNode(JFloat(xmin), JFloat(xmax), JFloat(ymin), JFloat(ymax),
                          cx, cy, M, lv, true, BHNode[], lf)
        end
        xmid = (xmin + xmax) / 2; ymid = (ymin + ymax) / 2
        kids = BHNode[_rec(xmin, xmid, ymin, ymid, lv + 1),
                      _rec(xmid, xmax, ymin, ymid, lv + 1),
                      _rec(xmin, xmid, ymid, ymax, lv + 1),
                      _rec(xmid, xmax, ymid, ymax, lv + 1)]
        M = sum(k.M for k in kids)
        cx = M > 0 ? sum(k.M * k.cx for k in kids) / M : JFloat((xmin + xmax) / 2)
        cy = M > 0 ? sum(k.M * k.cy for k in kids) / M : JFloat((ymin + ymax) / 2)
        return BHNode(JFloat(xmin), JFloat(xmax), JFloat(ymin), JFloat(ymax),
                      cx, cy, M, lv, false, kids, nothing)
    end
    return _rec(mf.tree.xmin, mf.tree.xmax, mf.tree.ymin, mf.tree.ymax, 0)
end

# monopole multipole kernel (green: no transcendentals, pure O(1))
@inline function _bh_mpole(q::BHNode, x::Real, y::Real)
    Rx = x - q.cx; Ry = y - q.cy
    R2 = Rx * Rx + Ry * Ry
    R2 < 1e-20 && return 0.0, 0.0
    invR2 = 1 / R2
    return Float64(q.M / pi) * Rx * invR2, Float64(q.M / pi) * Ry * invR2
end

"""
    ax, ay = quadtree_deflection_bh(root, x, y; theta=0.5, maxdepth=16)

Barnes–Hut deflection from a pre-built tree `root = quadtree_bh_tree(mf)`
at point(s) (x, y).  `theta` is the MAC opening angle (smaller = more
exact, slower); `maxdepth` guards pathological deep trees.  Returns two
arrays broadcastable to the shape of `x`, `y`.

For 200² grids on a level-7 P1 tree this is ~30-100× faster than
`quadtree_deflection` at θ∈[0.5,0.8] with sub-percent |α| error.
"""
function quadtree_deflection_bh(root::BHNode, x, y; theta::Real=0.5,
                                maxdepth::Int=16)
    ax = zeros(JFloat, size(x)); ay = zeros(JFloat, size(y))
    for i in eachindex(x)
        xi = x[i]; yi = y[i]
        sax = 0.0; say = 0.0
        stack = BHNode[root]
        while !isempty(stack)
            q = pop!(stack)
            dx = xi - q.cx; dy = yi - q.cy
            s = q.xmax - q.xmin
            if q.isleaf
                q.leaf === nothing && continue
                dax, day = _rect_deflection(xi, yi, q.leaf.xmin, q.leaf.xmax,
                                            q.leaf.ymin, q.leaf.ymax)
                sax += Float64(q.leaf.value) * dax
                say += Float64(q.leaf.value) * day
            elseif s / sqrt(dx * dx + dy * dy) < theta || q.level >= maxdepth
                dax, day = _bh_mpole(q, xi, yi)
                sax += dax; say += day
            else
                append!(stack, q.children)
            end
        end
        ax[i] = sax; ay[i] = say
    end
    return ax, ay
end

# ═══════════════════════════════════════════════════════════════
#  Potential ψ(θ) of the mass field   (Poisson: ∇²ψ = 2κ)
#
#  ψ(θ) = (1/π) ∫ κ(θ') ln|θ − θ'| d²θ'
#
#  For a uniform-κ rectangle the kernel has EXACT closed-form
#  antiderivative H(u,v) with ∂²H/∂u∂v = ln(u²+v²):
#      H(u,v) = u·v·ln(u²+v²) − 3uv + u²·atan(v/u) + v²·atan(u/v)
#  (derived by two integrations by parts; verified numerically to
#  sub-per-mille against a 400² grid).  The potential of one leaf is
#  the four-corner sum H(u2,v2) − H(u1,v2) − H(u2,v1) + H(u1,v1),
#  divided by 2π (because ln|θ−θ'| = ½·ln(u²+v²); this keeps the
#  consistency ∂ψ = α with the deflection antiderivative G).
#  Same branch convention as _G (principal atan).
#
#  P0 leaves: exact.  P1/P2: approximated by nsub² sub-rectangles
#  with κ taken at each sub-cell centre — exactly the SAME
#  discretisation level as quadtree_deflection, so ψ and α are
#  mutually consistent to the same order.
# ═══════════════════════════════════════════════════════════════

# antiderivative for the ln kernel: ∂²H/∂u∂v = ln(u²+v²)
@inline function _H(u::Real, v::Real)
    r2 = u * u + v * v
    r2 < 1e-20 && return 0.0           # singular corner: regularized
    th1 = atan(v, u)                    # atan2 branch ∈ (−π, π]
    if u < 0                            # restore principal atan(v/u)
        th1 = th1 > 0 ? th1 - pi : th1 + pi
    end
    th2 = atan(u, v)                    # principal atan(u/v) via swap
    if v < 0
        th2 = atan(u / v)
    end
    return u * v * log(r2) - 3 * u * v + u * u * th1 + v * v * th2
end

# Exact potential of a uniform-κ rectangle [xlo,xhi]×[ylo,yhi] at (x,y),
# independent of κ (caller multiplies); /(2π) factor built in.
#
# NOTE the /(2π): ψ = (1/π)∫∫ κ·ln|θ−θ'| d²θ'  =  (1/2π)∫∫ κ·ln(u²+v²) dudv,
# because ln|θ−θ'| = ½·ln(u²+v²) and H is the antiderivative of
# ln(u²+v²) (∂²H/∂u∂v = ln(u²+v²)).  A /π factor would double the
# potential and break the consistency ∂ψ = α with _rect_deflection.
# Corner convention matches _rect_deflection: u = x − x', v = y − y'.
function _rect_potential(x::Real, y::Real, xlo, xhi, ylo, yhi)
    u1 = x - xhi; u2 = x - xlo
    v1 = y - yhi; v2 = y - ylo
    return (_H(u2, v2) - _H(u1, v2) - _H(u2, v1) + _H(u1, v1)) / (2 * pi)
end

"""
    psi = quadtree_potential(mf, x, y; nsub=2)

Fermat potential ψ(θ) of the quad-tree mass field, consistent with
`quadtree_deflection` (same nsub sub-rectangle discretisation, same
principal-atan branch convention).  P0 leaves are exact; P1/P2 are
approximated at the sub-cell-centre κ level.  Returns an array
broadcastable to the shape of `x`, `y`.
"""
function quadtree_potential(mf::MassField, x, y; nsub::Int=2)
    leaves = _leaves(mf)
    psi = zeros(JFloat, size(x))
    for lf in leaves
        hx = (lf.xmax - lf.xmin) / nsub
        hy = (lf.ymax - lf.ymin) / nsub
        for js in 1:nsub, is in 1:nsub
            sx0 = lf.xmin + (is - 1) * hx
            sy0 = lf.ymin + (js - 1) * hy
            sx1 = sx0 + hx;  sy1 = sy0 + hy
            xc = (sx0 + sx1) / 2; yc = (sy0 + sy1) / 2
            kc = JFloat(_leaf_value(lf, xc - _leaf_cx(lf), yc - _leaf_cy(lf),
                                    mf.p_order))
            kc == 0 && continue
            for i in eachindex(x)
                psi[i] += kc * _rect_potential(x[i], y[i], sx0, sx1, sy0, sy1)
            end
        end
    end
    return psi
end

"""
    fxx, fxy, fyy = quadtree_hessian(mf, x, y; nsub=2, diff=0.0)

Second derivatives of the Fermat potential, obtained by O(h²) central
central differences of `quadtree_deflection` — the SAME strategy the
NIE/SIS models already use in this repo (see NIEkappa.LensHessian).
`diff=0.0` → adaptive h = cbrt(eps)·(1+|θ|) per point.  Works for any
p_order (no moment antiderivatives needed).  Returns three arrays
broadcastable to the shape of `x`, `y`.
"""
function quadtree_hessian(mf::MassField, x, y; nsub::Int=2, diff::Real=0.0)
    T = promote_type(eltype(x), eltype(y), Float64)
    ax = zeros(JFloat, size(x)); ay = zeros(JFloat, size(y))
    # adaptive O(h²)-optimal step, same formula as NIEkappa
    h0 = diff > 0 ? T(diff) : cbrt(eps(real(T)))
    hx = Float64.(h0) .* (1.0 .+ abs.(x))
    hy = Float64.(h0) .* (1.0 .+ abs.(y))

    fx_m, _    = quadtree_deflection(mf, x .- hx, y; nsub=nsub)
    fx_p, _    = quadtree_deflection(mf, x .+ hx, y; nsub=nsub)
    fx_ym, fyx = quadtree_deflection(mf, x, y .- hy; nsub=nsub)
    fx_yp, fyy = quadtree_deflection(mf, x, y .+ hy; nsub=nsub)
    _, fy_m    = quadtree_deflection(mf, x, y .- hy; nsub=nsub)
    _, fy_p    = quadtree_deflection(mf, x, y .+ hy; nsub=nsub)

    two_hx = 2 .* hx
    two_hy = 2 .* hy
    f_xx = (fx_p .- fx_m) ./ two_hx
    f_xy = (fx_yp .- fx_ym) ./ two_hy   # ∂f_x/∂y = ∂f_y/∂x to O(h²)
    f_yy = (fy_p .- fy_m) ./ two_hy
    return f_xx, f_xy, f_yy
end


# ═══════════════════════════════════════════════════════════════
#  Multipole deflection — accelerated (treecode-style) approximation
#
#  Instead of integrating the kernel EXACTLY over each rectangular
#  leaf (quadtree_deflection), approximate each leaf by its low-order
#  mass moments (standard Barnes–Hut / treecode idea).  For the P1
#  (linear) leaf field on a symmetric rectangle, all moments are
#  closed forms:
#      M   = value·A                          (monopole total mass)
#      D   = A·(gx·W²/12, gy·H²/12)           (dipole; linear field is asymmetric)
#      I2  = value·A·(W²+H²)/12               (scalar 2nd moment ∫κρ²)
#      Q   = value·A·diag(W², H²)/12          (2nd-moment tensor ∫κρρ)
#  (linear parts drop out of M, Q, I2 by odd symmetry; they only
#   appear in the dipole D.)
#
#  Expansion of the 2D Green's function, ψ(R)=(1/π)∫κ ln|R−ρ|d²ρ, to
#  order ρ²/R², with α = ∇ψ:
#      α(R) ≈ (1/π)[ M·R/R² − D/R² + 2(R·D)R/R⁴
#                    − I2·R/R⁴ − 2Q·R/R⁴ + 4(R·Q·R)R/R⁶ ]
#  Valid when |ρ|/|R| ≪ 1 (observation far from the leaf); the near
#  field is NOT accurate (monopole diverges toward the leaf centre) —
#  this is exactly the accuracy/speed trade-off treecodes exploit.
# ═══════════════════════════════════════════════════════════════

"""
    ax, ay = quadtree_deflection_multipole(mf, x, y; order=2)

Deflection α(θ) from the quad-tree mass field `mf`, approximating
each leaf by its multipole moments (treecode style) instead of the
exact rectangle integral:
- `order=1`: monopole only  (each leaf → point mass M at its centre)
- `order=2`: monopole + dipole + quadrupole
Returns arrays broadcastable to `x`, `y`.  Accurate in the FAR field
(|θ−leaf-centre| ≫ leaf size); near-field is approximate by design.
"""
function quadtree_deflection_multipole(mf::MassField, x, y; order::Int=2)
    leaves = _leaves(mf)
    ax = zeros(JFloat, size(x)); ay = zeros(JFloat, size(y))
    for lf in leaves
        W = lf.xmax - lf.xmin
        H = lf.ymax - lf.ymin
        v = JFloat(lf.value)
        gx = JFloat(lf.gx); gy = JFloat(lf.gy)
        A = JFloat(W * H)
        (v == 0 && gx == 0 && gy == 0) && continue
        # moments (leaf-local: ρ measured from leaf centre)
        M  = v * A
        Dx = A * gx * (W * W / 12);  Dy = A * gy * (H * H / 12)
        I2 = v * A * (W * W + H * H) / 12
        Qx = v * A * (W * W / 12);   Qy = v * A * (H * H / 12)   # Q = diag(Qx,Qy)
        cxc = _leaf_cx(lf); cyc = _leaf_cy(lf)
        for i in eachindex(x)
            Rx = x[i] - cxc; Ry = y[i] - cyc
            R2 = Rx * Rx + Ry * Ry
            R2 < 1e-20 && continue       # degenerate: skip obs at leaf centre
            R4 = R2 * R2
            # monopole
            ax[i] += (M / pi) * Rx / R2
            ay[i] += (M / pi) * Ry / R2
            if order >= 2
                RdotD = Rx * Dx + Ry * Dy
                RdotQR = Rx * (Qx * Rx) + Ry * (Qy * Ry)   # = R·Q·R  (Q diagonal)
                # dipole
                ax[i] += (1 / pi) * (-Dx / R2 + 2 * RdotD * Rx / R4)
                ay[i] += (1 / pi) * (-Dy / R2 + 2 * RdotD * Ry / R4)
                # quadrupole
                QRx = Qx * Rx; QRy = Qy * Ry
                cQ = 4 * RdotQR
                ax[i] += (1 / pi) * (-I2 * Rx / R4 - 2 * QRx / R4 + cQ * Rx / (R4 * R2))
                ay[i] += (1 / pi) * (-I2 * Ry / R4 - 2 * QRy / R4 + cQ * Ry / (R4 * R2))
            end
        end
    end
    return ax, ay
end

# ═══════════════════════════════════════════════════════════════
#  QuadTreeLens — adapt the reconstructed mass field to the unified
#  AbstractLens interface, so it can be used anywhere a lens model
#  (NIE, SIS, NFW, …) is expected: LensPlane, LensFermat,
#  LensMagnification, ForwardModel.render, etc.
#
#  Implements the same four functions as the analytic models:
#    lens_potential  → quadtree_potential   (P0 exact, P1/P2 nsub)
#    lens_derivative → quadtree_deflection  (existing)
#    lens_hessian    → quadtree_hessian     (central diff — NIE style)
#    lens_check      → no validated params (free-form field); no-op
# ═══════════════════════════════════════════════════════════════

"""
    QuadTreeLens(mf::MassField; norm=1.0, method=:exact, theta=0.5)

Wrap a reconstructed quad-tree `MassField` as an `AbstractLens`, giving
it the same four-function interface as the analytic lens models
(potential / derivative / hessian / check).  Scales per-leaf moments
by `norm` (default 1) for mass-calibration convenience.

Part of the quad-tree lens stack: bottom mesh = `LensMeshRefine`,
reconstruction = this module, image-plane fit layer = `QuadTreeFit`.

# Deflection method
- `method=:exact` (default): `quadtree_deflection` — O(leaves × points),
  exact to the nsub discretisation.  Slow for render loops.
- `method=:bh`: Barnes–Hut tree — O(points × log leaves), ~30-100×
  faster at sub-percent error (θ∈[0.5,0.8]) on P1 fields.  The tree is
  built once at construction and **rebuilt automatically whenever
  `mf` is mutated by `update_field!` / `refine_mass!`** (a `rev`
  counter on `MassField` tracks stale caches), so the lens is always
  consistent with the current field.

# Example
```julia
lens = QuadTreeLens(mf; method=:bh, theta=0.5)
ax, ay = Jens.LensBase.lens_derivative(lens, x, y)   # fast, ~1% error
psi    = Jens.LensBase.lens_potential(lens, x, y)
fxx, fxy, fyy = Jens.LensBase.lens_hessian(lens, x, y)
# refine the field further; the lens stays correct:
accumulate_residuals!(mf, xs, ys, r); update_field!(mf); refine_mass!(mf)
ax2, ay2 = Jens.LensBase.lens_derivative(lens, x, y)  # auto-rebuilt tree
```
"""
mutable struct QuadTreeLens <: AbstractLens
    mf::MassField
    norm::Float64
    method::Symbol
    theta::Float64
    bhtree::Union{Nothing, BHNode}
    bh_rev::UInt
    function QuadTreeLens(mf::MassField; norm::Real=1.0, method::Symbol=:exact,
                          theta::Real=0.5)
        method in (:exact, :bh) || error("method must be :exact or :bh")
        bt = method == :bh ? quadtree_bh_tree(mf) : nothing
        return new(mf, Float64(norm), method, Float64(theta), bt, mf.rev)
    end
end

function lens_derivative(l::QuadTreeLens, x, y; z_source=nothing, nsub::Int=2,
                         theta=nothing, kwargs...)
    if l.method == :bh
        l.bh_rev != l.mf.rev && _rebuild_bh!(l)
        th = theta === nothing ? l.theta : theta
        ax, ay = quadtree_deflection_bh(l.bhtree, x, y; theta=th)
        return l.norm .* ax, l.norm .* ay
    else
        ax, ay = quadtree_deflection(l.mf, x, y; nsub=nsub)
        return l.norm .* ax, l.norm .* ay
    end
end

lens_potential(l::QuadTreeLens, x, y; z_source=nothing, kwargs...) =
    l.norm .* quadtree_potential(l.mf, x, y; kwargs...)

lens_hessian(l::QuadTreeLens, x, y; z_source=nothing, nsub::Int=2, kwargs...) =
    l.norm .* quadtree_hessian(l.mf, x, y; nsub=nsub, kwargs...)

lens_check(::QuadTreeLens; kwargs...) = nothing   # field, not params

# rebuild cached Barnes–Hut tree after field mutation (internal)
function _rebuild_bh!(l::QuadTreeLens)
    l.bhtree = quadtree_bh_tree(l.mf)
    l.bh_rev = l.mf.rev
    return l
end

end # module LensMassRecon
