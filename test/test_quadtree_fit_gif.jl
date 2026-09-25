# ═══════════════════════════════════════════════════════════════
#  Quadtree refinement GIF: P1 reconstruction growing with max_level
#  from 1 → 10, over a semi-transparent (alpha=0.2) NIE truth field.
#
#  Each frame is saved as a PNG under img/quadtree_fit_frames/
#  (kept on disk!), and the frames are then assembled into
#  img/quadtree_fit.gif  (assembled by scripts/make_quadtree_gif.py).
# ═══════════════════════════════════════════════════════════════
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: n_leaves
using PyPlot
const patches = PyPlot.pyimport("matplotlib.patches")
import Jens.LensModel: NIEkappa

const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6
function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys; b=b, s=s, q=q, varphi=varphi)
    return (fxx .+ fyy) ./ 2
end

const EXT = 3.0
const PART = 0.4

# residual threshold per level: coarse levels use a loose tau so the
# tree grows visibly with each level; for the deepest levels we only
# tighten the threshold 2× per extra level so refinement continues
# GENTLY — dropping it 10× made lv9/lv10 subdivide the whole domain
# into a uniform fine mesh (36k leaves), losing the "fine core / coarse
# exterior" shape that a quadtree really has.
const TAU_BASE = 5e-4
function tau_for_level(lv)
    lv <= 8 ? TAU_BASE : TAU_BASE / 2^(lv - 8)
end

# reconstruction sampling points (uniform, covers the whole domain so
# every leaf can accumulate observations up to its max_level)
nx = 161
c = collect(LinRange(-EXT, EXT, nx))
xs = Float64[]; ys = Float64[]
for y in c, x in c
    push!(xs, x); push!(ys, y)
end
kt = nie_kappa(xs, ys)

# display grid (slightly inside the domain for a nicer view)
gd = 300
d = collect(LinRange(-EXT + 0.15, EXT - 0.15, gd))
Xg = [x for y in d, x in d]
Yg = [y for y in d, x in d]
kt_g = nie_kappa(vec(Xg), vec(Yg))
maxk = maximum(kt_g)

outdir = joinpath(@__DIR__, "..", "img", "quadtree_fit_frames")
mkpath(outdir)

cmap = Dict(0 => "viridis", 1 => "viridis", 2 => "viridis")

function draw_frame(po, lv, krec, lrec)
    fig, ax = plt.subplots(figsize=(7.2, 6.0))
    # 1) NIE truth as a pale backdrop (alpha=0.2)
    ax.imshow(reshape(kt_g, gd, gd), extent=(-(EXT-0.15), EXT-0.15, -(EXT-0.15), EXT-0.15),
              origin="lower", cmap="inferno", alpha=0.2, vmax=maxk)
    # 2) P1 reconstructed kappa field
    k2 = reshape(krec, gd, gd)
    im = ax.imshow(k2, extent=(-(EXT-0.15), EXT-0.15, -(EXT-0.15), EXT-0.15),
                   origin="lower", cmap=cmap[po], alpha=0.85, vmax=maxk)
    # 3) quadtree grid lines
    for lf in lrec
        ax.add_patch(patches.Rectangle((lf.xmin, lf.ymin), lf.xmax-lf.xmin, lf.ymax-lf.ymin,
                                       fill=false, edgecolor="white", lw=0.5, alpha=0.75))
    end
    ax.set_xlim(-(EXT-0.15), EXT-0.15)
    ax.set_ylim(-(EXT-0.15), EXT-0.15)
    ax.set_title("P$po  quadtree  max_level=$lv  ·  $(n_leaves) leaves  ·  pale = NIE (α=0.2)",
                 fontsize=11)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="κ")
    fig.tight_layout()
    fname = joinpath(outdir, "quadtree_P$(po)_lv$(lpad(lv,2,'0')).png")
    fig.savefig(fname, dpi=140)
    plt.close(fig)
    return fname
end

# main: P1 (p_order=1), levels 1..10
po = 1
for lv in 1:10
    mf = MassField(-EXT, EXT, -EXT, EXT; max_level=lv, eta=0.5,
                   damp_pow=0.0, min_obs=1, p_order=po)
    rs = collect(kt)
    lt_since = 0      # consecutive iterations without growth
    for it in 1:60
        before = n_leaves(mf.tree)
        accumulate_residuals!(mf, xs, ys, rs)
        update_field!(mf)
        refine_mass!(mf; tau=tau_for_level(lv), part=PART)
        rs .= kt .- field_value(mf, xs, ys)
        lt_since = n_leaves(mf.tree) == before ? lt_since + 1 : 0
        lt_since >= 3 && break   # converged: 3 stable passes
    end
    krec = field_value(mf, vec(Xg), vec(Yg))
    fname = draw_frame(po, lv, krec, mf.tree.leaves)
    println("level=$lv  leaves=$(n_leaves(mf.tree))  →  $fname")
end
println("\nAll frames saved under: ", outdir)
