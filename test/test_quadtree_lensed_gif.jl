# ═══════════════════════════════════════════════════════════════
#  Lensed-image GIF: how a Gaussian source looks through a P1
#  quadtree-masses reconstruction as max_level grows 1 → 8.
#
#  Each frame shows a 2-panel figure:
#    left  — the quadtree mass distribution κ(x,y) + grid overlay
#    right — the lensed image:  β = θ − α(θ),  I = Gauss(β),
#            rendered with the SAME quadtree deflection.
#  The true NIE lensed image is shown as a pale reference on the right.
#
#  Levels stop at 8: quadtree_deflection is O(leaves × nsub² × pixels),
#  so deeper trees cost minutes per frame with no visible image gain.
#  All 8 PNG frames are kept under img/quadtree_lensed_frames/ and the
#  assembled GIF goes to img/quadtree_lensed_fit.gif.
# ═══════════════════════════════════════════════════════════════
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine
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
const TAU_BASE = 5e-4
function tau_for_level(lv)
    lv <= 8 ? TAU_BASE : TAU_BASE / 2^(lv - 8)
end

# ── reconstruction sampling (uniform over the domain) ──
nx = 161
c = collect(LinRange(-EXT, EXT, nx))
xs = Float64[]; ys = Float64[]
for y in c, x in c
    push!(xs, x); push!(ys, y)
end
kt = nie_kappa(xs, ys)

# ── Gaussian source in the SOURCE plane (β) ──
#    slightly offset from the lens centre so we get an arc, not a ring
src_βx, src_βy = 0.18, 0.10
src_σ  = 0.06
src_I(βx, βy) = exp(-((βx - src_βx)^2 + (βy - src_βy)^2) / (2 * src_σ^2))

# ── lensed-image renderer (vectorized): I(θ) = src(θ − α(θ)) ──
# (inline broadcasting in the main loop; no separate mesh helper needed)

outdir = joinpath(@__DIR__, "..", "img", "quadtree_lensed_frames")
mkpath(outdir)

# evaluation/display grid for lensed images (image plane, arc region)
# modest resolution (180²) so deep-tree rendering stays affordable
θwin = 0.9
gd = 180
ge = collect(range(-θwin, θwin; length=gd))   # for image
Xg = repeat(reshape(ge, :, 1), 1, gd)          # (gd, gd): columns are x
Yg = repeat(reshape(ge, 1, :), gd, 1)          # rows are y
# mass-field display grid
md = collect(range(-EXT + 0.4, EXT - 0.4, length=240))
Xm = [x for y in md, x in md]
Ym = [y for y in md, x in md]
maxk = maximum(nie_kappa(vec(Xm), vec(Ym)))

# true NIE lensed image ONCE (independent of level)
at_x, at_y = NIEkappa.LensDerivative(Xg, Yg; b=b, s=s, q=q, varphi=varphi)
img_t = src_I.(Xg .- at_x, Yg .- at_y)

MAXLV = 8
for po in (1,)           # P1 (could loop p_order ∈ 0,1,2)
    for lv in 1:MAXLV
        mf = MassField(-EXT, EXT, -EXT, EXT; max_level=lv, eta=0.5,
                       damp_pow=0.0, min_obs=1, p_order=po)
        rs = collect(kt)
        lt_since = 0
        for it in 1:60
            before = n_leaves(mf.tree)
            accumulate_residuals!(mf, xs, ys, rs)
            update_field!(mf)
            refine_mass!(mf; tau=tau_for_level(lv), part=PART)
            rs .= kt .- field_value(mf, xs, ys)
            lt_since = n_leaves(mf.tree) == before ? lt_since + 1 : 0
            lt_since >= 3 && break
        end
        L = n_leaves(mf.tree)

        # reconstructed deflection over the whole image mesh (one call);
        # nsub=1 keeps the deep-tree frames affordable
        axa, aya = quadtree_deflection(mf, Xg, Yg; nsub=1)
        img_q = src_I.(Xg .- axa, Yg .- aya)

        # ── 2-panel frame ──
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.5, 5.4))
        # left: quadtree mass distribution
        km = field_value(mf, vec(Xm), vec(Ym))
        imL = axL.imshow(reshape(km, 240, 240),
                         extent=(-(EXT-0.4), EXT-0.4, -(EXT-0.4), EXT-0.4),
                         origin="lower", cmap="viridis", vmax=maxk)
        for lf in mf.tree.leaves
            axL.add_patch(patches.Rectangle((lf.xmin, lf.ymin),
                                            lf.xmax-lf.xmin, lf.ymax-lf.ymin,
                                            fill=false, edgecolor="white",
                                            lw=0.35, alpha=0.7))
        end
        axL.set_title("P$po quadtree κ  ·  max_level=$lv  ·  $(L) leaves")
        axL.set_xlim(-(EXT-0.4), EXT-0.4); axL.set_ylim(-(EXT-0.4), EXT-0.4)
        fig.colorbar(imL, ax=axL, fraction=0.046, pad=0.04, label="κ")
        axL.set_xlabel("x"); axL.set_ylabel("y")

        # right: lensed image (quadtree lens) over faint true-NIE image
        axR.imshow(img_t, extent=(-θwin, θwin, -θwin, θwin),
                   origin="lower", cmap="inferno", alpha=0.20, vmax=1.0)
        imR = axR.imshow(img_q, extent=(-θwin, θwin, -θwin, θwin),
                         origin="lower", cmap="inferno", alpha=0.85, vmax=1.0)
        axR.set_title("lensed image: β=θ−α(θ)  (pale = true NIE)")
        axR.set_xlabel("θ_x"); axR.set_ylabel("θ_y")
        axR.set_xlim(-θwin, θwin); axR.set_ylim(-θwin, θwin)
        fig.colorbar(imR, ax=axR, fraction=0.046, pad=0.04, label="I")

        fig.suptitle("P$po quadtree as a lens — max_level $lv  ($L leaves)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = joinpath(outdir, "quadtree_lensed_P$(po)_lv$(lpad(lv,2,'0')).png")
        fig.savefig(fname, dpi=130)
        plt.close(fig)
        println("P$po lv=$lv leaves=$L  →  $fname")
    end
end
println("\nAll frames → ", outdir)
