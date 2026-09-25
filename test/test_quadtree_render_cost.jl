# QuadTreeLens inside ForwardModel + render, with an honest cost
# comparison against an analytic lens (NIE).
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensSystem: ForwardModel, render
using Jens.LensGenerator: GenGrid, LensedPlane, LightPlane
using Jens.LensBase: SingleModel
using Jens.LightModel: ExtendedSource
import Jens.LensModel: NIE, NIEkappa
using Jens.LightModel.GaussianLight: GaussianSphere
using Jens.LensCosmo: Cosmology

const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6
function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys; b=b, s=s, q=q, varphi=varphi)
    return (fxx .+ fyy) ./ 2
end

# ── reconstruct quadtree (P1) from NIE κ ──
EXT = 3.0
nx = 111
c = collect(LinRange(-EXT, EXT, nx))
xs = Float64[]; ys = Float64[]
for y in c, x in c
    push!(xs, x); push!(ys, y)
end
kt = nie_kappa(xs, ys)
mf = MassField(-EXT, EXT, -EXT, EXT; max_level=8, eta=0.5,
               damp_pow=0.0, min_obs=1, p_order=1)
rs = collect(kt)
for it in 1:40
    before = Jens.LensMeshRefine.n_leaves(mf.tree)
    accumulate_residuals!(mf, xs, ys, rs)
    update_field!(mf)
    refine_mass!(mf; tau=5e-4, part=0.4)
    rs .= kt .- field_value(mf, xs, ys)
    it >= 4 && Jens.LensMeshRefine.n_leaves(mf.tree) == before && break
end
println("P1 quadtree leaves = ", Jens.LensMeshRefine.n_leaves(mf.tree))

# ── source (Gaussian sphere via ExtendedSource) ──
src = ExtendedSource(GaussianSphere; amp=1.0, sigma=0.05,
                     xcentre=0.18, ycentre=0.10)

# ── forward models: analytic (NIE) vs quadtree ──
cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
grid = GenGrid(pix_n=200, pix_size=0.018)   # ~3.6 arcsec box

lens_nie = SingleModel(NIE; theta_E=1.2, s_scale=0.15, e1=0.1, e2=0.0)
lens_qt  = QuadTreeLens(mf)

sys_nie = ForwardModel(
    lens_plane   = LensedPlane(lens_nie; z_lens=0.5, cosmology=cosmo),
    source_plane = LightPlane(src; z=1.5),
    grid         = grid,
)
sys_qt = ForwardModel(
    lens_plane   = LensedPlane(lens_qt; z_lens=0.5, cosmology=cosmo),
    source_plane = LightPlane(src; z=1.5),
    grid         = grid,
)

println("\n=== render() through ForwardModel ===")
t0 = time()
img_nie = render(sys_nie)
t_nie = time() - t0
println("  NIE render:      ", round(t_nie; digits=3), " s   (analytic, O(N))")

t0 = time()
img_qt = render(sys_qt)
t_qt = time() - t0
println("  QuadTree render: ", round(t_qt; digits=3), " s   (quadtree, O(leaves·nsub²·N))")
println("  speedup ratio (NIE / quadtree): ", round(t_nie / t_qt; digits=1), "×")

# compare images (after matching overall flux scale — mass-sheet)
scl = sum(img_nie .* img_qt) / sum(img_qt .^ 2)
rel = abs.(img_nie .- scl .* img_qt) ./ (img_nie .+ 1e-6)
mbright = (img_nie .> 0.2 * maximum(img_nie))
println("\n  image residual (bright px, after flux scale): med=",
        round(100 * median(vec(rel[mbright])); digits=2), "%")
println("  flux scale (qt vs NIE) = ", round(scl; digits=4))

# cost scaling: deflection cost vs #leaves
println("\n=== cost scaling: quadtree deflection vs leaves (200² grid) ===")
xg, yg = grid.xg, grid.yg
for lv in (4, 6, 8)
    mf2 = MassField(-EXT, EXT, -EXT, EXT; max_level=lv, eta=0.5,
                    damp_pow=0.0, min_obs=1, p_order=1)
    r2 = collect(kt)
    for it in 1:40
        b2 = Jens.LensMeshRefine.n_leaves(mf2.tree)
        accumulate_residuals!(mf2, xs, ys, r2)
        update_field!(mf2)
        refine_mass!(mf2; tau=5e-4, part=0.4)
        r2 .= kt .- field_value(mf2, xs, ys)
        it >= 4 && Jens.LensMeshRefine.n_leaves(mf2.tree) == b2 && break
    end
    L2 = Jens.LensMeshRefine.n_leaves(mf2.tree)
    t0 = time()
    quadtree_deflection(mf2, xg, yg; nsub=2)
    t1 = time() - t0
    println("  max_level=$lv  leaves=$L2  deflection: ", round(t1; digits=3),
            " s    (NIE render was ", round(t_nie; digits=3), " s)")
end
