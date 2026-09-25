# Cost + quality trade-off: exact quadtree deflection (nsub=1/2) vs
# multipole, on the same lensed-image region as the render test.
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
import Jens.LensModel: NIEkappa, NIE

const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6
function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys; b=b, s=s, q=q, varphi=varphi)
    return (fxx .+ fyy) ./ 2
end
function nie_deriv(xg, yg)
    return NIE.LensDerivative(xg, yg; theta_E=1.2, s_scale=0.15,
                              e1=0.1, e2=0.0, xcentre=0.0, ycentre=0.0)
end

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
L = Jens.LensMeshRefine.n_leaves(mf.tree)
println("P1 leaves = $L")

# 200² grid
gd = 200
ge = collect(range(-0.9, 0.9; length=gd))
Xg = repeat(reshape(ge, :, 1), 1, gd)
Yg = repeat(reshape(ge, 1, :), gd, 1)
ax_t, ay_t = nie_deriv(Xg, Yg)
mg = sqrt.(ax_t.^2 .+ ay_t.^2) .> 0.2   # arc region (bright)

println("\n=== 成本 (200², $L 叶, 一次 deflection) ===")
for (name, f) in [
        ("exact nsub=1", () -> quadtree_deflection(mf, Xg, Yg; nsub=1)),
        ("exact nsub=2", () -> quadtree_deflection(mf, Xg, Yg; nsub=2)),
        ("multipole order=2", () -> quadtree_deflection_multipole(mf, Xg, Yg; order=2)),
    ]
    t0 = time(); ax, ay = f(); dt = time() - t0
    # quality vs true NIE (after scale match on the arc region)
    scl = sum(ax_t[mg] .* ax[mg] .+ ay_t[mg] .* ay[mg]) / sum(ax[mg].^2 .+ ay[mg].^2)
    err = sqrt.((ax_t .- scl .* ax).^2 .+ (ay_t .- scl .* ay).^2) ./ (mg .+ sqrt.(ax_t.^2 .+ ay_t.^2))
    println("  $(name):  $(round(dt; digits=2)) s   |α| rel err (arc region): med=",
            round(100*median(vec(err[mg])); digits=2), "%")
end
