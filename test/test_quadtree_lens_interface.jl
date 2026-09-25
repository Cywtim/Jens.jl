# End-to-end: QuadTreeLens used through the SAME unified interface as
# analytic lenses (LensBase): LensPlane, LensMagnification, LensFermat.
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensBase: LensPlane, LensMagnification, LensFermat
using Jens.LensMeshRefine: n_leaves
import Jens.LensModel: NIEkappa, NIE

const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6
function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys; b=b, s=s, q=q, varphi=varphi)
    return (fxx .+ fyy) ./ 2
end

# ── reconstruct NIE as P1 quadtree (ext=3.0, 111² 采样) ──
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
    before = n_leaves(mf.tree)
    accumulate_residuals!(mf, xs, ys, rs)
    update_field!(mf)
    refine_mass!(mf; tau=5e-4, part=0.4)
    rs .= kt .- field_value(mf, xs, ys)
    it >= 4 && n_leaves(mf.tree) == before && break
end
println("P1 quadtree leaves = ", n_leaves(mf.tree))
lens = QuadTreeLens(mf)

# ── evaluation grid (visible arc region) ──
gd = 121
ge = collect(range(-1.0, 1.0; length=gd))
Xg = repeat(reshape(ge, :, 1), 1, gd)
Yg = repeat(reshape(ge, 1, :), gd, 1)

# truth NIE as reference
ref = (LensModel=NIE, theta_E=1.2, s_scale=0.15, e1=0.1, e2=0.0,
       xcentre=0.0, ycentre=0.0)
function nie_deriv(xg, yg)
    ax, ay = NIE.LensDerivative(xg, yg; theta_E=1.2, s_scale=0.15,
                                e1=0.1, e2=0.0, xcentre=0.0, ycentre=0.0)
    return ax, ay
end
function nie_pot(xg, yg)
    return NIE.LensPotential(xg, yg; theta_E=1.2, s_scale=0.15,
                             e1=0.1, e2=0.0, xcentre=0.0, ycentre=0.0)
end
function nie_hess(xg, yg)
    return NIE.LensHessian(xg, yg; theta_E=1.2, s_scale=0.15,
                           e1=0.1, e2=0.0, xcentre=0.0, ycentre=0.0, diff=1e-6)
end

println("\n=== A) LensPlane: β = θ − α(θ) (QuadTreeLens via unified interface) ===")
bx_q, by_q = LensPlane(Xg, Yg; LensModel=lens)
bx_t, by_t = LensPlane(Xg, Yg; LensModel=NIE,
                       LensKwargs=(theta_E=1.2, s_scale=0.15,
                                   e1=0.1, e2=0.0))
# compare at a few points (arc region, away from scale degeneracy by
# comparing the SHAPE of the β map after matching the overall scale)
ax_q, ay_q = Jens.LensBase.lens_derivative(lens, Xg, Yg)
ax_t, ay_t = nie_deriv(Xg, Yg)
# relative scale between qtree and truth deflection
scl = sum(ax_t .* ax_q .+ ay_t .* ay_q) / sum(ax_q.^2 .+ ay_q.^2)
println("  α scale (qtree vs NIE) = ", round(scl; digits=4), "   (mass-sheet residual)")
err = sqrt.((ax_t .- scl .* ax_q).^2 .+ (ay_t .- scl .* ay_q).^2) ./ sqrt.(ax_t.^2 .+ ay_t.^2)
println("  deflection residual after scale: med = ",
        round(100*median(vec(err)); digits=2), "%   p90 = ", round(100*quantile(vec(err),0.9); digits=2), "%")

println("\n=== B) LensMagnification: μ = 1/((1−h_xx)(1−h_yy) − h_xy²) ===")
μ_q = LensMagnification(Xg, Yg; LensModel=lens)
μ_t = LensMagnification(Xg, Yg; LensModel=NIE,
                        LensKwargs=(theta_E=1.2, s_scale=0.15,
                                    e1=0.1, e2=0.0))
mbig = abs.(μ_t) .> 5   # near critical curve, skip singular regions
rel_m = abs.(μ_q[mbig] .- μ_t[mbig]) ./ (abs.(μ_t[mbig]) .+ 1e-9)
println("  critical-curve pixels (|μ|>5): ", count(mbig),
        "   |μ| rel diff: med=", round(100*median(rel_m); digits=1), "%   max=",
        round(100*quantile(rel_m,0.95); digits=1), "%")

println("\n=== C) LensFermat: τ = ½|β−θ|² − ψ ===")
τ_q = LensFermat(Xg, Yg, [0.0, 0.0]; LensModel=lens)
τ_t = LensFermat(Xg, Yg, [0.0, 0.0]; LensModel=NIE,
                 LensKwargs=(theta_E=1.2, s_scale=0.15,
                             e1=0.1, e2=0.0))
# τ is only defined up to a constant (mass-sheet); compare after mean-subtraction
τ_q0 = τ_q .- mean(τ_q); τ_t0 = τ_t .- mean(τ_t)
corr = sum(τ_q0 .* τ_t0) / sqrt(sum(τ_q0.^2) * sum(τ_t0.^2))
println("  τ correlation (mean-subtracted) = ", round(corr; digits=5))

println("\n=== D) 冒烟: LensMagnification + LensFermat 数值有限? ===")
println("  μ_q 有限: ", all(isfinite, μ_q), "   τ_q 有限: ", all(isfinite, τ_q))
