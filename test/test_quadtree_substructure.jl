# Quadtree value for COMPLEX lenses: capture what a parameterised base
# lens cannot express (an off-centre sub-halo / blob).
#   κ_true(θ) = κ_NIE(θ; θ*)               (the parameterised part)
#             + κ_blob(θ)                   (un-parameterised substructure)
#   Model:    analytic NIE(θ) + quadtree absorbing κ_true − κ_NIE(θ)
#   Verify:   the quadtree residual field peaks at the blob location,
#             i.e. substructure IS captured by the quadtree part.
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
import Jens.LensModel: NIE
using Jens.LensBase: SingleModel, lens_hessian

const bT, sT, e1T = 1.2, 0.15, 0.142857
# blob: off-centre Gaussian κ bump (sub-halo proxy)
const blob_amp, blob_sig, blob_x, blob_y = 0.35, 0.12, 0.55, -0.35
lens_true = SingleModel(NIE; theta_E=bT, s_scale=sT, e1=e1T, e2=0.0)

function κ_nie(lens, xs, ys)
    hxx, hxy, hyy = lens_hessian(lens, xs, ys)
    return (hxx .+ hyy) ./ 2
end
function κ_blob(xs, ys)
    return blob_amp .* exp.(-((xs .- blob_x).^2 .+ (ys .- blob_y).^2) ./ (2*blob_sig^2))
end
κ_true_fn(xs, ys) = κ_nie(lens_true, xs, ys) .+ κ_blob(xs, ys)

# ═══════ reconstruct quadtree absorbing κ_true − κ_NIE(θ_correct) ═══════
EXT = 3.0; nx = 90
c = collect(range(-EXT, EXT; length=nx))
xs = Float64[]; ys = Float64[]
for y in c, x in c; push!(xs,x); push!(ys,y); end
r_NIE = κ_nie(lens_true, xs, ys)
r_true = κ_true_fn(xs, ys)
resid = r_true .- r_NIE          # what the quadtree must absorb (blob only)

mf = MassField(-EXT,EXT,-EXT,EXT; max_level=7, eta=0.5, damp_pow=0.0,
               min_obs=1, p_order=1)
rs = collect(resid)
for it in 1:25
    accumulate_residuals!(mf, xs, ys, rs); update_field!(mf)
    refine_mass!(mf; tau=5e-4, part=0.4, roi=1.2, cap_outside=4)
    rs .= resid .- field_value(mf, xs, ys)
end

println("叶数 = ", Jens.LensMeshRefine.n_leaves(mf.tree))
println("重建残差 rms = ", round(sqrt(mean(rs.^2)); sigdigits=3))

# ═══════ evaluate reconstructed κ at the blob centre vs far away ═══════
kc = field_value(mf, [blob_x], [blob_y])
kfar = field_value(mf, [-blob_x], [-blob_y])
println("\nquadtree κ 在 blob 中心 (0.55,-0.35)      = ", round(kc[1]; sigdigits=4))
println("quadtree κ 在镜像远处 ($(-blob_x),$(-blob_y)) = ", round(kfar[1]; sigdigits=4))
println("真 blob κ 中心 = ", round(blob_amp; sigdigits=4))
println("→ blob 是否被捕捉: ", kc[1] > 3*abs(kfar[1]) && kc[1] > 0.5*blob_amp ?
      "✅ 是 (中心显著增强, 远侧≈0)" : "❌ 否")

# ═══════ does the quadtree ADD value over pure analytic? ═══════
# correct comparison: total κ = analytic(NIE) + quadtree residual field,
# fitted against κ_true with one global mass-sheet scale allowed
kv = field_value(mf, xs, ys)
total_hybrid = r_NIE .+ kv
scl_h = sum(r_true .* total_hybrid) / sum(total_hybrid.^2)
scl_a = sum(r_true .* r_NIE) / sum(r_NIE.^2)
χ2_analytic = sum((r_true .- scl_a .* r_NIE).^2)
χ2_hybrid   = sum((r_true .- scl_h .* total_hybrid).^2)
println("\n纯解析 vs 解析+quadtree 的 κ 拟合残差 (χ²):")
println("  纯 NIE(θ*):          χ² = ", round(χ2_analytic; sigdigits=4),
        "  rms=", round(sqrt(mean((r_true .- scl_a .* r_NIE).^2)); sigdigits=3))
println("  NIE + quadtree 残差:  χ² = ", round(χ2_hybrid; sigdigits=4),
        "  rms=", round(sqrt(mean((r_true .- scl_h .* total_hybrid).^2)); sigdigits=3))
println("  改善 = ", round(100*(1 - χ2_hybrid/χ2_analytic); digits=1), "%  ← blob 被捕捉的量化价值")
