# ═══════════════════════════════════════════════════════════════
#  test_quadtree_deflection_vs_nie.jl
#
#  对比: quadtree_deflection (quad-tree 质量场 → 偏折角 α)
#        vs NIE (解析质量剖面的偏折)
#
#  流程:
#    1. 用 NIE (非奇异等温椭球) 作为真值: κ_true = (f_xx+f_yy)/2,
#       真偏折 α_true = NIEkappa.LensDerivative
#    2. LensMassRecon 从残差迭代重建 quad-tree κ 场
#    3. quadtree_deflection 由重建场算偏折 α_recon
#    4. 对比: 最优线性标度 scale + 相对残差分布
#
#  验证要点 (实测基线, max_level=8, 80×80 采样):
#    scale        ≈ 1.11   (在 0.6..1.6 之间)
#    median 相对残差 ≈ 0.095 (< 0.25)
#    p90  相对残差 ≈ 0.19  (< 0.5)
#
#  运行:
#    julia --project=. test/test_quadtree_deflection_vs_nie.jl
# ═══════════════════════════════════════════════════════════════

using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: n_leaves
import Jens.LensModel: NIEkappa

# ── 真值: NIE 质量剖面 + 真偏折 ─────────────────────────────
const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6

function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys;
        b = b, s = s, q = q, varphi = varphi)
    return (fxx .+ fyy) ./ 2
end

# 均匀采样观测点 ([−1.8,1.8]², 80×80)
nx = 80
cc = collect(LinRange(-1.8, 1.8, nx))
xs = Float64[]; ys = Float64[]
for y in cc, x in cc
    push!(xs, x); push!(ys, y)
end
k_true = nie_kappa(xs, ys)
ax_t, ay_t = NIEkappa.LensDerivative(xs, ys;
    b = b, s = s, q = q, varphi = varphi)

println("=== NIE 真值 ===")
println("  参数: b=$b s=$s q=$q φ=$(round(varphi; digits=3)) rad")
println("  κ 范围: ", round(minimum(k_true); digits=4), " … ",
        round(maximum(k_true); digits=4))
println("  |α| 范围: ", round(minimum(sqrt.(ax_t .^ 2 .+ ay_t .^ 2)); digits=3),
        " … ", round(maximum(sqrt.(ax_t .^ 2 .+ ay_t .^ 2)); digits=3))

# ── LensMassRecon 迭代重建 ──────────────────────────────────
mf = MassField(-2.0, 2.0, -2.0, 2.0; max_level = 8, eta = 0.5,
               damp_pow = 0.0, min_obs = 1)
rs = collect(k_true)

for it in 1:10
    accumulate_residuals!(mf, xs, ys, rs)
    update_field!(mf)
    refine_mass!(mf; tau = 0.01, part = 0.4)
    # residual uses the FULL P1 model (leaf value + slope), not leaf-centre value
    # (in-place broadcast: no rebinding, avoids soft-scope at top level)
    rs .= k_true .- p1_value(mf, xs, ys)
end

# ── 由重建场算偏折 (quadtree_deflection) ────────────────────
ax_r, ay_r = quadtree_deflection(mf, xs, ys)
mag_t = sqrt.(ax_t .^ 2 .+ ay_t .^ 2)
mag_r = sqrt.(ax_r .^ 2 .+ ay_r .^ 2)

# 最优线性标度 (α_t ≈ scale·α_r, 最小二乘)
scale = sum(ax_t .* ax_r .+ ay_t .* ay_r) / sum(ax_r .^ 2 .+ ay_r .^ 2)

# 相对残差分布
resid2 = sqrt.((ax_t .- scale .* ax_r) .^ 2 .+ (ay_t .- scale .* ay_r) .^ 2)
relax = resid2 ./ (mag_t .+ 1e-6)

println("\n=== 对比: quadtree_deflection vs NIE 真偏折 ===")
println("  重建叶子数: ", n_leaves(mf.tree), "  (均匀 80×80 = ", nx * nx, ")")
println("  |α_recon| 范围: ", round(minimum(mag_r); digits=3), " … ",
        round(maximum(mag_r); digits=3))
println("  最优线性标度 scale: ", round(scale; digits=3), "  (理想=1)")
println("  相对残差: median=", round(median(relax); digits=3),
        " mean=", round(mean(relax); digits=3),
        " p90=", round(quantile(relax, 0.9); digits=3))

# ── 断言 (固化基线) ──────────────────────────────────────────
@assert 0.6 < scale < 1.6    "最优线性标度应在 (0.6,1.6) (实测 ~1.11)"
@assert median(relax) < 0.25 "median 相对残差应 <0.25 (实测 ~0.10)"
@assert quantile(relax, 0.9) < 0.5  "p90 相对残差应 <0.5 (实测 ~0.19)"
@assert n_leaves(mf.tree) < 1000    "自适应叶子应 << 均匀网格 (实测 ~127)"

println("\n=== 全部通过 ===")
