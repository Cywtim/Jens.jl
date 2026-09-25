# ═══════════════════════════════════════════════════════════════
#  test_quadtree_mass_vs_nie.jl
#
#  对比: LensMassRecon (quad-tree 质量重建) vs NIE (解析质量剖面)
#
#  流程:
#    1. 用 NIE (非奇异等温椭球) 作为"真值"质量分布:
#       κ_true(x,y) = (f_xx+f_yy)/2  来自 NIEkappa.LensHessian
#    2. 在 [−2,2]² 均匀采样 κ_true 作为观测残差
#    3. LensMassRecon 从残差迭代重建 quad-tree κ 场
#    4. 对比: 总质量 / 峰值 / RMS / 自适应叶子数
#
#  验证要点 (对应实测基线):
#    总质量 ratio ≈ 99.6%   (> 90% 门限)
#    峰值恢复   ≈ 91%
#    相对 RMS   ≈ 6%
#    叶子数     ≈ 73  (远小于均匀 80×80=6400)
#
#  运行:
#    julia --project=. test/test_quadtree_mass_vs_nie.jl
# ═══════════════════════════════════════════════════════════════

using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: n_leaves
import Jens.LensModel: NIEkappa

# ── 真值: NIE 质量剖面 ──────────────────────────────────────
const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6

function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys;
        b = b, s = s, q = q, varphi = varphi,
        xcentre = 0., ycentre = 0.)
    return (fxx .+ fyy) ./ 2
end

# 均匀采样观测点 ([−1.9,1.9]², 80×80)
nx = 80
cc = collect(LinRange(-1.9, 1.9, nx))
xs = Float64[]; ys = Float64[]
for y in cc, x in cc
    push!(xs, x); push!(ys, y)
end
k_true = nie_kappa(xs, ys)

println("=== NIE 真值 ===")
println("  参数: b=$b s=$s q=$q φ=$(round(varphi; digits=3)) rad")
println("  κ 范围: ", round(minimum(k_true); digits=4), " … ",
        round(maximum(k_true); digits=4))

# ── LensMassRecon 迭代重建 ──────────────────────────────────
mf = MassField(-2.0, 2.0, -2.0, 2.0; max_level = 7, eta = 0.5,
               damp_pow = 0.0, min_obs = 1)
rs = collect(k_true)

for it in 1:8
    accumulate_residuals!(mf, xs, ys, rs)
    update_field!(mf)
    refine_mass!(mf; tau = 0.01, part = 0.4)
    # residual uses the FULL P1 model (leaf value + slope), not leaf-centre value
    # (in-place broadcast: no rebinding, avoids soft-scope at top level)
    rs .= k_true .- p1_value(mf, xs, ys)
end

# ── 对比 ────────────────────────────────────────────────────
k_recon = p1_value(mf, xs, ys)
resid = k_true .- k_recon

dA = (3.8 / nx)^2
mass_true = sum(k_true) * dA
mass_rec  = sum(k_recon) * dA
peak_true = maximum(k_true)
peak_rec  = maximum(k_recon)
rms_rel   = sqrt(mean(resid .^ 2)) / peak_true
mae_rel   = mean(abs.(resid)) / peak_true
ratio     = mass_rec / mass_true
n_leaves_ = n_leaves(mf.tree)

println("\n=== 对比: NIE 真值 vs quad-tree 重建 (P1 线性叶) ===")
println("  总质量 ratio: ", round(100 * ratio; digits = 2), "%  ",
        "(真 ", round(mass_true; digits = 4), " / 重建 ",
        round(mass_rec; digits = 4), ")")
println("  峰值 κ:       真 ", round(peak_true; digits = 4),
        " / 重建 ", round(peak_rec; digits = 4),
        "  (", round(100 * peak_rec / peak_true; digits = 1), "%)")
println("  MAE 相对峰值: ", round(100 * mae_rel; digits = 2), "%")
println("  RMS 相对峰值: ", round(100 * rms_rel; digits = 2), "%")
println("  自适应叶子数: ", n_leaves_, "  (均匀 80×80 = ", nx * nx, ")")

# ── 断言 (固化基线) ──────────────────────────────────────────
@assert ratio > 0.90   "总质量恢复应 >90% (实测 >99%)"
@assert peak_rec / peak_true > 0.80   "峰值恢复应 >80% (实测 >85%)"
@assert rms_rel < 0.15  "相对 RMS 应 <15% (P1 实测 ~4%)"
@assert n_leaves_ < 1000  "自适应叶子应远少于均匀网格"

println("\n=== 全部通过 ===")
