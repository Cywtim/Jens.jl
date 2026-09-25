# Independent validation of quadtree_potential / quadtree_hessian.
# Oracle for ψ: direct numerical 2D quadrature of ∫∫ κ ln|θ−θ'| d²θ'
# (NOT using the H-formula, so this is an independent check).
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: n_leaves

# ── helper: single constant-κ leaf field ──
function make_single_leaf(; xlo=-0.5, xhi=0.5, ylo=-0.3, yhi=0.3, kappa=1.0)
    mf = MassField(xlo*2, xhi*2, ylo*2, yhi*2; max_level=1, p_order=0)
    lf = mf.tree.leaves[1]
    lf.xmin = xlo; lf.xmax = xhi; lf.ymin = ylo; lf.ymax = yhi
    lf.value = kappa
    return mf
end

# numerical oracle: ψ_num(x,y) = (1/π)∫∫ κ·ln|θ−θ'| dx'dy'
#   with ln|θ−θ'| = ½·ln(u²+v²)  (same ½-norm as the quadtree
#   deflection antiderivative — keeps ∂ψ = α consistent)
function psi_oracle(mf, x, y; N=401)
    lf = mf.tree.leaves[1]
    kx = collect(range(lf.xmin, lf.xmax; length=N))
    ky = collect(range(lf.ymin, lf.ymax; length=N))
    dx = (lf.xmax - lf.xmin) / (N - 1); dy = (lf.ymax - lf.ymin) / (N - 1)
    s = 0.0
    for yy in ky, xx in kx
        r2 = (x - xx)^2 + (y - yy)^2
        r2 < 1e-14 && continue
        s += 0.5 * log(r2)
    end
    return s * dx * dy / pi
end

println("=== 1) P0 单叶 potential 精确性（与独立数值积分对照）===")
mf0 = make_single_leaf()
for (x, y) in [(-1.2, 0.8), (0.9, -1.3), (2.0, 1.5), (0.1, -0.2), (-3.0, -2.0)]
    pc = quadtree_potential(mf0, [x], [y])[1]
    pn = psi_oracle(mf0, x, y)
    println("  ($x,$y):  closed=", round(pc; digits=6),
            "  oracle=", round(pn; digits=6),
            "  rel=", round(abs(pc-pn)/abs(pn); digits=6))
end

println("\n=== 2) ∇ψ 与 α 一致性 (∂x∂ψ vs −α_x? 按本仓约定) ===")
# ψ(θ) = (1/π)∫κ ln|θ−θ'| ; α = ∇ψ ; verify dψ/dx == α_x numerically
h = 1e-6
for (x, y) in [(-1.2, 0.8), (0.9, -1.3), (2.0, 1.5)]
    dp = (quadtree_potential(mf0, [x+h], [y])[1] - quadtree_potential(mf0, [x-h], [y])[1]) / (2h)
    ax, _ = quadtree_deflection(mf0, [x], [y]; nsub=2)
    # quadtree_deflection includes factor /2π per _rect_deflection.
    # Oracle derivative convention: dψ/dx with ψ=(1/π)...
    println("  ($x,$y):  dψ/dx=", round(dp; digits=6), "   α_quadtree=", round(ax[1]; digits=6),
            "  ratio=", round(dp / ax[1]; digits=4))
end

println("\n=== 3) Hessian：quadtree_hessian vs 对 deflection 的解析差分 vs potential 二阶差 ===")
# For a constant-κ leaf, f_xx = ∂α_x/∂x. Compare:
#   (a) quadtree_hessian (its own central diff)
#   (b) independent central diff of quadtree_deflection
xf = [-1.2, 0.9, 2.0]
for x in xf, y in [-0.5]
    hxx, hxy, hyy = quadtree_hessian(mf0, [x], [y]; diff=1e-6)
    # independent: ∂α_x/∂x
    hm = 1e-6
    axm, _ = quadtree_deflection(mf0, [x-hm], [y]; nsub=2)
    axp, _ = quadtree_deflection(mf0, [x+hm], [y]; nsub=2)
    ay_m, _ = quadtree_deflection(mf0, [x], [y-hm]; nsub=2)
    ay_p, _ = quadtree_deflection(mf0, [x], [y+hm]; nsub=2)
    dxx = (axp[1] - axm[1]) / (2hm)
    dyy = (ay_p[1] - ay_m[1]) / (2hm)
    println("  ($x,$y):  hessian f_xx=", round(hxx[1]; digits=6),
            "  独立差分 ∂α_x/∂x=", round(dxx; digits=6),
            "  rel=", round(abs(hxx[1]-dxx)/abs(dxx); digits=5),
            "  | f_yy:", round(hxx[1]; digits=4), "/", round(dyy; digits=4))
end

println("\n=== 4) 一个真实 P1 四叶场冒烟测试（无 oracle，仅自洽）===")
xt = collect(LinRange(-1.0, 1.0, 5)); yt = collect(LinRange(-1.0, 1.0, 5))
xs = Float64[]; ys = Float64[]
for y in yt, x in xt
    push!(xs, x); push!(ys, y)
end
kt = 1.0 .+ 0.5 .* xs .- 0.3 .* ys     # P1 linear true field
mf1 = MassField(-1.2, 1.2, -1.2, 1.2; max_level=2, p_order=1)
rs = collect(kt)
for it in 1:20
    accumulate_residuals!(mf1, xs, ys, rs)
    update_field!(mf1)
    refine_mass!(mf1; tau=0.9, part=0.5)
    rs .= kt .- field_value(mf1, xs, ys)
end
println("  leaves=", n_leaves(mf1.tree))
psi = quadtree_potential(mf1, [0.3], [0.2])[1]
ax, ay = quadtree_deflection(mf1, [0.3], [0.2])
fxx, fxy, fyy = quadtree_hessian(mf1, [0.3], [0.2])
println("  ψ(0.3,0.2)=", round(psi; digits=5),
        "  α=(", round(ax[1]; digits=5), ", ", round(ay[1]; digits=5), ")",
        "  H=(", round(fxx[1]; digits=5), ", ", round(fxy[1]; digits=5), ", ", round(fyy[1]; digits=5), ")")
println("  Poisson 自洽: ∇²ψ≈2κ →  2κ-∇²ψ=",
        round(2*field_value(mf1, [0.3], [0.2])[1] - (fxx[1]+fyy[1]); digits=5))
