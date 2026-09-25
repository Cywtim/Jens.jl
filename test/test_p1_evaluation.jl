# ═══════════════════════════════════════════════════════════════
#  P1 (linear-leaf) upgrade evaluation vs constant-leaf baseline
#
#  Same NIE test as test_quadtree_deflection_vs_nie.jl, three
#  resolutions.  Reports:
#    * leaf count
#    * κ reconstruction quality   (RMS, peak via mass ratio)
#    * P1-fit quality             (mean post-fit resid => how linear the field is)
#    * deflection quality vs NIE  (scale, median/p90 relative resid)
#    * edge bias                  (leaves at |x|>1.5: P1-eval vs NIE κ)
# ═══════════════════════════════════════════════════════════════
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: n_leaves, leaf_centers
import Jens.LensModel: NIEkappa

const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6

function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys; b=b, s=s, q=q, varphi=varphi)
    return (fxx .+ fyy) ./ 2
end

function run_recon(ml, nxs, tau, part; iters=12)
    cc = collect(LinRange(-1.8, 1.8, nxs))
    xs = Float64[]; ys = Float64[]
    for y in cc, x in cc
        push!(xs, x); push!(ys, y)
    end
    k_true = nie_kappa(xs, ys)
    mf = MassField(-2.0, 2.0, -2.0, 2.0; max_level=ml, eta=0.5,
                   damp_pow=0.0, min_obs=1)
    rs = collect(k_true)
    for it in 1:iters
        accumulate_residuals!(mf, xs, ys, rs)
        update_field!(mf)
        refine_mass!(mf; tau=tau, part=part)
        # residual must use the FULL P1 model (value+slope), not leaf-centre value:
        rs = k_true .- p1_value(mf, xs, ys)
    end
    return mf, xs, ys, k_true
end

function evaluate(mf, xs, ys, k_true)
    leaves = mf.tree.leaves
    # κ reconstruction: P1-evaluated field (center + slope) vs constant (value)
    k_p1 = p1_value(mf, xs, ys)
    kv = mass_kappa(mf)
    k_cnt = zeros(Float64, length(xs))
    for i in eachindex(xs)
        for (k, lf) in enumerate(leaves)
            if lf.xmin <= xs[i] <= lf.xmax && lf.ymin <= ys[i] <= lf.ymax
                k_cnt[i] = kv[k]; break
            end
        end
    end
    rms_k_p1 = sqrt(mean((k_true .- k_p1).^2)) / maximum(k_true)
    rms_k_cnt = sqrt(mean((k_true .- k_cnt).^2)) / maximum(k_true)
    # deflection
    ax_t, ay_t = NIEkappa.LensDerivative(xs, ys; b=b, s=s, q=q, varphi=varphi)
    ax_r, ay_r = quadtree_deflection(mf, xs, ys)   # P1-aware (nsub=2 default)
    mag_t = sqrt.(ax_t .^ 2 .+ ay_t .^ 2)
    scale = sum(ax_t .* ax_r .+ ay_t .* ay_r) / sum(ax_r .^ 2 .+ ay_r .^ 2)
    rel = sqrt.((ax_t .- scale .* ax_r).^2 .+ (ay_t .- scale .* ay_r).^2) ./
          (mag_t .+ 1e-6)
    # edge bias: leaves with x-outside ±1.5, compare P1-eval at cell corners vs NIE
    edge_max = 0.0; edge_n = 0; edge_sum = 0.0
    for lf in leaves
        (lf.xmin < -1.5 || lf.xmax > 1.5) || continue
        for (px, py) in ((lf.xmin, (lf.ymin + lf.ymax)/2),
                         ((lf.xmin + lf.xmax)/2, lf.ymin),
                         (lf.xmax, (lf.ymin + lf.ymax)/2),
                         ((lf.xmin + lf.xmax)/2, lf.ymax))
            khat = lf.value + lf.gx * (px - (lf.xmin + lf.xmax)/2) +
                              lf.gy * (py - (lf.ymin + lf.ymax)/2)
            kt = nie_kappa([px], [py])[1]
            d = abs(khat - kt)
            edge_max = max(edge_max, d); edge_sum += d; edge_n += 1
        end
    end
    edge_mean = edge_n > 0 ? edge_sum / edge_n : NaN
    return (leaves=n_leaves(mf.tree), rms_k_p1=rms_k_p1, rms_k_cnt=rms_k_cnt,
            scale=scale, med=median(rel), p90=quantile(rel, 0.9),
            edge_mean=edge_mean, edge_max=edge_max)
end

println("=== P1 (linear-leaf) evaluation on NIE test ===")
for (label, ml, nxs, tau, part, iters) in [
    ("coarse", 8, 100, 0.005, 0.4, 12),
    ("fine", 10, 160, 0.002, 0.4, 16),
    ("ultra", 12, 240, 0.001, 0.4, 20),
]
    mf, xs, ys, kt = run_recon(ml, nxs, tau, part; iters=iters)
    r = evaluate(mf, xs, ys, kt)
    println(rpad("$label", 8),
            " leaves=", rpad(r.leaves, 6),
            " κ_RMS[P1]=", rpad(round(r.rms_k_p1*100, digits=2), 7), "%",
            " κ_RMS[const]=", rpad(round(r.rms_k_cnt*100, digits=2), 7), "%",
            " α_med=", rpad(round(r.med*100, digits=1), 6), "%",
            " α_p90=", rpad(round(r.p90*100, digits=1), 6), "%",
            " scale=", round(r.scale, digits=3),
            " edge_bias_mean=", round(r.edge_mean, digits=4),
            " edge_bias_max=", round(r.edge_max, digits=4))
end
