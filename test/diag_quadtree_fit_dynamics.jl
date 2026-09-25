# Re-examine the quadtree fitting dynamics:
#   - residual decay over iterations at fixed tau
#   - per-level damping freezing deep leaves
#   - P1 slope (gx/gy) drift
#   - leaf count growth
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

EXT = 3.0
nx = 111
c = collect(LinRange(-EXT, EXT, nx))
xs = Float64[]; ys = Float64[]
for y in c, x in c
    push!(xs, x); push!(ys, y)
end
kt = nie_kappa(xs, ys)

println("=== 迭代动力学 (P1, max_level=8, tau=5e-4, eta=0.5, damp_pow=0.0) ===")
mf = MassField(-EXT, EXT, -EXT, EXT; max_level=8, eta=0.5,
               damp_pow=0.0, min_obs=1, p_order=1)
rs = collect(kt)
for it in 1:15
    before = n_leaves(mf.tree)
    accumulate_residuals!(mf, xs, ys, rs)
    update_field!(mf)
    refine_mass!(mf; tau=5e-4, part=0.4)
    rs .= kt .- field_value(mf, xs, ys)
    L = n_leaves(mf.tree)
    rms = sqrt(mean(rs .^ 2))
    maxr = maximum(abs.(rs))
    # slope stats
    gx = [lf.gx for lf in mf.tree.leaves]
    gy = [lf.gy for lf in mf.tree.leaves]
    println("  it=$it  leaves=$L  rms=$(round(rms;sigdigits=3))  max|r|=$(round(maxr;sigdigits=3))  " *
            "med|gx|=$(round(median(abs.(gx));sigdigits=3))  max|gx|=$(round(maximum(abs.(gx));sigdigits=3))  " *
            "med|gy|=$(round(median(abs.(gy));sigdigits=3))")
    L == before && it > 5 && break
end

println("\n=== 阻尼影响: damp_pow=1.0 (默认) vs 0.0 ===")
for dp in (0.0, 1.0)
    mf2 = MassField(-EXT, EXT, -EXT, EXT; max_level=8, eta=0.5,
                    damp_pow=dp, min_obs=1, p_order=1)
    r2 = collect(kt)
    for it in 1:12
        accumulate_residuals!(mf2, xs, ys, r2)
        update_field!(mf2)
        refine_mass!(mf2; tau=5e-4, part=0.4)
        r2 .= kt .- field_value(mf2, xs, ys)
    end
    L2 = n_leaves(mf2.tree)
    rms2 = sqrt(mean(r2 .^ 2))
    # leaves below level 3 vs above
    ls = [lf.level for lf in mf2.tree.leaves]
    println("  damp_pow=$dp  leaves=$L2  rms=$(round(rms2;sigdigits=3))  max_level=$(maximum(ls))")
end

println("\n=== min_obs 影响: 1 vs 3 ===")
for mo in (1, 3)
    mf3 = MassField(-EXT, EXT, -EXT, EXT; max_level=8, eta=0.5,
                    damp_pow=0.0, min_obs=mo, p_order=1)
    r3 = collect(kt)
    for it in 1:12
        accumulate_residuals!(mf3, xs, ys, r3)
        update_field!(mf3)
        refine_mass!(mf3; tau=5e-4, part=0.4)
        r3 .= kt .- field_value(mf3, xs, ys)
    end
    L3 = n_leaves(mf3.tree)
    rms3 = sqrt(mean(r3 .^ 2))
    println("  min_obs=$mo  leaves=$L3  rms=$(round(rms3;sigdigits=3))")
end
