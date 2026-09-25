# Where do runaway P1 slopes live, and do they corrupt the field?
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: n_leaves
import Jens.LensModel: NIEkappa

const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6
function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys; b=b, s=s, q=q, varphi=varphi)
    return (fxx .+ fyy) ./ 2
end

EXT = 3.0
nx = 151
c = collect(LinRange(-EXT, EXT, nx))
xs = Float64[]; ys = Float64[]
for y in c, x in c
    push!(xs, x); push!(ys, y)
end
kt = nie_kappa(xs, ys)

mf = MassField(-EXT, EXT, -EXT, EXT; max_level=8, eta=0.5,
               damp_pow=0.0, min_obs=1, p_order=1)
rs = collect(kt)
for it in 1:25
    accumulate_residuals!(mf, xs, ys, rs)
    update_field!(mf)
    refine_mass!(mf; tau=5e-4, part=0.4)
    rs .= kt .- field_value(mf, xs, ys)
end
L = n_leaves(mf.tree)
println("leaves=$L  final rms=$(round(sqrt(mean(rs.^2));sigdigits=3))")

# classify leaves by |slope|
big = [lf for lf in mf.tree.leaves if max(abs(lf.gx), abs(lf.gy)) > 3]
println("leaves with max|gx,gy|>3: ", length(big), "  (of $L)")
for lf in big[1:min(end,8)]
    println("  leaf (x∈[$(round(lf.xmin;sigdigits=3)),$(round(lf.xmax;sigdigits=3))], " *
            "y∈[$(round(lf.ymin;sigdigits=3)),$(round(lf.ymax;sigdigits=3))]) " *
            "level=$(lf.level)  value=$(round(lf.value;sigdigits=3)) " *
            "gx=$(round(lf.gx;sigdigits=3)) gy=$(round(lf.gy;sigdigits=3))")
end

# does the reconstructed κ still match truth at those leaves?
print("\nBig-slope leaf centres vs true κ:\n")
for lf in big[1:min(end,5)]
    cx = (lf.xmin+lf.xmax)/2; cy = (lf.ymin+lf.ymax)/2
    ktrue = nie_kappa([cx], [cy])[1]
    krec  = field_value(mf, [cx], [cy])[1]
    println("  centre($(round(cx;sigdigits=3)),$(round(cy;sigdigits=3))): " *
            "κ_true=$(round(ktrue;sigdigits=3))  κ_rec=$(round(krec;sigdigits=3))  diff=$(round(krec-ktrue;sigdigits=3))")
end
