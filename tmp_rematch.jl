# Re-match tau at ext=5.0 for equal leaf budget (~130)
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
function biased_grid(ext, n_core, n_outer)
    core = collect(range(-0.9, 0.9; length=n_core))
    right = collect(range(0.9, ext; length=n_outer+1))[2:end]
    left = collect(range(-ext, -0.9; length=n_outer+1))[1:end-1]
    return vcat(left, core, right)
end
ext = 5.0
cc = biased_grid(ext, 100, 45)
xs = Float64[]; ys = Float64[]
for y in cc, x in cc
    push!(xs, x); push!(ys, y)
end
kt = nie_kappa(xs, ys)
ne = 80
ce = collect(LinRange(-1.8, 1.8, ne))
xe = Float64[]; ye = Float64[]
for y in ce, x in ce
    push!(xe, x); push!(ye, y)
end
ax_e, ay_e = NIEkappa.LensDerivative(xe, ye; b=b, s=s, q=q, varphi=varphi)
mag_e = sqrt.(ax_e.^2 .+ ay_e.^2)

function run(po, tau)
    mf = MassField(-ext, ext, -ext, ext; max_level=10, eta=0.5,
                   damp_pow=0.0, min_obs=1, p_order=po)
    rs = collect(kt)
    for it in 1:10
        accumulate_residuals!(mf, xs, ys, rs)
        update_field!(mf)
        refine_mass!(mf; tau=tau, part=0.4)
        rs .= kt .- field_value(mf, xs, ys)
    end
    ax_r, ay_r = quadtree_deflection(mf, xe, ye)
    scale = sum(ax_e .* ax_r .+ ay_e .* ay_r) / sum(ax_r.^2 .+ ay_r.^2)
    resid2 = sqrt.((ax_e .- scale .* ax_r).^2 .+ (ay_e .- scale .* ay_r).^2)
    rel = resid2 ./ (mag_e .+ 1e-6)
    return (leaves=n_leaves(mf.tree), scale=scale, med=median(rel),
            p90=quantile(rel, 0.9))
end

println("=== ext=5.0 tau rematch (target ~130 leaves) ===")
tau_grid = Dict(
    0 => [0.15, 0.1, 0.07, 0.05, 0.03, 0.02],
    1 => [0.04, 0.02, 0.01, 0.006, 0.004, 0.002],
    2 => [0.004, 0.002, 0.0008, 0.0004, 0.0002, 0.0001],
)
for target in (130,)
    println("\n────── 目标叶数 ≈ $target ──────")
    for po in (0, 1, 2)
        best = nothing
        for tau in tau_grid[po]
            r = run(po, tau)
            d = abs(r.leaves - target)
            if best === nothing || d < best.d
                best = (d=d, tau=tau, r=r)
            end
        end
        r = best.r
        println("P$po: tau=", rpad(best.tau, 9), " leaves=", rpad(r.leaves, 5),
                " scale=", round(r.scale; digits=3),
                " |α|med=", round(100*r.med; digits=1), "%",
                " p90=", round(100*r.p90; digits=1), "%")
    end
end
