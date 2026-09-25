# Domain-enlargement test: does α_y error on x∈[-2,-1] shrink with larger ext?
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

function run(ext, po, tau)
    cc = biased_grid(ext, 100, 45)
    xs = Float64[]; ys = Float64[]
    for y in cc, x in cc
        push!(xs, x); push!(ys, y)
    end
    kt = nie_kappa(xs, ys)
    mf = MassField(-ext, ext, -ext, ext; max_level=10, eta=0.5,
                   damp_pow=0.0, min_obs=1, p_order=po)
    rs = collect(kt)
    for it in 1:10
        accumulate_residuals!(mf, xs, ys, rs)
        update_field!(mf)
        refine_mass!(mf; tau=tau, part=0.4)
        rs .= kt .- field_value(mf, xs, ys)
    end
    return mf
end

function cut_error(mf, ycut, xmin, xmax)
    xc = collect(LinRange(xmin, xmax, 401)); yc = fill(ycut, 401)
    ax_t, ay_t = NIEkappa.LensDerivative(xc, yc; b=b, s=s, q=q, varphi=varphi)
    a, c = quadtree_deflection(mf, xc, yc)
    scale = sum(ax_t .* a .+ ay_t .* c) / sum(a.^2 .+ c.^2)
    ex = ax_t .- scale .* a; ey = ay_t .- scale .* c
    return ex, ey, ax_t, ay_t, scale
end

# evaluate on the SAME fixed strip x∈[-2,-1], y=0.4 for all ext
xrange = (-2.0, -1.0)
println("=== 扩域实验: α_y 相对误差在 x∈[-2,-1] (y=0.4), P2 ===")
println(rpad("ext",6), rpad("leaves",7), rpad("scale",8),
        rpad("α_y med%",9), rpad("α_y max%",9), rpad("α_x med%",10), "α_y abs")
for ext in (3.0, 4.0, 5.0)
    mf = run(ext, 2, 0.0008)
    ex, ey, ax_t, ay_t, scale = cut_error(mf, 0.4, -3.2, 3.2)
    # median over the x∈[-2,-1] slice
    xc = collect(range(-3.2, 3.2; length=401))
    sl = (xc .>= -2.0) .& (xc .<= -1.0)
    rel_y = abs.(ey[sl]) ./ (abs.(ay_t[sl]) .+ 1e-9)
    rel_x = abs.(ex[sl]) ./ (abs.(ax_t[sl]) .+ 1e-9)
    println(rpad(ext,6), rpad(n_leaves(mf.tree),7), rpad(round(scale; digits=3),8),
            rpad(round(100*median(rel_y); digits=1),9),
            rpad(round(100*maximum(rel_y); digits=1),9),
            rpad(round(100*median(rel_x); digits=1),10),
            round(1000*median(abs.(ey[sl])); digits=1), "e-3")
end

# also full-cut table for ext=5.0 to see if left/right became symmetric
mf5 = run(5.0, 2, 0.0008)
ex, ey, ax_t, ay_t, scale = cut_error(mf5, 0.4, -3.2, 3.2)
xc = collect(range(-3.2, 3.2; length=401))
println("\next=5.0 全截线分段 (y=0.4):")
seg = [(-3.2,-2.0), (-2.0,-1.4), (-1.4,-1.0), (-1.0,-0.3), (-0.3,0.3),
       (0.3,1.0), (1.0,1.4), (1.4,2.0), (2.0,3.2)]
println(rpad("x段",12), rpad("med|ay|",9), rpad("rel_y%",9), rpad("rel_x%",9), "|ey|(e-3)")
for (lo, hi) in seg
    m = (xc .>= lo) .& (xc .<= hi)
    rel_y = 100*abs.(ey[m]) ./ (abs.(ay_t[m]) .+ 1e-9)
    rel_x = 100*abs.(ex[m]) ./ (abs.(ax_t[m]) .+ 1e-9)
    println(rpad("[$lo,$hi]",12), rpad(round(1000*median(abs.(ay_t[m])); digits=0),9),
            rpad(round(100*median(rel_y); digits=1),9), rpad(round(100*median(rel_x); digits=1),9),
            round(1000*median(abs.(ey[m])); digits=1))
end
