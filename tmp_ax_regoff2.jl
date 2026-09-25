# Quantify the α_y deviation on x∈[-2,-1] along the y=0.4 cut
using Statistics
using Jens
using Jens.LensMassRecon
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

ext = 3.0
cc = biased_grid(3.0, 100, 45)
xs = Float64[]; ys = Float64[]
for y in cc, x in cc
    push!(xs, x); push!(ys, y)
end
kt = nie_kappa(xs, ys)
mf = MassField(-ext, ext, -ext, ext; max_level=10, eta=0.5,
               damp_pow=0.0, min_obs=1, p_order=2)
rs = collect(kt)
for it in 1:10
    accumulate_residuals!(mf, xs, ys, rs)
    update_field!(mf)
    refine_mass!(mf; tau=0.0008, part=0.4)
    rs .= kt .- field_value(mf, xs, ys)
end

# cut along y=0.4, x from -2.6 to 2.6
ycut = 0.4
xc = collect(LinRange(-2.6, 2.6, 521)); yc = fill(ycut, 521)
ax_t, ay_t = NIEkappa.LensDerivative(xc, yc; b=b, s=s, q=q, varphi=varphi)
a, c = quadtree_deflection(mf, xc, yc)
# best scale on the cut
scale = sum(ax_t .* a .+ ay_t .* c) / sum(a.^2 .+ c.^2)
ax_r = scale .* a; ay_r = scale .* c

println("cut y=$ycut, global scale on cut=", round(scale; digits=4))
println("\n按 x 分段:  α_y 绝对误差 与 相对误差")
seg = [(-2.6,-2.0), (-2.0,-1.4), (-1.4,-1.0), (-1.0,-0.3), (-0.3,0.3),
       (0.3,1.0), (1.0,1.4), (1.4,2.0), (2.0,2.6)]
println(rpad("x段",14), rpad("med|ayNIE|",11), rpad("med|ey|",9),
        rpad("med rel%",10), rpad("med|axNIE|",11), rpad("med|ex|",9), "med rel x%")
for (lo, hi) in seg
    m = (xc .>= lo) .& (xc .<= hi)
    ey = ay_t[m] .- ay_r[m]
    ex = ax_t[m] .- ax_r[m]
    rel_y = abs.(ey) ./ (abs.(ay_t[m]) .+ 1e-9)
    rel_x = abs.(ex) ./ (abs.(ax_t[m]) .+ 1e-9)
    println(rpad("[$lo,$hi]",14),
            rpad(round(1000*median(abs.(ay_t[m])); digits=0),11),
            rpad(round(1000*median(abs.(ey)); digits=0),9),
            rpad(round(100*median(rel_y); digits=1),10),
            rpad(round(1000*median(abs.(ax_t[m])); digits=0),11),
            rpad(round(1000*median(abs.(ex)); digits=0),9),
            round(100*median(rel_x); digits=1))
end

# detailed points in x∈[-2,-1]
println("\n详细: x∈[-2,-1], y=$ycut (scale=", round(scale; digits=4), ")")
println(rpad("x",7), rpad("ay_NIE",9), rpad("ay_rec",9), rpad("|ey|",9), "rel_y%")
for x0 in (-2.6, -2.4, -2.2, -2.0, -1.8, -1.6, -1.4, -1.2, -1.0, -0.8, -0.6, -0.4, -0.2, 0.0)
    i = argmin(abs.(xc .- x0))
    ry = 100 * abs(ay_t[i] - ay_r[i]) / abs(ay_t[i])
    println(rpad(x0,7), rpad(round(ay_t[i];digits=4),9), rpad(round(ay_r[i];digits=4),9),
            rpad(round(1000*abs(ay_t[i]-ay_r[i]); digits=1),9), round(ry; digits=1))
end
