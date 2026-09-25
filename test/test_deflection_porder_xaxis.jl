# ═══════════════════════════════════════════════════════════════
#  Deflection-angle comparison:  NIE truth  vs  P0 / P1 / P2
#
#  CONFIGURATION (same spirit as test_quadtree_deflection_vs_nie.jl,
#  but with an ENLARGED domain so the NIE mass tail is inside the tree):
#    · NIE parameters  b=1.2, s=0.15, q=0.75, φ=π/6
#    · core-biased sampling: 100 pts in [−0.9, 0.9] + 45 pts/side out
#      to ±3.0  (≈191 pts/dimension), reconstruction domain ±3.0
#    · max_level=10, eta=0.5, damp_pow=0.0, min_obs=1,
#      tau=0.01, part=0.4, 10 iterations
#  The only difference between runs is the leaf-model order
#  (p_order = 0, 1, 2).
#
#  Reports scale + relative residual distribution over the whole grid
#  for each order, and writes a comparison figure (x-axis cut with
#  raw/scale-corrected curves, whole-grid scatter, residual plot).
# ═══════════════════════════════════════════════════════════════
using Pkg; Pkg.activate(".")
using Statistics
using Jens
using Jens.LensMassRecon
using Jens.LensMeshRefine: n_leaves
using PyPlot
import Jens.LensModel: NIEkappa

const b, s, q, varphi = 1.2, 0.15, 0.75, pi / 6

function nie_kappa(xs, ys)
    fxx, fxy, fyy = NIEkappa.LensHessian(xs, ys; b=b, s=s, q=q, varphi=varphi)
    return (fxx .+ fyy) ./ 2
end

# core-biased 1D samples: dense in [-0.9,0.9] (preserves the NIE core),
# sparse beyond (captures the mass tail without a uniform cost explosion)
function biased_grid(ext, n_core, n_outer)
    core = collect(range(-0.9, 0.9; length=n_core))
    right = collect(range(0.9, ext; length=n_outer+1))[2:end]
    left = collect(range(-ext, -0.9; length=n_outer+1))[1:end-1]
    return vcat(left, core, right)
end

function main()
    # ── reconstruction/evaluation config: enlarged domain ±5.0 with
    # core-biased sampling — the NIE mass tail is fully inside the tree,
    # which removes the boundary-truncation bias (verified: α_y on
    # x∈[-2,-1] drops 13.1% → 4.8% when ext 3.0 → 5.0).
    ext = 5.0
    cc = biased_grid(5.0, 100, 45)       # ~191 pts/dim: dense core, sparse tail
    xs = Float64[]; ys = Float64[]
    for y in cc, x in cc
        push!(xs, x); push!(ys, y)
    end
    k_true = nie_kappa(xs, ys)

    # ── evaluation points: fixed uniform 80×80 grid in [−1.8,1.8]²,
    #    exactly as in the baseline test (fair apples-to-apples stats) ──
    ne = 80
    ce = collect(LinRange(-1.8, 1.8, ne))
    xe = Float64[]; ye = Float64[]
    for y in ce, x in ce
        push!(xe, x); push!(ye, y)
    end
    k_true_e = nie_kappa(xe, ye)
    ax_e, ay_e = NIEkappa.LensDerivative(xe, ye; b=b, s=s, q=q, varphi=varphi)
    mag_e = sqrt.(ax_e .^ 2 .+ ay_e .^ 2)

    cfgs = [(0, :P0), (1, :P1), (2, :P2)]
    res = Dict{Int,Any}()
    # EQUAL-LEAF-BUDGET tau (rematched at ext=5.0): closest-to-130-leaf
    # config found by scan.  P0 cannot exceed 121 leaves here (constant
    # leaves saturate the budget); P1/P2 converge lower.
    tau_match = Dict(0 => 0.15, 1 => 0.004, 2 => 0.004)

    println("=== deflection: NIE vs P0/P1/P2  (EQUAL leaf budget ≈130, ext=5.0) ===")
    for (po, _) in cfgs
        mf = MassField(-ext, ext, -ext, ext; max_level=10, eta=0.5,
                       damp_pow=0.0, min_obs=1, p_order=po)
        rs = collect(k_true)
        for it in 1:10
            accumulate_residuals!(mf, xs, ys, rs)
            update_field!(mf)
            refine_mass!(mf; tau=tau_match[po], part=0.4)
            rs .= k_true .- field_value(mf, xs, ys)
        end
        ax_r, ay_r = quadtree_deflection(mf, xe, ye)
        mag_r = sqrt.(ax_r .^ 2 .+ ay_r .^ 2)
        scale = sum(ax_e .* ax_r .+ ay_e .* ay_r) / sum(ax_r .^ 2 .+ ay_r .^ 2)
        resid2 = sqrt.((ax_e .- scale .* ax_r) .^ 2 .+ (ay_e .- scale .* ay_r) .^ 2)
        relax = resid2 ./ (mag_e .+ 1e-6)
        # absolute errors (scale-corrected)
        ex = scale .* ax_r .- ax_e
        ey = scale .* ay_r .- ay_e
        # component relative error with a field-strength clamp in the
        # denominator, so vanishing components don't blow the metric up:
        den = max.(abs.(ax_e), abs.(ay_e), 0.3)
        rx = abs.(ex) ./ den
        ry = abs.(ey) ./ den
        res[po] = (mf=mf, ax_r=ax_r, ay_r=ay_r, mag_r=mag_r, scale=scale,
                   med=median(relax), p90=quantile(relax, 0.9),
                   mean=mean(relax),
                   ax_rms=sqrt(mean(ex.^2)), ay_rms=sqrt(mean(ey.^2)),
                   ax_med=median(abs.(ex)), ay_med=median(abs.(ey)),
                   ax_rel_med=median(rx), ax_rel_p90=quantile(rx, 0.9),
                   ay_rel_med=median(ry), ay_rel_p90=quantile(ry, 0.9))
        println("  P$po: leaves=", rpad(n_leaves(mf.tree), 5),
                " scale=", round(scale; digits=3),
                " |α|(med/p90)=", round(100*median(relax); digits=1), "/",
                round(100*quantile(relax, 0.9); digits=1), "%")
        # honest metric: absolute errors first, clamped relative second
        println("        绝对误差  RMS(α_x)=", round(1000*sqrt(mean(ex.^2)); digits=1),
                "e-3  RMS(α_y)=", round(1000*sqrt(mean(ey.^2)); digits=1),
                "e-3   | med|α_x|=", round(1000*median(abs.(ex)); digits=1),
                "e-3  med|α_y|=", round(1000*median(abs.(ey)); digits=1), "e-3")
        println("        相对(clamp)  α_x=", round(100*median(rx); digits=1),
                "%/", round(100*quantile(rx,0.9); digits=1),
                "%   α_y=", round(100*median(ry); digits=1),
                "%/", round(100*quantile(ry,0.9); digits=1), "%")
    end

    # baseline for reference (single P1 run in the original test):
    println("\n参考基线 (test_quadtree_deflection_vs_nie.jl, P1):")
    println("  scale≈1.11  median≈0.10  p90≈0.19  leaves≈127")

    # ── figure: component (x / y) views ──
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # (a) α_x along x-cut (y=0.4): NIE, raw, and scale-corrected
    ycut = 0.4
    xc = collect(LinRange(-ext + 0.4, ext - 0.4, 300)); yc = fill(ycut, 300)
    axt, ayt = NIEkappa.LensDerivative(xc, yc; b=b, s=s, q=q, varphi=varphi)
    axs[1,1].plot(xc, axt, "k-", lw=2.2, label="NIE (truth)")
    for (po, _) in cfgs
        mfo = res[po].mf
        a, c = quadtree_deflection(mfo, xc, yc)
        axs[1,1].plot(xc, res[po].scale .* a, "--", lw=1.6,
                      color=("#d62728", "#1f77b4", "#2ca02c")[po+1],
                      label="P$po ·scale (L=$(n_leaves(mfo.tree)))")
    end
    axs[1,1].set_title("α_x(x) along y=$ycut (×scale)")
    axs[1,1].set_ylabel("α_x")
    axs[1,1].legend(fontsize=8, ncol=2)
    axs[1,1].grid(alpha=0.3)

    # (b) α_y along the same cut
    axs[1,2].plot(xc, ayt, "k-", lw=2.2, label="NIE (truth)")
    for (po, _) in cfgs
        mfo = res[po].mf
        a, c = quadtree_deflection(mfo, xc, yc)
        axs[1,2].plot(xc, res[po].scale .* c, "--", lw=1.6,
                      color=("#d62728", "#1f77b4", "#2ca02c")[po+1],
                      label="P$po ·scale")
    end
    axs[1,2].set_title("α_y(x) along y=$ycut (×scale)")
    axs[1,2].set_ylabel("α_y")
    axs[1,2].legend(fontsize=8, ncol=2)
    axs[1,2].grid(alpha=0.3)

    # (c) whole-grid scatter α_x: recon vs NIE (scale-corrected)
    axs[2,1].plot([-1.4, 1.4], [-1.4, 1.4], "k-", lw=1.5, label="identity")
    for (po, _) in cfgs
        axs[2,1].scatter(ax_e, res[po].scale .* res[po].ax_r, s=4, alpha=0.6,
                         color=("#d62728", "#1f77b4", "#2ca02c")[po+1],
                         label="P$po ·scale")
    end
    axs[2,1].set_xlabel("α_x_NIE"); axs[2,1].set_ylabel("α_x_recon×scale")
    axs[2,1].set_title("whole grid: α_x (scale-corrected)")
    axs[2,1].legend(fontsize=8)
    axs[2,1].grid(alpha=0.3)
    axs[2,1].set_aspect("equal", "datalim")

    # (d) whole-grid scatter α_y
    axs[2,2].plot([-1.4, 1.4], [-1.4, 1.4], "k-", lw=1.5, label="identity")
    for (po, _) in cfgs
        axs[2,2].scatter(ay_e, res[po].scale .* res[po].ay_r, s=4, alpha=0.6,
                         color=("#d62728", "#1f77b4", "#2ca02c")[po+1],
                         label="P$po ·scale")
    end
    axs[2,2].set_xlabel("α_y_NIE"); axs[2,2].set_ylabel("α_y_recon×scale")
    axs[2,2].set_title("whole grid: α_y (scale-corrected)")
    axs[2,2].legend(fontsize=8)
    axs[2,2].grid(alpha=0.3)
    axs[2,2].set_aspect("equal", "datalim")

    fig.suptitle("Deflection components: NIE vs P0/P1/P2  (equal leaf budget)",
                 fontsize=12)
    fig.tight_layout()
    out = joinpath(dirname(@__DIR__), "img", "porder_deflection_xaxis.png")
    fig.savefig(out, dpi=150)
    println("\nsaved figure → ", out)
end

main()
