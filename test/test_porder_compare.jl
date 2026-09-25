# ═══════════════════════════════════════════════════════════════
#  P0 / P1 / P2 — dedicated three-order comparison on the NIE test
#
#  Same NIE field, two resolutions, each run under p_order ∈ {0,1,2}.
#  Reports (ASCII table) and writes a 2×2 comparison figure:
#    · κ reconstruction RMS (relative to peak)
#    · adaptive leaf count
#    · deflection median relative residual vs NIE
#    · peak recovery (κ at true NIE peak, fraction)
#  ───────────────────────────────────────────────────────────────
#  All three orders share the identical SOLVE→ESTIMATE→MARK→REFINE
#  loop; only the per-leaf polynomial model changes (p_order).
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

function run_recon(ml, nxs, tau, part, po; iters)
    cc = collect(LinRange(-1.8, 1.8, nxs))
    xs = Float64[]; ys = Float64[]
    for y in cc, x in cc
        push!(xs, x); push!(ys, y)
    end
    k_true = nie_kappa(xs, ys)
    mf = MassField(-2.0, 2.0, -2.0, 2.0; max_level=ml, eta=0.5,
                   damp_pow=0.0, min_obs=1, p_order=po)
    rs = collect(k_true)
    for it in 1:iters
        accumulate_residuals!(mf, xs, ys, rs)
        update_field!(mf)
        refine_mass!(mf; tau=tau, part=part)
        rs .= k_true .- field_value(mf, xs, ys)   # full-model residual
    end
    return mf, xs, ys, k_true
end

function evaluate(mf, xs, ys, k_true, po)
    kv = field_value(mf, xs, ys)
    rms = sqrt(mean((k_true .- kv).^2)) / maximum(k_true)
    # peak recovery: reconstructed κ at the true NIE peak (centre)
    k_peak = field_value(mf, [0.0], [0.0])[1]
    peak_frac = k_peak / maximum(k_true)
    # deflection vs NIE
    ax_t, ay_t = NIEkappa.LensDerivative(xs, ys; b=b, s=s, q=q, varphi=varphi)
    ax_r, ay_r = quadtree_deflection(mf, xs, ys)
    mag_t = sqrt.(ax_t .^ 2 .+ ay_t .^ 2)
    scale = sum(ax_t .* ax_r .+ ay_t .* ay_r) / sum(ax_r .^ 2 .+ ay_r .^ 2)
    rel = sqrt.((ax_t .- scale .* ax_r).^2 .+ (ay_t .- scale .* ay_r).^2) ./
          (mag_t .+ 1e-6)
    return (leaves=n_leaves(mf.tree), rms=rms, med=median(rel),
            p90=quantile(rel, 0.9), scale=scale, peak=peak_frac)
end

function main()
    println("=== P0 / P1 / P2 comparison on NIE (κ, α) ===")
    cfgs = [("coarse", 8, 100, 0.005, 0.4, 12),
            ("fine",  10, 160, 0.002, 0.4, 16)]
    rows = Dict{Tuple{String,Int},NamedTuple}()   # (res, po) => eval
    for (label, ml, nxs, tau, part, iters) in cfgs
        for po in (0, 1, 2)
            println("running ", label, " P", po, " ...")
            mf, xs, ys, kt = run_recon(ml, nxs, tau, part, po; iters=iters)
            rows[(label, po)] = evaluate(mf, xs, ys, kt, po)
        end
    end

    # ── ASCII table ──
    println("\n" * repeat("^", 78))
    for (label, _, _, _, _, _) in cfgs
        println("\n[$label]")
        println(rpad("order", 7), rpad("leaves", 8),
                rpad("κ_RMS%", 9), rpad("α_med%", 9), rpad("α_p90%", 9),
                rpad("scale", 8), "peak_ratio")
        for po in (0, 1, 2)
            r = rows[(label, po)]
            println(rpad("P$po", 7), rpad(r.leaves, 8),
                    rpad(round(100*r.rms, digits=2), 9),
                    rpad(round(100*r.med, digits=1), 9),
                    rpad(round(100*r.p90, digits=1), 9),
                    rpad(round(r.scale, digits=3), 8),
                    round(100*r.peak, digits=1))
        end
    end

    # ── figure ──
    fig, axs = plt.subplots(2, 2, figsize=(9, 7))
    colors = Dict(0 => "#d62728", 1 => "#1f77b4", 2 => "#2ca02c")
    rescolors = ["#1f77b4", "#ff7f0e"]

    # κ RMS (κ_RMS for each order, coarse & fine side by side)
    for (i, (label, _, _, _, _, _)) in enumerate(cfgs)
        axs[1,1].bar([po + (i - 0.5)*0.55 for po in (0,1,2)],
                     [100*rows[(label, po)].rms for po in (0,1,2)],
                     width=0.5, color=[colors[po] for po in (0,1,2)],
                     alpha=0.8, label=label)
    end
    axs[1,1].set_title("κ reconstruction RMS (rel. peak, %)")
    axs[1,1].set_xticks([0, 1, 2]); axs[1,1].set_xticklabels(["P0", "P1", "P2"])
    axs[1,1].legend(fontsize=8)

    # leaves
    for (i, (label, _, _, _, _, _)) in enumerate(cfgs)
        axs[1,2].bar([po + (i - 0.5)*0.55 for po in (0,1,2)],
                     [rows[(label, po)].leaves for po in (0,1,2)],
                     width=0.5, color=[colors[po] for po in (0,1,2)],
                     alpha=0.8, label=label)
    end
    axs[1,2].set_yscale("log")
    axs[1,2].set_title("adaptive leaf count")
    axs[1,2].set_xticks([0, 1, 2]); axs[1,2].set_xticklabels(["P0", "P1", "P2"])
    axs[1,2].legend(fontsize=8)

    # α median
    for (i, (label, _, _, _, _, _)) in enumerate(cfgs)
        axs[2,1].plot([0, 1, 2], [100*rows[(label, po)].med for po in (0,1,2)],
                      "o-", color=rescolors[i], label=label)
    end
    axs[2,1].set_title("deflection median rel. resid (%)")
    axs[2,1].set_yscale("log")
    axs[2,1].set_xticks([0, 1, 2]); axs[2,1].set_xticklabels(["P0", "P1", "P2"])
    axs[2,1].legend(fontsize=8)

    # peak recovery
    for (i, (label, _, _, _, _, _)) in enumerate(cfgs)
        axs[2,2].plot([0, 1, 2], [100*rows[(label, po)].peak for po in (0,1,2)],
                      "s-", color=rescolors[i], label=label)
    end
    axs[2,2].axhline(100, color="k", ls=":", lw=0.7)
    axs[2,2].set_title("peak κ recovery (%)")
    axs[2,2].set_xticks([0, 1, 2]); axs[2,2].set_xticklabels(["P0", "P1", "P2"])
    axs[2,2].legend(fontsize=8)
    axs[2,2].set_ylim(60, 105)

    fig.suptitle("Quad-tree leaf-model order: P0 vs P1 vs P2 (NIE test)",
                 fontsize=12)
    fig.tight_layout()
    out = joinpath(dirname(@__DIR__), "porder_compare.png")
    fig.savefig(out, dpi=150)
    println("\nsaved figure → ", out)
end

main()
