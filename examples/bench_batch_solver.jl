#!/usr/bin/env julia
using Cosmology, Statistics, Random
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS, Shear, NFW
using Jens.LensGenerator: LensedPlane
using Jens.LensSolver: solve_images, batch_solve_images

Random.seed!(42)

cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
z_src = 1.5

# ── 3 lens models for testing ──
models = [
    ("SIS+Shear", LensedPlane(CombinedLens(
        SIS   => (theta_E=0.8, xcentre=0.0, ycentre=0.0),
        Shear => (gamma1=0.05, gamma2=-0.02, xcentre=0.0, ycentre=0.0),
    ); z_lens=0.3, cosmology=cosmo)),
    ("NFW+Shear", LensedPlane(CombinedLens(
        NFW   => (Rs=5.0, alpha_Rs=1.0, xcentre=0.0, ycentre=0.0),
        Shear => (gamma1=0.05, gamma2=-0.02, xcentre=0.0, ycentre=0.0),
    ); z_lens=0.3, cosmology=cosmo)),
    ("SIS only",  LensedPlane(CombinedLens(
        SIS   => (theta_E=1.0, xcentre=0.0, ycentre=0.0),
    ); z_lens=0.3, cosmology=cosmo)),
]

for (name, lp) in models
    println("\n════════════ $(name) ════════════")
    
    # ── Accuracy: 200 random source positions ──
    n_test = 200
    pos_err = Float64[]
    mu_err   = Float64[]
    missed   = 0
    extra    = 0
    count_match = 0
    
    for _ in 1:n_test
        bx = 0.3 * randn()
        by = 0.3 * randn()
        
        ref = solve_images(lp, bx, by; z_source=z_src, search_radius=2.5)
        bat = batch_solve_images(lp, bx, by; z_source=z_src, search_radius=2.5)
        
        if length(ref) == length(bat)
            count_match += 1
        end
        missed += max(0, length(ref) - length(bat))
        extra  += max(0, length(bat) - length(ref))
        
        # Match images: for each ref, find nearest bat
        for (rx, ry, rmu) in ref
            best_d = Inf
            best_mu = NaN
            for (bx2, by2, bmu) in bat
                d = sqrt((rx-bx2)^2 + (ry-by2)^2)
                if d < best_d
                    best_d = d
                    best_mu = bmu
                end
            end
            if best_d < 0.01  # matched within 0.01 arcsec
                push!(pos_err, best_d)
                if isfinite(rmu) && isfinite(best_mu)
                    push!(mu_err, abs(rmu - best_mu) / max(abs(rmu), 1.0))
                end
            end
        end
    end
    
    println("  $(n_test) sources: matched=$(count_match)/$(n_test), missed=$(missed), extra=$(extra)")
    println("  position error [arcsec]:  mean=$(round(mean(pos_err)*1e6, digits=2))e-6  max=$(round(maximum(pos_err)*1e6, digits=2))e-6")
    println("  magnification error [rel]: mean=$(round(mean(mu_err), digits=6))  max=$(round(maximum(mu_err), digits=6))")
    
    # ── Timing ──
    bx_test = 0.1 * randn(n_test)
    by_test = 0.1 * randn(n_test)
    
    # Warmup
    solve_images(lp, 0.05, -0.03; z_source=z_src)
    batch_solve_images(lp, 0.05, -0.03; z_source=z_src)
    
    t_ref = @elapsed for i in 1:n_test
        solve_images(lp, bx_test[i], by_test[i]; z_source=z_src)
    end
    t_bat = @elapsed for i in 1:n_test
        batch_solve_images(lp, bx_test[i], by_test[i]; z_source=z_src)
    end
    
    println("  Timing ($n_test sources, serial):")
    println("    nlsolve: $(round(t_ref, digits=3))s  ($(round(t_ref/n_test*1000, digits=2)) ms/src)")
    println("    batch:   $(round(t_bat, digits=3))s  ($(round(t_bat/n_test*1000, digits=2)) ms/src)")
    println("    speedup: $(round(t_ref/t_bat, digits=1))×")
    
    # ── Multi-source batch timing ──
    t_bat_vec = @elapsed batch_solve_images(lp, bx_test, by_test; z_source=z_src)
    println("  Batch all $n_test at once: $(round(t_bat_vec, digits=3))s ($(round(t_bat_vec/n_test*1000, digits=3)) ms/src)")
    println("    vs serial nlsolve: $(round(t_ref/t_bat_vec, digits=0))×")
end
println("\nDone.")