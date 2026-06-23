#!/usr/bin/env julia
using Cosmology
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: SIS, Shear, NFW
using Jens.LensGenerator: LensedPlane
using Jens.LensSolver: solve_images, batch_solve_images

cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
lens_mass = CombinedLens(
    SIS   => (theta_E=0.8, xcentre=0.0, ycentre=0.0),
    Shear => (gamma1=0.05, gamma2=-0.02, xcentre=0.0, ycentre=0.0),
)
lp = LensedPlane(lens_mass; z_lens=0.3, cosmology=cosmo)
z_src = 1.5

# Test a few source positions
test_betas = [(0.1, 0.05), (0.05, -0.03), (-0.08, 0.02), (0.0, 0.15)]

for (bx, by) in test_betas
    println("\nbeta = ($bx, $by)")
    
    # Old solver
    ref = solve_images(lp, bx, by; z_source=z_src)
    println("  nlsolve:   $(length(ref)) image(s)")
    for (tx, ty, mu) in ref
        println("    ($(round(tx,digits=5)), $(round(ty,digits=5)))  mu=$(round(mu,digits=3))")
    end
    
    # Batch solver
    bat = batch_solve_images(lp, bx, by; z_source=z_src)
    println("  batch:     $(length(bat)) image(s)")
    for (tx, ty, mu) in bat
        println("    ($(round(tx,digits=5)), $(round(ty,digits=5)))  mu=$(round(mu,digits=3))")
    end
end

# Test multi-source batch
bx_vec = [0.1, 0.05, -0.08, 0.0]
by_vec = [0.05, -0.03, 0.02, 0.15]
println("\n─── Multi-source batch ───")
@time all_bat = batch_solve_images(lp, bx_vec, by_vec; z_source=z_src)
@time all_ref = [solve_images(lp, bx_vec[i], by_vec[i]; z_source=z_src) for i in 1:4]
for i in 1:4
    println("  src[$i]: ref=$(length(all_ref[i])), bat=$(length(all_bat[i]))")
end
println("Done.")