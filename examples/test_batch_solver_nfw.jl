#!/usr/bin/env julia
using Cosmology
using Jens
using Jens.LensModel.ComLens: CombinedLens
using Jens.LensModel: NFW, Shear
using Jens.LensGenerator: LensedPlane
using Jens.LensSolver: solve_images, batch_solve_images

cosmo = Cosmology.FlatLCDM(0.7, 0.3, 0.0, 0.0)
lens_mass = CombinedLens(
    NFW   => (Rs=5.0, alpha_Rs=1.0,  xcentre=0.0, ycentre=0.0),
    Shear => (gamma1=0.05, gamma2=-0.02, xcentre=0.0, ycentre=0.0),
)
lp = LensedPlane(lens_mass; z_lens=0.3, cosmology=cosmo)
z_src = 1.5

# Check a central and off-axis source
for (bx, by) in [(0.05, -0.03), (0.15, 0.10)]
    println("\nbeta = ($bx, $by)")
    ref = solve_images(lp, bx, by; z_source=z_src, search_radius=3.0)
    bat = batch_solve_images(lp, bx, by; z_source=z_src, search_radius=3.0)
    println("  nlsolve: $(length(ref))  batch: $(length(bat))")
    for i in 1:max(length(ref), length(bat))
        if i <= length(ref)
            println("    ref: ($(round(ref[i][1],digits=5)), $(round(ref[i][2],digits=5))) mu=$(round(ref[i][3],digits=2))")
        end
        if i <= length(bat)
            println("    bat: ($(round(bat[i][1],digits=5)), $(round(bat[i][2],digits=5))) mu=$(round(bat[i][3],digits=2))")
        end
    end
end
println("\nDone.")