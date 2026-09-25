# ═══════════════════════════════════════════════════════════════
#  QuadTreeFit — image-plane fitting with a quadtree residual field
#
#  Scheme 1b (hybrid): a small analytic base lens (parameterised by θ)
#  is COMPLEMENTED by a quad-tree mass field that absorbs whatever the
#  base model cannot explain (substructure, asymmetry, un-modelled
#  mass).  Per fitness evaluation:
#
#      κ_res(θ) = κ_target − κ_analytic(θ)   →  rebuild quad-tree
#      α_total  = α_analytic(θ) + α_quadtree(Barnes–Hut)
#      image    = render(α_total)
#      logp     = masked_logp(image, data)
#
#  This is the image-plane counterpart of the κ-plane scheme: it needs
#  a *deflection/rendering* path (unlike pure κ fitting) but keeps the
#  parameter vector small (only the analytic base parameters), letting
#  the quadtree carry structural freedom without inflating θ.
#
#  κ_target can be:
#    • synthesised (true lens κ) for validation,
#    • a κ map reconstructed from weak-lensing / convergence data,
#    • the residual κ of a previous joint fit (nested bi-level use).
#
#  ── Part of the quad-tree lens stack ─────────────────────────────
#  This is the IMAGE-PLANE FIT LAYER of the stack.  Bottom mesh:
#  `LensMeshRefine` (QuadLeaf/QuadTree); reconstructed κ/α/ψ/H:
#  `LensMassRecon` (MassField, quadtree_*, QuadTreeLens).  This module
#  assembles them into a `θ -> logp` for MCMC.
#  ──────────────────────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════

module QuadTreeFit

using Statistics
using Jens: JFloat
import Jens.LensBase: AbstractLens, lens_derivative
using Jens.LensMassRecon:
    MassField, accumulate_residuals!, update_field!, refine_mass!,
    field_value, QuadTreeLens, quadtree_bh_tree
using Jens.LensSystem: ForwardModel, render, masked_logp
using Jens.LensGenerator: LensedPlane, LightPlane
using Jens.LightModel: evaluate_source

export quadtree_image_logp, HybridQuadLens

"""
    HybridQuadLens(base, quad::QuadTreeLens)

An `AbstractLens` whose deflection/potential is the SUM of an analytic
base lens and a quadtree field lens.  Used to render
α_total = α_base(θ) + α_quadtree over a single `ForwardModel`.
"""
struct HybridQuadLens{B, Q<:QuadTreeLens} <: AbstractLens
    base::B
    quad::Q
end

lens_derivative(h::HybridQuadLens, x, y; kwargs...) =
    lens_derivative(h.base, x, y; kwargs...) .+ lens_derivative(h.quad, x, y; kwargs...)

"""
    logp = quadtree_image_logp(data, sigma; grid_fn, src,
                               z_lens, z_source, cosmology, mask,
                               α_analytic, κ_analytic_grid, κ_target,
                               lens_base,           # (θ) -> AbstractLens
                               nθ::Int, ...)

Build an image-plane log-posterior `θ -> logp` for scheme 1b:

a small analytic lens parametrised by `θ` is complemented by a
quad-tree field absorbing the residual κ = κ_target − κ_analytic(θ);
the combined lens renders via Barnes–Hut and is compared to `data`
with a Gaussian `masked_logp`.

# Arguments (keyword)
- `data`: observed image (same grid as `grid_fn`) with noise `sigma`
- `grid_fn::Function`: `(pix_n, pix_size) -> GenGrid` (shared grid spec)
- `src`: ExtendedSource / AbstractLight
- `z_lens, z_source, cosmology`: geometry for LensedPlane
- `mask`: pixel mask (from Jens.LensMask), or nothing
- `lens_base::Function`: `θ -> AbstractLens` analytic base
- `κ_analytic::Function`: `(θ, x, y) -> κ` base-model convergence
- `κ_target::Function`: `(x, y) -> κ` the target field the quadtree
  absorbs the residual of (true lens κ, or observed κ map)
- `recon_iters::Int=10`     — refinement passes per evaluation
- `recon_sample_n::Int=80`  — training grid points per side ([-ext,ext]²)
- `ext::Real=3.0`           — reconstruction domain half-width
- `max_level::Int=7`, `tau`, `part`, `roi`, `cap_outside` — refinement
- `bh_theta::Real=0.5`      — Barnes–Hut opening angle
- `nsub::Int=2`             — exact-leaf subdivision (potential/hessian)
"""
function quadtree_image_logp(data, sigma::Real; grid_fn, src,
                             z_lens::Real, z_source::Real, cosmology,
                             mask=nothing,
                             lens_base, κ_analytic, κ_target,
                             recon_iters::Int=10,
                             recon_sample_n::Int=80,
                             ext::Real=3.0,
                             max_level::Int=7, tau::Real=5e-4,
                             part::Real=0.4,
                             roi=nothing, cap_outside::Int=typemax(Int),
                             bh_theta::Real=0.5, nsub::Int=2)

    # reconstruction training grid (fixed per evaluation)
    gx = collect(range(-ext, ext; length=recon_sample_n))
    tx = Float64[]; ty = Float64[]
    for y in gx, x in gx
        push!(tx, x); push!(ty, y)
    end

    function logp(θ)
        # analytic base model at current θ
        base = lens_base(θ)
        # residual field: target κ − base-model κ (on train grid)
        kbase = κ_analytic(θ, tx, ty)
        ktgt  = κ_target(tx, ty)
        r = ktgt .- kbase

        # (re)build quadtree absorbing the residual
        mf = MassField(-ext, ext, -ext, ext;
                       max_level=max_level, eta=0.5, damp_pow=0.0,
                       min_obs=1, p_order=1)
        rs = collect(r)
        for _ in 1:recon_iters
            accumulate_residuals!(mf, tx, ty, rs)
            update_field!(mf)
            refine_mass!(mf; tau=tau, part=part, roi=roi,
                         cap_outside=cap_outside)
            rs .= r .- field_value(mf, tx, ty)
        end

        # combined lens + geometry + render + compare
        qtl = QuadTreeLens(mf; method=:bh, theta=bh_theta)
        combined = HybridQuadLens(base, qtl)
        sys = ForwardModel(
            lens_plane=LensedPlane(combined; z_lens=z_lens, cosmology=cosmology),
            source_plane=LightPlane(src; z=z_source),
            grid=grid_fn())
        return masked_logp(sys, data, sigma, mask)
    end
    return logp
end

end # module QuadTreeFit
