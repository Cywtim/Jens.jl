module LensLOS

    using Cosmology

    import ..LensBase: lens_derivative, lens_hessian, lens_mass, lens_check
    import ..LensCosmo: lens_distance_ratio, LensedPlane

    export ExternalTidal, LensedPlaneWithLOS

    # ═══════════════════════════════════════════════════════════════
    #  ExternalTidal — 2×2 LOS tidal matrix
    #
    #      T_ext = | 1-κ-γ1   -γ2  |
    #              |   -γ2    1-κ+γ1|
    #
    #  Encodes the tidal gravitational effect of line-of-sight
    #  large-scale structure on the light bundle. This matrix
    #  multiplies the lens Jacobian:
    #      A_total = T_ext · (I - H_lens)
    #
    #  Ref: Schneider (2014) arXiv:1409.0015, Sec. 5
    #       Fleury et al. (2021) arXiv:2104.08883
    # ═══════════════════════════════════════════════════════════════

    struct ExternalTidal
        κ_ext ::Float64   # external convergence
        γ1_ext::Float64   # external shear (γ1 component)
        γ2_ext::Float64   # external shear (γ2 component)
    end

    # ── Convenience constructor: convergence only ──
    ExternalTidal(κ_ext::Float64) = ExternalTidal(κ_ext, 0.0, 0.0)

    # ── Identity (no LOS effect) ──
    const NO_LOS = ExternalTidal(0.0, 0.0, 0.0)

    # ═══════════════════════════════════════════════════════════════
    #  Helper: extract 2×2 tidal matrix components from ExternalTidal
    # ═══════════════════════════════════════════════════════════════

    function _tidal_components(t::ExternalTidal)
        T11 = 1.0 - t.κ_ext - t.γ1_ext
        T12 =            - t.γ2_ext   # = T21
        T22 = 1.0 - t.κ_ext + t.γ1_ext
        return (T11, T12, T22)
    end

    # ═══════════════════════════════════════════════════════════════
    #  LensedPlaneWithLOS — wraps a LensedPlane with LOS tidal matrix
    #
    #  Purely additive: wrapping a LensedPlane rather than modifying
    #  it. All existing code continues to work unchanged.
    #
    #  USAGE:
    #      cosmo = Cosmology.FlatLCDM(0.7, 0.3)
    #      cl    = CombinedLens(SIE=>(b=0.8, e=0.3, ...))
    #      lp    = LensedPlane(cl; z_lens=0.3, z_source=1.5, cosmology=cosmo)
    #      tidal = ExternalTidal(κ_ext=0.05, γ1_ext=0.02, γ2_ext=-0.01)
    #      lp_los = LensedPlaneWithLOS(lp, tidal)
    #
    #      bx, by = LB.LensPlane(xg, yg; LensModel=lp_los, LensKwargs=Dict())
    # ─────────────────────────────────────────────────────────────
    #  NOTE: lens_derivative is unchanged — LOS contributes to the
    #  Jacobian (magnification) but not to the first-order deflection
    #  in the dominant-lens approximation (Bar-Kana 1996, Fleury 2021).
    # ─────────────────────────────────────────────────────────────
    # ═══════════════════════════════════════════════════════════════

    struct LensedPlaneWithLOS{L, C, T<:ExternalTidal}
        plane::LensedPlane{L, C}
        tidal::T
    end

    # ── Keyword constructor ──────────────────────────────────────
    function LensedPlaneWithLOS(lp::LensedPlane, tidal::ExternalTidal)
        return LensedPlaneWithLOS{typeof(lp.lens),
                                  typeof(lp.cosmology),
                                  typeof(tidal)}(lp, tidal)
    end

    # ═══════════════════════════════════════════════════════════════
    #  lens_* interface — dispatches only on LensedPlaneWithLOS type
    # ═══════════════════════════════════════════════════════════════

    # ── deflection: pass through (no LOS deflection in dominant-lens approx) ──
    function lens_derivative(lp::LensedPlaneWithLOS, x, y; kwargs...)
        return lens_derivative(lp.plane, x, y; kwargs...)
    end

    # ── Hessian: apply T_ext to the Jacobian ─────────────────────
    #  A_lens = I - H_lens
    #  A_total = T_ext · A_lens
    #  Returns H_los where A_total = I - H_los  (Hessian convention)
    function lens_hessian(lp::LensedPlaneWithLOS, x, y; kwargs...)
        # Step 1: get main lens Hessian (already includes D_ls/D_s scaling)
        fxx, fxy, fyy = lens_hessian(lp.plane, x, y; kwargs...)

        # Step 2: tidal matrix components
        T11, T12, T22 = _tidal_components(lp.tidal)

        # Step 3: A_lens = I - H_lens
        A11_lens = 1 .- fxx
        A12_lens =    .- fxy       # = A21 (H is symmetric → A is symmetric)
        A22_lens = 1 .- fyy

        # Step 4: A_total = T_ext · A_lens
        A11_total = T11 .* A11_lens .+ T12 .* A12_lens
        A12_total = T11 .* A12_lens .+ T12 .* A22_lens
        # A21 = T12·A11 + T22·A12 = A12_total (since T is symmetric)
        A22_total = T12 .* A12_lens .+ T22 .* A22_lens

        # Step 5: convert back to Hessian convention
        #  A = I - H  →  fxx = 1 - A11,  fxy = -A12,  fyy = 1 - A22
        return (1 .- A11_total), (-A12_total), (1 .- A22_total)
    end

    # ── lensing potential: add LOS quadrupole term ───────────────
    #  ψ_los(θ) = ½κ_ext·|θ|² + ½γ1_ext·(θ_x² - θ_y²) + γ2_ext·θ_x·θ_y
    function lens_mass(lp::LensedPlaneWithLOS, x, y; kwargs...)
        psi_main = lens_mass(lp.plane, x, y; kwargs...)
        t = lp.tidal
        psi_los = @. 0.5 * t.κ_ext * (x^2 + y^2) +
                      0.5 * t.γ1_ext * (x^2 - y^2) +
                           t.γ2_ext * (x * y)
        return psi_main .+ psi_los
    end

    # ── validation: delegate to inner plane ──────────────────────
    function lens_check(lp::LensedPlaneWithLOS; kwargs...)
        lens_check(lp.plane; kwargs...)
    end

end
