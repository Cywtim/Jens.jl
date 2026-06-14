module LensLOS

    using Cosmology

    import ..LensBase: AbstractLens, lens_derivative, lens_hessian, lens_potential, lens_check

    export ExternalTidal, WithTidal

    # ═══════════════════════════════════════════════════════════════
    #  ExternalTidal — 2×2 LOS tidal matrix
    #
    #      T_ext = [[1-κ-γ1,-γ2 ], [-γ2, 1-κ+γ1]]
    #
    #  The tidal gravitational effect of line-of-sight. The lens Jacobian:
    #      A_total = T_ext · (I - H_lens)
    #
    #  Schneider (2014) arXiv:1409.0015, 
    #  Fleury et al. (2021) arXiv:2104.08883
    #                            10.1088/1475-7516/2021/08/024
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
    #  WithTidal — wraps ANY AbstractLens with LOS tidal matrix
    #
    #  Generic combinator: works on LensModule, CombinedLens,
    #  LensedPlane, MultiLensedPlane, or even another WithTidal.
    #
    #  USAGE:
    #      lp    = LensedPlane(cl; z_lens=0.3, z_source=1.5, cosmology=cosmo)
    #      tidal = ExternalTidal(κ_ext=0.05, γ1_ext=0.02, γ2_ext=-0.01)
    #      wt    = WithTidal(lp, tidal)
    #
    #      # Also works directly on a lens model:
    #      wt2   = WithTidal(LensModule(SIE), tidal)
    #
    #      bx, by = LB.LensPlane(xg, yg; LensModel=wt, LensKwargs=Dict())
    #
    #  NOTE: lens_derivative is unchanged — LOS contributes to the
    #  Jacobian (magnification) but not to the first-order deflection
    #  in the dominant-lens approximation (Bar-Kana 1996, Fleury 2021).
    # ═══════════════════════════════════════════════════════════════

    struct WithTidal{L<:AbstractLens, T<:ExternalTidal} <: AbstractLens
        lens::L
        tidal::T
    end

    # ═══════════════════════════════════════════════════════════════
    #  lens_* interface — dispatch on WithTidal
    # ═══════════════════════════════════════════════════════════════

    # ── deflection: pass through (no LOS deflection in dominant-lens approx) ──
    function lens_derivative(wt::WithTidal, x, y; kwargs...)
        return lens_derivative(wt.lens, x, y; kwargs...)
    end

    # ── Hessian: apply T_ext to the Jacobian ─────────────────────
    #  A_lens = I - H_lens
    #  A_total = T_ext · A_lens
    #  Returns H_los where A_total = I - H_los  (Hessian convention)
    function lens_hessian(wt::WithTidal, x, y; kwargs...)
        # Step 1: get inner lens Hessian
        fxx, fxy, fyy = lens_hessian(wt.lens, x, y; kwargs...)

        # Step 2: tidal matrix components
        T11, T12, T22 = _tidal_components(wt.tidal)

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
    function lens_potential(wt::WithTidal, x, y; kwargs...)
        psi_main = lens_potential(wt.lens, x, y; kwargs...)
        t = wt.tidal
        psi_los = @. 0.5 * t.κ_ext * (x^2 + y^2) +
                      0.5 * t.γ1_ext * (x^2 - y^2) +
                           t.γ2_ext * (x * y)
        return psi_main .+ psi_los
    end

    # ── validation: delegate to inner lens ──────────────────────
    function lens_check(wt::WithTidal; kwargs...)
        lens_check(wt.lens; kwargs...)
    end

end
