module LensLOS

    using Cosmology

    import ..LensBase: AbstractLens, lens_derivative, lens_hessian, lens_potential, lens_check

    export ExternalTidal, WithTidal

    # ═══════════════════════════════════════════════════════════════
    #  ExternalTidal — 2×2 LOS tidal matrix
    #
    #      T_ext = [[1-kappa-gamma1, -gamma2], [-gamma2, 1-kappa+gamma1]]
    #
    #  The tidal gravitational effect of line-of-sight. The lens Jacobian:
    #      A_total = T_ext · (I - H_lens)
    #
    #  Schneider (2014) arXiv:1409.0015, 
    #  Fleury et al. (2021) arXiv:2104.08883
    #                            10.1088/1475-7516/2021/08/024
    # ═══════════════════════════════════════════════════════════════

    struct ExternalTidal{T<:Real}
        kappa_ext ::T   # external convergence
        gamma1_ext::T   # external shear (gamma1 component)
        gamma2_ext::T   # external shear (gamma2 component)
    end

    # ── Convenience constructor: convergence only ──
    ExternalTidal(kappa_ext::Real) = ExternalTidal(kappa_ext, zero(kappa_ext), zero(kappa_ext))

    # ── Keyword constructor ──
    function ExternalTidal(; kappa_ext::Real=0.0,
                           gamma1_ext::Real=0.0,
                           gamma2_ext::Real=0.0)
        T = promote_type(typeof(kappa_ext), typeof(gamma1_ext), typeof(gamma2_ext))
        return ExternalTidal{T}(convert(T, kappa_ext),
                                convert(T, gamma1_ext),
                                convert(T, gamma2_ext))
    end

    # ── Identity (no LOS effect) ──
    const NO_LOS = ExternalTidal(0.0, 0.0, 0.0)

    # ═══════════════════════════════════════════════════════════════
    #  Helper: extract 2×2 tidal matrix components from ExternalTidal
    # ═══════════════════════════════════════════════════════════════

    function _tidal_components(t::ExternalTidal)
        one_t = one(t.kappa_ext)
        T11 = one_t - t.kappa_ext - t.gamma1_ext
        T12 =            - t.gamma2_ext   # = T21
        T22 = one_t - t.kappa_ext + t.gamma1_ext
        return (T11, T12, T22)
    end

    # ═══════════════════════════════════════════════════════════════
    #  WithTidal — wraps ANY AbstractLens with LOS tidal matrix
    #
    #  Generic combinator: works on LensModule, CombinedLens,
    #  LensedPlane, MultiLensedPlane, or even another WithTidal.
    #
    #  USAGE:
    #      lp    = LensedPlane(cl; z_lens=0.3, cosmology=cosmo)
    #      tidal = ExternalTidal(kappa_ext=0.05, gamma1_ext=0.02, gamma2_ext=-0.01)
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
    function lens_derivative(wt::WithTidal, x, y; z_source=nothing, kwargs...)
        return lens_derivative(wt.lens, x, y; z_source=z_source, kwargs...)
    end

    # ── Hessian: apply T_ext to the Jacobian ─────────────────────
    #  A_lens = I - H_lens
    #  A_total = T_ext · A_lens
    #  Returns H_los where A_total = I - H_los  (Hessian convention)
    function lens_hessian(wt::WithTidal, x, y; z_source=nothing, kwargs...)
        # Step 1: get inner lens Hessian
        fxx, fxy, fyy = lens_hessian(wt.lens, x, y; z_source=z_source, kwargs...)

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
    #  psi_los(theta) = 0.5·kappa_ext·|theta|² + 0.5·gamma1_ext·(theta_x² - theta_y²) + gamma2_ext·theta_x·theta_y
    function lens_potential(wt::WithTidal, x, y; z_source=nothing, kwargs...)
        psi_main = lens_potential(wt.lens, x, y; z_source=z_source, kwargs...)
        t = wt.tidal
        half_t = one(t.kappa_ext) / 2
        psi_los = @. half_t * t.kappa_ext * (x^2 + y^2) +
                     half_t * t.gamma1_ext * (x^2 - y^2) +
                         t.gamma2_ext * (x * y)
        return psi_main .+ psi_los
    end

    # ── validation: delegate to inner lens ──────────────────────
    function lens_check(wt::WithTidal; z_source=nothing, kwargs...)
        lens_check(wt.lens; kwargs...)
    end

end
