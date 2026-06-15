module MGE

    # ═══════════════════════════════════════════════════════════════
    #  Multi-Gaussian Expansion (MGE) — approximate radial profiles
    #  with a sum of circular Gaussian lens components.
    #
    #  References:
    #    Cappellari (2002) MNRAS 333, 400
    #    Shajib  (2019) MNRAS 488, 1387
    # ═══════════════════════════════════════════════════════════════

    using LinearAlgebra
    using Statistics

    import ..LensBase: lens_derivative, lens_hessian, AbstractLens

    export fit_nfw_mge, fit_profile_mge, mge_report
    export circular_gaussian_lens, build_mge_lens, MGECombinedLens

    # ═══════════════════════════════════════════════════════════════
    #  Circular Gaussian lens — analytic (no erfi, no singularities)
    #
    #  kappa(R)      = kappa0 * exp(-R^2 / (2*sigma^2))
    #  alpha(R)      = kappa0 * 2*sigma^2 * (1-exp(-R^2/(2*sigma^2))) / R
    #  kappa_bar(R)  = kappa0 * 2*sigma^2/R^2 * (1-exp(-R^2/(2*sigma^2)))
    #  gamma(R)      = kappa_bar(R) - kappa(R)
    # ═══════════════════════════════════════════════════════════════

    function _circgauss_deflection(x, y, kappa0::Float64, sigma::Float64)
        R2 = @. x^2 + y^2
        # For R near zero, alpha → kappa0 * R (linear, no singularity)
        alpha = similar(R2)
        small = R2 .< 1e-20
        alpha[small] .= kappa0
        invR2 = @. 1.0 / R2
        invR2[small] .= 0.0
        fac = @. kappa0 * 2.0 * sigma^2
        expfac = @. exp(-R2 / (2.0 * sigma^2))
        alpha_nonzero = @. fac * (1.0 - expfac) * invR2
        alpha[.!small] .= alpha_nonzero[.!small]
        ax = @. alpha * x
        ay = @. alpha * y
        return ax, ay
    end

    function _circgauss_hessian(x, y, kappa0::Float64, sigma::Float64)
        R2 = @. x^2 + y^2
        small = R2 .< 1e-20
        safe_R2 = copy(R2)
        safe_R2[small] .= 1.0

        invR2 = @. 1.0 / safe_R2
        invR4 = @. invR2 * invR2

        sigma2 = sigma^2
        expfac = @. exp(-safe_R2 / (2.0 * sigma2))

        kappa = @. kappa0 * expfac

        kappa_bar = @. kappa0 * 2.0 * sigma2 * invR2 * (1.0 - expfac)
        gamma_iso = @. kappa_bar - kappa  # isotropic gamma (|gamma|)

        cos2phi = @. (x^2 - y^2) * invR2
        sin2phi = @. 2.0 * x * y * invR2

        f_xx = @. kappa + gamma_iso * cos2phi
        f_yy = @. kappa - gamma_iso * cos2phi
        f_xy = @. gamma_iso * sin2phi

        # Fix small-R limit: Hessian → kappa0/2 * identity
        f_xx[small] .= kappa0 / 2.0
        f_yy[small] .= kappa0 / 2.0
        f_xy[small] .= 0.0

        return f_xx, f_xy, f_yy
    end

    function _circgauss_potential(x, y, kappa0::Float64, sigma::Float64)
        R2 = @. x^2 + y^2
        sigma2 = sigma^2
        # potential = kappa0 * sigma^2 * (Ei(-R^2/(2*sigma^2)) - log(R^2/(2*sigma^2)) - gamma)
        # Use integral form for numerical stability
        # Phi(R) = integral_0^R alpha(r) dr + constant
        # For convenience, return relative potential (not used for deflection/hessian)
        small = R2 .< 1e-20
        safe_R2 = copy(R2)
        safe_R2[small] .= sigma2

        # Use exponential integral E1 approximation
        arg = @. safe_R2 / (2.0 * sigma2)
        # E1(x) ≈ -log(x) - EulerGamma for small x; use series for stability
        E1_approx = @. -log(arg) - MathConstants.eulergamma + arg - arg^2/4.0 + arg^3/18.0
        pot = @. kappa0 * sigma2 * (E1_approx - log(arg) - MathConstants.eulergamma)
        pot[small] .= 0.0
        return pot
    end

    # ═══════════════════════════════════════════════════════════════
    #  Build a CombinedLens from MGE components
    #
    #  Returns a struct-based CombinedLens that wraps the circular
    #  Gaussian formulas directly. No erfi, no q parameter.
    # ═══════════════════════════════════════════════════════════════

    """
        cl = build_mge_lens(result; q=0.99)

    Build a `ComLens.CombinedLens` from an MGE fit result.

    Uses circular Gaussian analytic formulas (no erfi issues).
    The `q` parameter is ignored for circular Gaussians but
    kept for API compatibility.
    """
    function build_mge_lens(result; q::Float64=0.99)
        # We use the Jens Gaussian model with q=0.99 as a fallback,
        # but the recommended approach is to use CombinedLens with
        # the circular Gaussian formulas.

        # Just build from the result struct
        n = result.n_components
        sigmas = result.sigmas
        kappa0s = result.kappa0s
        return MGECombinedLens(sigmas, kappa0s)
    end

    # ═══════════════════════════════════════════════════════════════
    #  MGECombinedLens: struct-based lens model
    #  Implements AbstractLens interface for use with LensPlane etc.
    # ═══════════════════════════════════════════════════════════════

    struct MGECombinedLens <: AbstractLens
        sigmas::Vector{Float64}
        kappa0s::Vector{Float64}

        function MGECombinedLens(sigmas::Vector{Float64}, kappa0s::Vector{Float64})
            @assert length(sigmas) == length(kappa0s) "sigmas and kappa0s must have same length"
            @assert all(sigmas .> 0) "sigmas must be positive"
            @assert all(kappa0s .>= 0) "kappa0s must be non-negative"
            new(sigmas, kappa0s)
        end
    end

    function circular_gaussian_lens(x, y, lens::MGECombinedLens; what::Symbol=:deflection)
        ax = zeros(size(x))
        ay = zeros(size(y))
        if what == :deflection || what == :all
            for (sigma, kappa0) in zip(lens.sigmas, lens.kappa0s)
                dax, day = _circgauss_deflection(x, y, kappa0, sigma)
                ax .+= dax
                ay .+= day
            end
            what == :deflection && return ax, ay
        end

        if what == :hessian || what == :all
            fxx = zeros(size(x))
            fxy = zeros(size(x))
            fyy = zeros(size(x))
            for (sigma, kappa0) in zip(lens.sigmas, lens.kappa0s)
                hxx, hxy, hyy = _circgauss_hessian(x, y, kappa0, sigma)
                fxx .+= hxx
                fxy .+= hxy
                fyy .+= hyy
            end
            what == :hessian && return fxx, fxy, fyy
        end

        # :all
        return ax, ay, fxx, fxy, fyy
    end

    # ── LensBase dispatch: use with LensPlane ──
    function lens_derivative(lens::MGECombinedLens, x, y; kwargs...)
        return circular_gaussian_lens(x, y, lens; what=:deflection)
    end
    function lens_hessian(lens::MGECombinedLens, x, y; kwargs...)
        return circular_gaussian_lens(x, y, lens; what=:hessian)
    end

    # ═══════════════════════════════════════════════════════════════
    #  NFW projected surface density (kappa)
    # ═══════════════════════════════════════════════════════════════

    """
        _nfw_f(x)

    NFW projected density function f(x) where x = R/Rs.

    Bartelmann (1996), Wright & Brainerd (2000).
    """
    function _nfw_f(x::Float64)
        if x < 1.0
            denom = sqrt(1.0 - x^2)
            return (1.0 - 2.0 / denom * atanh(denom / (1.0 + x))) / (x^2 - 1.0)
        elseif x ≈ 1.0
            return 1.0 / 3.0
        else
            denom = sqrt(x^2 - 1.0)
            return (1.0 - 2.0 / denom * atan(denom / (1.0 + x))) / (x^2 - 1.0)
        end
    end

    function _nfw_g(x::Float64)
        # g(x) used in alpha(R) = 4*rho0*Rs*g(x)/x where x=R/Rs
        if x < 1.0
            x = max(x, 1e-6)
            return log(x / 2.0) + 1.0 / sqrt(1.0 - x^2) * acosh(1.0 / x)
        elseif x ≈ 1.0
            return 1.0 + log(0.5)
        else
            return log(x / 2.0) + 1.0 / sqrt(x^2 - 1.0) * acos(1.0 / x)
        end
    end

    function nfw_kappa(R::AbstractArray, alpha_Rs::Float64, Rs::Float64,
                       rho0::Float64)
        kappa_vals = similar(R)
        for i in eachindex(R)
            x = max(R[i] / Rs, 1e-8)
            kappa_vals[i] = 2.0 * rho0 * Rs * _nfw_f(x)
        end
        return kappa_vals
    end

    function nfw_kappa(R, alpha_Rs::Float64, Rs::Float64)
        rho0 = alpha_Rs / (4.0 * Rs^2 * (1.0 + log(0.5)))
        return nfw_kappa(R, alpha_Rs, Rs, rho0)
    end

    nfw_kappa(R::Float64, alpha_Rs::Float64, Rs::Float64,
              rho0::Float64) = nfw_kappa([R], alpha_Rs, Rs, rho0)[1]

    # ═══════════════════════════════════════════════════════════════
    #  Gaussian basis: kappa(R) = kappa0 * exp(-R^2/(2*sigma^2))
    # ═══════════════════════════════════════════════════════════════

    function _gaussian_basis(sigmas::Vector{Float64}, R::Vector{Float64})
        n_r = length(R)
        n_g = length(sigmas)
        G = zeros(n_r, n_g)
        for i in 1:n_g
            inv2s2 = 1.0 / (2.0 * sigmas[i]^2)
            for j in 1:n_r
                G[j, i] = exp(-R[j]^2 * inv2s2)
            end
        end
        return G
    end

    # ═══════════════════════════════════════════════════════════════
    #  Non-negative least squares by iterative constraint removal
    # ═══════════════════════════════════════════════════════════════

    function _nnls(A::Matrix{Float64}, b::Vector{Float64};
                   max_iter::Int=50, tol::Float64=1e-12)
        n = size(A, 2)
        active = trues(n)

        for _ in 1:max_iter
            x = zeros(n)
            cols = (1:n)[active]
            if isempty(cols)
                return zeros(n)
            end
            x[active] = A[:, active] \ b

            neg_vals = [(i, x[i]) for i in cols if x[i] < -tol]
            if isempty(neg_vals)
                x[x .< 0.0] .= 0.0
                return x
            end

            # Remove only the single most negative
            worst = argmin([v for (_, v) in neg_vals])
            i_remove = neg_vals[worst][1]
            active[i_remove] = false
        end

        x = zeros(n)
        cols = (1:n)[active]
        if !isempty(cols)
            x[active] = A[:, active] \ b
        end
        x[x .< 0.0] .= 0.0
        return x
    end

    # ═══════════════════════════════════════════════════════════════
    #  Public: fit NFW with MGE
    # ═══════════════════════════════════════════════════════════════

    """
        result = fit_nfw_mge(alpha_Rs, Rs; n_gaussians=12, ...)

    Fit NFW profile with `n_gaussians` circular Gaussian components.

    Returns a `NamedTuple` with fields:
        - `gaussians`: Vector of `(sigma, kappa0)` NamedTuples
        - `sigmas`, `kappa0s`: as separate Vectors
        - `R_fit`, `kappa_nfw`, `kappa_mge`: evaluation arrays
        - `rms_error`: RMS relative error (excluding center)
        - `n_components`: active component count

    Use `build_mge_lens(result)` to get a lens model.
    Use `mge_report(result)` to print a summary.
    """
    function fit_nfw_mge(alpha_Rs::Float64, Rs::Float64;
                         n_gaussians::Int=12,
                         sigma_min::Float64=0.05 * Rs,
                         sigma_max::Float64=5.0 * Rs,
                         n_samples::Int=200,
                         r_max::Float64=10.0 * Rs)
        # 1. Radius grid (log-spaced)
        R = exp10.(range(log10(Rs * 1e-3), log10(r_max); length=n_samples))
        rho0 = alpha_Rs / (4.0 * Rs^2 * (1.0 + log(0.5)))

        # 2. Fit alpha(R)*R (effective enclosed mass proxy)
        #    alpha_NFW(R)*R = 4*rho0*Rs*g(R/Rs)
        #    alpha_gauss(R)*R = kappa0 * 2*sigma^2 * (1-exp(-R^2/(2*sigma^2)))
        alphaR_target = zeros(length(R))
        for i in eachindex(R)
            r_rs = R[i] / Rs
            alphaR_target[i] = 4.0 * rho0 * Rs * _nfw_g(r_rs)
        end

        # 3. Gaussian basis for alpha*R
        sigmas = exp10.(range(log10(sigma_min), log10(sigma_max); length=n_gaussians))
        G = zeros(length(R), length(sigmas))
        for i in 1:length(sigmas)
            s2 = sigmas[i]^2
            for j in eachindex(R)
                G[j, i] = 2.0 * s2 * (1.0 - exp(-R[j]^2 / (2.0 * s2)))
            end
        end

        # Weight: emphasize R near Rs (where most lensing happens)
        w = @. exp(-(log.(R / Rs) / 0.7)^2) .+ 0.01
        w_mat = Diagonal(w)
        G_w = w_mat * G
        b_w = w_mat * alphaR_target

        kappa0s = _nnls(G_w, b_w)

        # 4. Remove negligible components
        threshold = maximum(kappa0s) * 1e-12
        keep = kappa0s .> threshold
        sigmas = sigmas[keep]
        kappa0s = kappa0s[keep]

        if sum(keep) < length(keep) && sum(keep) > 0
            G2 = zeros(length(R), length(sigmas))
            for i in 1:length(sigmas)
                s2 = sigmas[i]^2
                for j in eachindex(R)
                    G2[j, i] = 2.0 * s2 * (1.0 - exp(-R[j]^2 / (2.0 * s2)))
                end
            end
            G2_w = w_mat * G2
            b_w = w_mat * alphaR_target
            kappa0s = _nnls(G2_w, b_w)
        end

        # 5. Compute fit quality (on kappa for reporting)
        kappa_target = nfw_kappa(R, alpha_Rs, Rs, rho0)
        kappa_fit = zeros(length(R))
        for (i, sig) in enumerate(sigmas)
            inv2s2 = 1.0 / (2.0 * sig^2)
            for j in eachindex(R)
                kappa_fit[j] += kappa0s[i] * exp(-R[j]^2 * inv2s2)
            end
        end

        rel_err = @. abs(kappa_fit - kappa_target) / max(kappa_target, 1e-12)
        center_mask = R .> Rs * 1e-3
        rms_error = sqrt(mean(rel_err[center_mask].^2))

        gaussians = [(sigma=sigma, kappa0=kappa0) for (sigma, kappa0) in zip(sigmas, kappa0s)]

        return (;
            gaussians,
            sigmas,
            kappa0s,
            R_fit      = R,
            kappa_nfw  = kappa_target,
            kappa_mge  = kappa_fit,
            rms_error,
            n_components = length(sigmas),
        )
    end

    # ═══════════════════════════════════════════════════════════════
    #  Public: fit arbitrary profile with MGE
    # ═══════════════════════════════════════════════════════════════

    function fit_profile_mge(R::Vector{Float64}, profile::Vector{Float64};
                             n_gaussians::Int=15,
                             sigma_min::Float64=nothing,
                             sigma_max::Float64=nothing)
        if sigma_min === nothing
            sigma_min = minimum(R) * 2.0
        end
        if sigma_max === nothing
            sigma_max = maximum(R) * 0.8
        end

        sigmas = exp10.(range(log10(sigma_min), log10(sigma_max); length=n_gaussians))
        G = _gaussian_basis(sigmas, R)
        kappa0s = _nnls(G, profile)

        threshold = maximum(kappa0s) * 1e-12
        keep = kappa0s .> threshold
        sigmas = sigmas[keep]
        kappa0s = kappa0s[keep]

        if sum(keep) < length(keep) && sum(keep) > 0
            G2 = _gaussian_basis(sigmas, R)
            kappa0s = _nnls(G2, profile)
        end

        kappa_fit = zeros(length(R))
        for (i, sig) in enumerate(sigmas)
            inv2s2 = 1.0 / (2.0 * sig^2)
            for j in eachindex(R)
                kappa_fit[j] += kappa0s[i] * exp(-R[j]^2 * inv2s2)
            end
        end

        rel_err = @. abs(kappa_fit - profile) / max(profile, 1e-12)
        rms_error = sqrt(mean(rel_err.^2))

        gaussians = [(sigma=sigma, kappa0=kappa0) for (sigma, kappa0) in zip(sigmas, kappa0s)]

        return (;
            gaussians,
            sigmas,
            kappa0s,
            R_fit      = R,
            kappa_target = profile,
            kappa_mge  = kappa_fit,
            rms_error,
            n_components = length(sigmas),
        )
    end

    # ═══════════════════════════════════════════════════════════════
    #  Report
    # ═══════════════════════════════════════════════════════════════

    function mge_report(result)
        println("MGE Fit Report")
        println("="^60)
        println("  Components:        $(result.n_components)")
        println("  RMS relative error: $(round(result.rms_error, digits=6))")
        println()
        println("  Gaussian components (sigma, kappa0):")
        for (i, g) in enumerate(result.gaussians)
            println("    [$i]  sigma=$(round(g.sigma, digits=6))  kappa0=$(round(g.kappa0, digits=8))")
        end
        println()
        println("  Build lens model:")
        println("    lens = MGE.build_mge_lens(result)")
        println("    # Or use directly:")
        println("    ax, ay = MGE.circular_gaussian_lens(x, y, lens)")
        return nothing
    end

end # module MGE