# ═══════════════════════════════════════════════════════════════
#  LensShapelet — shapelet-basis source reconstruction
#
#  Models the source-plane brightness as a linear combination
#  of Cartesian shapelets (Refregier 2003, MNRAS 338, 35):
#
#      S(x, y) = Σ_{n1,n2} c_{n1,n2} · φ_{n1,n2}(x, y; β)
#
#  where φ_{n1,n2} are 2D Gaussian-weighted Hermite polynomials.
#
#  Key property: lensing is linear in surface brightness, so the
#  lensed image = same linear combination of lensed basis functions.
#  Coefficients c are solved analytically via regularized least
#  squares at each MCMC step — only lens parameters + β are sampled.
#
#  USAGE:
#    basis = ShapeletBasis(n_max=6, beta=0.15)
#    A_conv = build_design(sys, basis, data, mask; lambda=0.01)
#    lp = shapelet_logp(A_conv, data, mask, sigma)
# ═══════════════════════════════════════════════════════════════

module LensShapelet

    using LinearAlgebra
    using SpecialFunctions: factorial

    using Jens.LensPSF: conv_psf
    using Jens.LensBase: lens_derivative
    using Jens.LensGenerator: LightPlane, MultiLightPlane

    export ShapeletBasis, n_basis
    export evaluate_basis, build_design, build_design!, solve_coeffs
    export shapelet_logp, shapelet_model
    export build_reg_matrix, profile_beta

    # ═══════════════════════════════════════════════════════════════
    #  ShapeletBasis
    # ═══════════════════════════════════════════════════════════════

    """
        ShapeletBasis(n_max::Int, beta::Real)

    Cartesian shapelet basis up to order `n_max` with scale `beta`
    (arcsec on the source plane).  `beta` is stored as Float32 for
    GPU compatibility; Float64 inputs are auto-converted.

    Total number of basis functions:
        N = (n_max + 1)(n_max + 2) / 2

    # Example
        basis = ShapeletBasis(8, 0.12f0)  # 45 basis functions, β=0.12
    """
    struct ShapeletBasis{T<:AbstractFloat}
        n_max::Int
        beta::T
    end

    """
        N = n_basis(basis::ShapeletBasis)

    Total number of shapelet basis functions.
    """
    n_basis(b::ShapeletBasis) = div((b.n_max + 1) * (b.n_max + 2), 2)


    # ═══════════════════════════════════════════════════════════════
    #  Hermite polynomials (physicist's convention, recurrence)
    # ═══════════════════════════════════════════════════════════════

    """
        Hn = hermite_poly(n::Int, x)

    Evaluate the n-th physicist's Hermite polynomial H_n(x) at `x`.
    Uses the recurrence:
        H₀(x) = 1,  H₁(x) = 2x,  H_{n+1} = 2x·H_n - 2n·H_{n-1}
    """
    function hermite_poly(n::Int, x::Real)
        n < 0 && throw(ArgumentError("n must be ≥ 0"))
        if n == 0
            return one(x)
        elseif n == 1
            return 2x
        end
        h_prev = one(x)
        h_curr = 2x
        for k in 2:n
            h_next = 2x * h_curr - 2(k - 1) * h_prev
            h_prev = h_curr
            h_curr = h_next
        end
        return h_curr
    end


    # ═══════════════════════════════════════════════════════════════
    #  Single shapelet basis function
    # ═══════════════════════════════════════════════════════════════

    """
        phi = shapelet_2d(n1, n2, beta, x, y)

    Evaluate the (n1, n2) Cartesian shapelet at position (x, y)
    on the source plane.

        φ_{n1,n2}(x,y; β) = (2^{n1+n2} π n1! n2! β²)^{-1/2}
                           · H_{n1}(x/β) · H_{n2}(y/β)
                           · exp(-(x²+y²) / (2β²))
    """
    function shapelet_2d(n1::Int, n2::Int, beta::Float64, x::Real, y::Real)
        xb = x / beta
        yb = y / beta
        norm_inv = sqrt(2.0^(n1 + n2) * pi * factorial(n1) * factorial(n2) * beta^2)
        return hermite_poly(n1, xb) * hermite_poly(n2, yb) *
               exp(-(xb^2 + yb^2) / 2) / norm_inv
    end

    # ── Vectorised version (x, y are arrays) ──
    function shapelet_2d(n1::Int, n2::Int, beta::Real,
                         x::AbstractArray, y::AbstractArray)
        T = promote_type(typeof(beta), eltype(x), eltype(y))
        xb = x ./ T(beta)
        yb = y ./ T(beta)
        norm_inv = T(1) / sqrt(T(2)^(n1 + n2) * T(pi) *
                    factorial(n1) * factorial(n2) * T(beta)^2)
        # Use recurrence for vectorised Hermite
        hx = _hermite_vec(n1, xb)
        hy = _hermite_vec(n2, yb)
        return @. norm_inv * hx * hy * exp(-(xb^2 + yb^2) / 2)
    end

    # Recurrence for vectorised Hermite
    function _hermite_vec(n::Int, x::AbstractArray)
        n == 0 && return ones(eltype(x), size(x))
        n == 1 && return 2 .* x
        h_prev = ones(eltype(x), size(x))
        h_curr = 2 .* x
        for k in 2:n
            h_next = @. 2 * x * h_curr - 2(k - 1) * h_prev
            h_prev = h_curr
            h_curr = h_next
        end
        return h_curr
    end


    # ═══════════════════════════════════════════════════════════════
    #  Evaluate all basis functions at given positions
    # ═══════════════════════════════════════════════════════════════

    """
        A = evaluate_basis(basis::ShapeletBasis, x, y)

    Evaluate all `N` shapelet basis functions at positions `(x, y)`.

    # Arguments
    - `basis`: ShapeletBasis
    - `x`, `y`: source-plane coordinates (scalars or arrays)

    # Returns
    `(M, N)` matrix where M = length(x) and N = n_basis(basis).
    A[k, n] = φ_n(source_x[k], source_y[k]).
    """
    function evaluate_basis(basis::ShapeletBasis, x, y)
        N = n_basis(basis)
        M = length(x)
        T = promote_type(typeof(basis.beta), eltype(x), eltype(y))
        A = Matrix{T}(undef, M, N)

        col = 1
        for n in 0:basis.n_max
            for n1 in 0:n
                n2 = n - n1
                A[:, col] .= vec(shapelet_2d(n1, n2, basis.beta, x, y))
                col += 1
            end
        end
        return A
    end


    # ═══════════════════════════════════════════════════════════════
    #  Build PSF-convolved design matrix
    # ═══════════════════════════════════════════════════════════════

    """
        build_design!(A_conv, A_flat, sys::ForwardModel,
                      basis::ShapeletBasis, data, mask;
                      xc_src=0.0, yc_src=0.0)

    In-place variant of `build_design`.  Writes the PSF-convolved
    design matrix into the pre-allocated `A_conv` (npix × N) and uses
    `A_flat` (same size) as scratch space.

    Essential for GPU MCMC: avoids allocating fresh `CuArray` at every
    MCMC step, which would otherwise incur 84,000 `cudaMalloc` calls.

    # Returns
    `(A_masked, mask_idx)` where `A_masked` is a **view** into
    `A_conv`, not a copy.  `mask_idx` is reused across calls.
    """
    function build_design!(A_conv, A_flat, sys, basis::ShapeletBasis,
                            data, mask; xc_src::Real=0.0, yc_src::Real=0.0)
        xg = sys.grid.xg
        yg = sys.grid.yg
        nx, ny = size(xg)
        N = n_basis(basis)

        # ── Step 1–2: ray-trace + centre shift ──
        betax, betay = _source_plane(sys, xg, yg)
        if xc_src != 0.0 || yc_src != 0.0
            T = eltype(betax)
            betax = betax .- T(xc_src)
            betay = betay .- T(yc_src)
        end

        # ── Step 3: evaluate basis → scratch ──
        A_tmp = evaluate_basis(basis, vec(betax), vec(betay))
        A_flat .= A_tmp

        # ── Step 4: PSF convolve each column → write into A_conv ──
        psf = sys.psf
        T = eltype(xg)
        pix_scale = T(sys.grid.pix_size)

        for col in 1:N
            img_col = reshape(view(A_flat, :, col), nx, ny)
            if psf !== nothing
                img_col = conv_psf(img_col, psf, pix_scale)
            end
            A_conv[:, col] .= vec(img_col)
        end

        # ── Step 5: apply mask ──
        if mask !== nothing
            mask_flat = vec(mask)
            mask_idx = findall(mask_flat)
            A_masked = view(A_conv, mask_idx, :)
        else
            mask_idx = collect(1:nx * ny)
            A_masked = A_conv
        end

        return A_masked, mask_idx
    end

    """
        A_masked, A_conv_full, mask_idx = build_design(
            sys::ForwardModel, basis::ShapeletBasis,
            data, mask; xc_src=0.0, yc_src=0.0)

    Full pipeline: ray-trace → translate source centre → evaluate
    basis → PSF convolve → apply mask.

    Allocates the design matrix internally.  For MCMC with many
    iterations, prefer `build_design!` with pre-allocated buffers.

    # Returns
    `(A_masked, A_conv_full, mask_idx)` where:
    - `A_masked` is the (n_masked, N) design matrix for fitting
    - `A_conv_full` is the full (n_pixels, N) matrix for rendering
    - `mask_idx` is the linear indices of masked pixels
    """
    function build_design(sys, basis::ShapeletBasis, data, mask;
                           xc_src::Real=0.0, yc_src::Real=0.0)
        xg = sys.grid.xg
        nx, ny = size(xg)
        npix = nx * ny
        N = n_basis(basis)

        A_conv = similar(xg, npix, N)
        A_flat = similar(xg, npix, N)

        A_masked, mask_idx = build_design!(A_conv, A_flat, sys, basis, data, mask;
                                            xc_src=xc_src, yc_src=yc_src)
        return A_masked, A_conv, mask_idx
    end

    """
        betax, betay = _source_plane(sys, xg, yg)

    Ray-trace image-plane positions to the source plane using the
    lens equation β = θ − α(θ).
    """
    function _source_plane(sys, xg, yg)
        zs = _get_z_source(sys)
        ax, ay = lens_derivative(sys.lens_plane, xg, yg;
                                  z_source=zs)
        return xg .- ax, yg .- ay
    end

    function _get_z_source(sys)
        sp = sys.source_plane
        if sp isa LightPlane
            return sp.z
        elseif sp isa MultiLightPlane
            return sp.planes[1][2]
        else
            error("Cannot extract z_source from source_plane of type $(typeof(sp))")
        end
    end


    # ═══════════════════════════════════════════════════════════════
    #  Regularization matrix builders
    # ═══════════════════════════════════════════════════════════════

    """
        Gamma = build_reg_matrix(basis::ShapeletBasis, x_src, y_src,
                                  kind::Symbol=:gradient)

    Build the regularization matrix ΓᵀΓ for gradient or curvature
    regularisation.

    # Regularisation kinds
    - `:gradient`  — ∫|∇S|² dxdy  (first-order smoothness)
    - `:curvature` — ∫|∇²S|² dxdy (second-order smoothness)

    # Arguments
    - `x_src`, `y_src`: source-plane positions (vectors, length M)
    - `basis`: ShapeletBasis

    # Returns
    `(N, N)` matrix where N = n_basis(basis).  Symmetric positive
    semi-definite.  Ready to use: `M_reg = M + lambda * Gamma`.

    # How it works
    For each source-plane pixel, compute the gradient of every basis
    function and accumulate ∫∇φ_i · ∇φ_j into the (i,j) entry of ΓᵀΓ.

    # Performance
    O(M · N²) — one-time cost, not per MCMC step.
    """

    function build_reg_matrix(basis::ShapeletBasis, x_src::AbstractVector,
                               y_src::AbstractVector, kind::Symbol=:gradient)
        N = n_basis(basis)
        M = length(x_src)
        Gamma = zeros(N, N)

        if kind == :gradient
            # ΓᵀΓ_ij = Σ_k (∂φ_i/∂x)_k · (∂φ_j/∂x)_k + (∂φ_i/∂y)_k · (∂φ_j/∂y)_k
            # where k iterates over source-plane pixels.
            #
            # Pre-compute gradient vectors for all basis functions
            # at all source positions, then accumulate outer products.
            #
            # For efficiency: build gradient matrix G of size (2M, N)
            # where rows 1:M are ∂/∂x and rows M+1:2M are ∂/∂y.
            # Then Gamma = G' * G.
            G = Matrix{typeof(basis.beta)}(undef, 2M, N)
            col = 1
            for n in 0:basis.n_max
                for n1 in 0:n
                    n2 = n - n1
                    phi, dphi_dx, dphi_dy = _shapelet_gradient(
                        n1, n2, basis.beta, x_src, y_src)
                    G[1:M, col]     .= dphi_dx
                    G[M+1:2M, col]  .= dphi_dy
                    col += 1
                end
            end
            mul!(Gamma, G', G)   # Gamma = G' * G,  size (N, N)

        elseif kind == :curvature
            # ΓᵀΓ_ij = Σ_k (∂²φ_i/∂x² + ∂²φ_i/∂y²)_k · (∂²φ_j/∂x² + ∂²φ_j/∂y²)_k
            L = Matrix{typeof(basis.beta)}(undef, M, N)
            col = 1
            for n in 0:basis.n_max
                for n1 in 0:n
                    n2 = n - n1
                    _, _, _, laplacian = _shapelet_curvature(
                        n1, n2, basis.beta, x_src, y_src)
                    L[:, col] .= laplacian
                    col += 1
                end
            end
            mul!(Gamma, L', L)

        else
            error("build_reg_matrix: unknown kind=:$kind.  " *
                  "Use :gradient or :curvature.")
        end

        # Normalise by pixel count (so lambda ~ O(1) independent of M)
        Gamma ./= M

        return Symmetric(Gamma)
    end


    # ═══════════════════════════════════════════════════════════════
    #  Shapelet gradient & curvature (analytic derivatives)
    # ═══════════════════════════════════════════════════════════════

    """
        phi, dphi_dx, dphi_dy = _shapelet_gradient(n1, n2, beta, x, y)

    Evaluate the (n1,n2) Cartesian shapelet AND its first derivatives
    at positions (x, y).  Returns vectors (length M).

    Uses the recurrence relation for Hermite derivatives:
        H'_n(x) = 2n · H_{n-1}(x)
    """
    function _shapelet_gradient(n1::Int, n2::Int, beta::Float64,
                                 x::AbstractVector, y::AbstractVector)
        xb = x ./ beta
        yb = y ./ beta
        T = eltype(x)
        ib = T(1) / beta

        # Prefactor
        norm_inv = T(1) / sqrt(T(2)^(n1 + n2) * T(pi) *
                    factorial(n1) * factorial(n2) * beta^2)
        gauss = @. exp(-(xb^2 + yb^2) / 2)

        # Hermite polynomials
        Hx = _hermite_vec(n1, xb)
        Hy = _hermite_vec(n2, yb)

        # phi = norm * H_{n1}(x/β) * H_{n2}(y/β) * exp(-r²/2)
        phi = @. norm_inv * Hx * Hy * gauss

        # dphi/dx = norm * [H'_{n1}·(1/β)·Hy - H_{n1}·Hy·(x/β²)] * gauss
        if n1 >= 1
            Hx_prime = T(2 * n1) .* _hermite_vec(n1 - 1, xb)
        else
            Hx_prime = zeros(T, size(xb))
        end
        dphi_dx = norm_inv .* (Hx_prime .* ib .* Hy .- Hx .* Hy .* xb ./ beta) .* gauss

        # dphi/dy = norm * [H_{n1}·H'_{n2}·(1/β) - H_{n1}·Hy·(y/β²)] * gauss
        if n2 >= 1
            Hy_prime = T(2 * n2) .* _hermite_vec(n2 - 1, yb)
        else
            Hy_prime = zeros(T, size(yb))
        end
        dphi_dy = norm_inv .* (Hx .* Hy_prime .* ib .- Hx .* Hy .* yb ./ beta) .* gauss

        return phi, dphi_dx, dphi_dy
    end


    """
        phi, d2phi_dx2, d2phi_dy2, laplacian = _shapelet_curvature(
            n1, n2, beta, x, y)

    Evaluate the Laplacian ∇²φ at positions (x, y).
    Uses analytic second derivatives of Gaussian-Hermite functions.
    """
    function _shapelet_curvature(n1::Int, n2::Int, beta::Float64,
                                  x::AbstractVector, y::AbstractVector)
        xb = x ./ beta
        yb = y ./ beta
        T = eltype(x)
        ib2 = T(1) / beta^2

        norm_inv = T(1) / sqrt(T(2)^(n1 + n2) * T(pi) *
                    factorial(n1) * factorial(n2) * beta^2)
        gauss = @. exp(-(xb^2 + yb^2) / 2)

        Hx = _hermite_vec(n1, xb)
        Hy = _hermite_vec(n2, yb)

        # H' and H'' via recurrence
        Hx_p1 = n1 >= 1 ? T(2*n1) .* _hermite_vec(n1-1, xb) : zeros(T, size(xb))
        Hx_p2 = n1 >= 2 ? T(4*n1*(n1-1)) .* _hermite_vec(n1-2, xb) : zeros(T, size(xb))
        Hy_p1 = n2 >= 1 ? T(2*n2) .* _hermite_vec(n2-1, yb) : zeros(T, size(yb))
        Hy_p2 = n2 >= 2 ? T(4*n2*(n2-1)) .* _hermite_vec(n2-2, yb) : zeros(T, size(yb))

        # d²φ/dx² = norm * [H''_n1·Hy/β² - 2·H'_n1·Hy·(x/β²) + H_n1·Hy·(x²/β²-1)/β²] · gauss
        d2phi_dx2 = norm_inv .* (
            Hx_p2 .* Hy .* ib2
            .- 2 .* Hx_p1 .* Hy .* xb ./ beta^2
            .+ Hx .* Hy .* (xb.^2 .- 1) .* ib2
        ) .* gauss

        # d²φ/dy² = norm * [H_n1·H''_n2/β² - 2·H_n1·H'_n2·(y/β²) + H_n1·Hy·(y²/β²-1)/β²] · gauss
        d2phi_dy2 = norm_inv .* (
            Hx .* Hy_p2 .* ib2
            .- 2 .* Hx .* Hy_p1 .* yb ./ beta^2
            .+ Hx .* Hy .* (yb.^2 .- 1) .* ib2
        ) .* gauss

        laplacian = @. d2phi_dx2 + d2phi_dy2
        phi = @. norm_inv * Hx * Hy * gauss

        return phi, d2phi_dx2, d2phi_dy2, laplacian
    end


    # ═══════════════════════════════════════════════════════════════
    #  Regularized least-squares solver (updated)
    # ═══════════════════════════════════════════════════════════════

    """
        c, chi2 = solve_coeffs(A_masked, data, mask_idx, sigma;
                               lambda=0.01, regularisation=:ridge,
                               reg_matrix=nothing)

    Solve for shapelet coefficients via regularized least squares.

        c = (A^T A + λ Γ^T Γ)^{-1} A^T d

    # Regularisation types
    - `:ridge`     — Γ = I  (penalises large coefficients)
    - `:gradient`  — Γ from `build_reg_matrix(..., :gradient)`.
      Penalises non-smooth sources.  Must pass `reg_matrix`.
    - `:curvature` — Γ from `build_reg_matrix(..., :curvature)`.
      Penalises curvature.  Must pass `reg_matrix`.
    - `:none`      — no regularisation (λ = 0)

    # Arguments
    - `reg_matrix`: (N,N) matrix from `build_reg_matrix`.
      Required for `:gradient` / `:curvature`, ignored for others.

    # Returns
    - `c`    : coefficient vector (length N)
    - `chi2` : Σ((A·c − d)/σ)²
    """
    function solve_coeffs(A_masked, data, mask_idx, sigma::Real;
                          lambda::Real=0.01, regularisation::Symbol=:ridge,
                          reg_matrix=nothing)
        d = data[mask_idx]
        sigma_T = eltype(A_masked)(sigma)
        A_w = A_masked ./ sigma_T
        M = A_w' * A_w
        b_vec = A_w' * d

        # Regularisation
        if regularisation == :none || lambda <= 0
            M_reg = M
        elseif regularisation == :ridge
            N = size(A_masked, 2)
            M_reg = M + lambda * I
        elseif regularisation in (:gradient, :curvature)
            reg_matrix === nothing && error(
                "solve_coeffs: regularisation=:$regularisation requires " *
                "reg_matrix=<result from build_reg_matrix>")
            M_reg = M + lambda * reg_matrix
        else
            error("solve_coeffs: unknown regularisation=:$regularisation.  " *
                  "Use :none, :ridge, :gradient, or :curvature.")
        end

        # M_reg is N×N (N = 28~45) — always solve on CPU.
        # GPU \ on tiny matrices has higher kernel-launch overhead
        # than the solve itself.  Pull back if on GPU.
        M_cpu = M_reg isa AbstractArray{<:Any,2} && !(M_reg isa Matrix) ?
                Array(M_reg) : M_reg
        b_cpu = b_vec isa AbstractArray{<:Any,2} && !(b_vec isa Matrix) ?
                Array(b_vec) : b_vec
        c_cpu = M_cpu \ b_cpu

        # Move c back to A_masked's device.
        # M_cpu\b_cpu returns Float64; convert to A_masked's eltype.
        T = eltype(A_masked)
        c = similar(A_masked, T, size(A_masked, 2))
        copyto!(c, T.(c_cpu))

        model = A_masked * c
        chi2 = sum(((model .- d) ./ sigma_T).^2)
        return c, chi2
    end


    # ═══════════════════════════════════════════════════════════════
    #  Convenience: MCMC logp
    # ═══════════════════════════════════════════════════════════════

    """
        lp = shapelet_logp(A_masked, data, mask_idx, sigma)
        lp = shapelet_logp(lens_params, ...)  # high-level version

    Compute log-posterior for a shapelet-based lens model.

    High-level version (rebuilds ForwardModel from params):
        shapelet_logp(sys_builder, basis, params, data, mask, sigma;
                      lambda=0.01)

    Low-level version (pre-built design matrix):
        shapelet_logp(A_masked, data, mask_idx, sigma)

    # Returns
    `-0.5 * chi2` where chi2 is the regularized least-squares residual.
    """
    function shapelet_logp(A_masked, data, mask_idx, sigma::Real)
        c, chi2 = solve_coeffs(A_masked, data, mask_idx, sigma)
        return -chi2 / 2, c, chi2
    end


    # ═══════════════════════════════════════════════════════════════
    #  Model image reconstruction
    # ═══════════════════════════════════════════════════════════════

    """
        model = shapelet_model(A_conv, c, npix)

    Reconstruct the full model image from coefficients and the
    PSF-convolved design matrix.

    # Arguments
    - `A_conv`: full (npix, N) design matrix from `build_design`
    - `c`: coefficient vector from `solve_coeffs`
    - `npix`: total number of image pixels (nx * ny)

    # Returns
    2D image matrix of size (nx, ny).
    """
    function shapelet_model(A_conv, c, nx::Int, ny::Int)
        model_flat = A_conv * c
        return reshape(model_flat, nx, ny)
    end


    # ═══════════════════════════════════════════════════════════════
    #  Convenience: β profile likelihood
    # ═══════════════════════════════════════════════════════════════

    """
        best_beta, chi2s = profile_beta(sys, n_max, betas, data, mask,
                                         sigma; kwargs...)

    Scan a grid of β values and return the optimal β (minimum χ²)
    plus the χ² values at each β.

    # Arguments
    - `sys`: ForwardModel (lens + grid)
    - `n_max`: shapelet order
    - `betas`: vector of β values to scan
    - `data`, `mask`, `sigma`: as usual

    # Returns
    - `best_beta`: β with minimum χ²
    - `chi2s`: vector of χ² values (same length as betas)

    # Example
    ```julia
    betas = range(0.05, 0.30; length=20)
    best_beta, chi2s = profile_beta(sys, 8, betas, data, mask, 0.01)
    ```
    """
    function profile_beta(sys, n_max::Int, betas::AbstractVector,
                           data, mask, sigma::Real;
                           xc_src=0.0, yc_src=0.0, lambda=0.01)
        chi2s = zeros(length(betas))
        for (i, beta) in enumerate(betas)
            basis = ShapeletBasis(n_max, beta)
            A_masked, _, _ = build_design(sys, basis, data, mask;
                                           xc_src=xc_src, yc_src=yc_src)
            _, chi2s[i] = solve_coeffs(A_masked, data, findall(vec(mask)),
                                       sigma; lambda=lambda)
        end
        best_idx = argmin(chi2s)
        return betas[best_idx], chi2s
    end

end # module LensShapelet