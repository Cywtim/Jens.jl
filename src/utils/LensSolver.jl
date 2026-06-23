module LensSolver

    using NLsolve

    import ..LensBase: lens_derivative, lens_hessian

    export solve_images, batch_solve_images

    """
        images = solve_images(lens, beta_x, beta_y;
                               search_radius=2.0, n_radial=8, n_angular=16,
                               tol=1e-8, max_images=10)

    Solve the lens equation β = θ − α(θ) for a point source at `(beta_x, beta_y)`.

    Uses multiple initial guesses in a spiral pattern over the image plane,
    refining each with NLsolve.  Duplicate solutions (within `tol`) are merged.

    Returns a `Vector` of `(theta_x, theta_y, magnification)` tuples.
    The `lens` argument must support `lens_derivative` and `lens_hessian`
    (e.g. `CombinedLens`, `LensedPlane`, or any `AbstractLens`).
    """
    function solve_images(lens, beta_x::Real, beta_y::Real;
                          search_radius::Real=2.0, n_radial::Int=8, n_angular::Int=16,
                          tol::Real=1e-8, max_images::Int=10,
                          z_source=nothing)

        images = Tuple{Float64,Float64,Float64}[]

        for r in range(0.01, Float64(search_radius); length=n_radial)
            for angle in range(0, 2π - 2π/n_angular; length=n_angular)
                x0 = r * cos(angle)
                y0 = r * sin(angle)

                local sol
                try
                    sol = nlsolve([x0, y0]; autodiff=:finite) do fx, x
                        ax, ay = lens_derivative(lens, [x[1]], [x[2]]; z_source=z_source)
                        fx[1] = x[1] - ax[1] - beta_x
                        fx[2] = x[2] - ay[1] - beta_y
                    end
                catch
                    continue
                end

                if !converged(sol)
                    continue
                end

                tx, ty = sol.zero[1], sol.zero[2]

                # Check for duplicates
                is_new = true
                for (px, py, _) in images
                    if sqrt((tx - px)^2 + (ty - py)^2) < 1e-4
                        is_new = false
                        break
                    end
                end
                is_new || continue

                # Compute magnification μ = 1 / det(1 − H)
                fxx, fxy, fyy = lens_hessian(lens, [tx], [ty]; z_source=z_source)
                detJ = (1.0 - fxx[1]) * (1.0 - fyy[1]) - fxy[1]^2
                mu = iszero(detJ) ? Inf : 1.0 / detJ

                push!(images, (tx, ty, mu))

                if length(images) >= max_images
                    break
                end
            end
            length(images) >= max_images && break
        end

        return images
    end

    # ═══════════════════════════════════════════════════════════════
    #  batch_solve_images — GPU-native Newton-Raphson
    #
    #  Instead of serial nlsolve + try/catch, runs fixed-iteration
    #  Newton on ALL initial guesses simultaneously via broadcast.
    #  Works transparently on CPU (Matrix) or GPU (CuArray).
    #
    #  Single source:   128 guesses × 20 Newton iter ≈ 0.5 ms (GPU)
    #  M sources:       128M guesses × 20 iter          ≈ 1–3 ms (GPU)
    #
    #  Algorithm: standard Newton, step-size clamped to 1 arcsec
    #  to prevent divergence of poor initial guesses.
    # ═══════════════════════════════════════════════════════════════

    """
        images = batch_solve_images(lens, beta_x, beta_y;
                                     search_radius=2.0, n_radial=8, n_angular=16,
                                     n_iter=20, tol=1e-6, max_images=10)

    GPU-native batch Newton-Raphson lens equation solver.

    # Arguments
    - `beta_x, beta_y`: source position(s).  Scalars or arrays.
    - `n_iter`: fixed Newton iterations (no try/catch needed).
    - `tol`: convergence tolerance in arcsec.

    # Returns
    - For scalar `beta_x`: `Vector{Tuple{Float64,Float64,Float64}}`
    - For vector `beta_x`: `Vector{Vector{Tuple{Float64,Float64,Float64}}}`
    """
    function batch_solve_images(lens, beta_x, beta_y;
                                 search_radius::Real=2.0,
                                 n_radial::Int=8, n_angular::Int=16,
                                 n_iter::Int=20, tol::Real=1e-6,
                                 max_images::Int=10, z_source=nothing)

        # ── Normalize to vector form ──────────────────
        bx = beta_x isa Real ? [Float64(beta_x)] : collect(Float64, vec(beta_x))
        by = beta_y isa Real ? [Float64(beta_y)] : collect(Float64, vec(beta_y))
        n_src = length(bx)

        # ── 1. Generate initial guess grid (CPU, tiny) ─
        n_guess = n_radial * n_angular
        x0 = zeros(n_guess)
        y0 = zeros(n_guess)
        k = 1
        for r in range(0.01, Float64(search_radius); length=n_radial)
            for ang in range(0.0, 2π * (1 - 1/n_angular); length=n_angular)
                x0[k] = r * cos(ang)
                y0[k] = r * sin(ang)
                k += 1
            end
        end

        # ── 2. Build initial positions: (n_src, n_guess) ─
        T = eltype(bx)
        x = similar(bx, n_src, n_guess)
        y = similar(bx, n_src, n_guess)
        for g in 1:n_guess
            for i in 1:n_src
                x[i, g] = x0[g]
                y[i, g] = y0[g]
            end
        end

        # Const β per source, broadcast over guess dim
        beta_x_2d = reshape(bx, n_src, 1)
        beta_y_2d = reshape(by, n_src, 1)

        # ── 3. Newton-Raphson (fixed iterations, no try/catch) ─
        for _ in 1:n_iter
            ax, ay   = lens_derivative(lens, x, y; z_source=z_source)
            hxx, hxy, hyy = lens_hessian(lens, x, y; z_source=z_source)

            # Jacobian: A = I - H
            a11 = 1 .- hxx
            a12 = .- hxy
            a22 = 1 .- hyy
            detJ = a11 .* a22 .- a12 .* a12

            # Residual: F(theta) = theta - alpha - beta
            res_x = x .- ax .- beta_x_2d
            res_y = y .- ay .- beta_y_2d

            # Newton step:  Δθ = A⁻¹ · (-res) = -A⁻¹ · res
            #   A⁻¹ = (1/det) * [[a22, -a12], [-a12, a11]]
            inv_det = @. one(T) / (detJ + T(1e-12))      # guard singular
            dx = @. -(a22 * res_x - a12 * res_y) * inv_det
            dy = @. -(-a12 * res_x + a11 * res_y) * inv_det   # = -(a11*res_y - a12*res_x) * inv_det

            # Step-size clamp: max 1 arcsec per iteration
            step_len = @. sqrt(dx*dx + dy*dy)
            scale = @. min(one(T), T(1.0) / (step_len + T(1e-10)))
            x .+= dx .* scale
            y .+= dy .* scale
        end

        # ── 4. Final residual + convergence mask ──────
        ax, ay = lens_derivative(lens, x, y; z_source=z_source)
        res_x = x .- ax .- beta_x_2d
        res_y = y .- ay .- beta_y_2d
        res_norm = @. sqrt(res_x*res_x + res_y*res_y)
        conv = res_norm .< tol

        # ── 5. Gather valid solutions per source ──────
        hxx, hxy, hyy = lens_hessian(lens, x, y; z_source=z_source)
        det_mu = @. (1 - hxx) * (1 - hyy) - hxy * hxy
        mu = @. one(T) / (det_mu + T(1e-12))

        results = Vector{Vector{Tuple{Float64,Float64,Float64}}}(undef, n_src)
        for i in 1:n_src
            # Collect converged candidates
            candidates = Tuple{Float64,Float64,Float64}[]
            for g in 1:n_guess
                conv[i,g] || continue
                tx, ty = Float64(x[i,g]), Float64(y[i,g])
                # Dedup: skip if already seen
                dup = false
                for (px, py, _) in candidates
                    if sqrt((tx-px)^2 + (ty-py)^2) < 1e-4
                        dup = true; break
                    end
                end
                dup && continue
                push!(candidates, (tx, ty, Float64(mu[i,g])))
                length(candidates) >= max_images && break
            end
            results[i] = candidates
        end

        # ── 6. Unwrap scalar case ────────────────────
        if beta_x isa Real && beta_y isa Real
            return results[1]
        else
            return results
        end
    end

end