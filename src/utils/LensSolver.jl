module LensSolver

    using NLsolve

    import ..LensBase: lens_derivative, lens_hessian

    export solve_images

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

end