module TimeDelay

    using Jens.LensSystem: ForwardModel,
                           source_redshift, source_beta,
                           get_cosmology, get_z_lens, maybe_z_source
    import Jens.LensBase: AbstractLens, LensFermat
    import Jens.LensCosmo: time_delay_distance
    import Jens.LensConstants: ARCSEC2_TO_RAD2, C_LIGHT_MPC_S
    using Jens.LensSolver: solve_images

    export LensTimeDelay, image_time_delays

    # ═══════════════════════════════════════════════════════════════
    #  Grid-based time-delay surface
    # ═══════════════════════════════════════════════════════════════

    """
        Δt = LensTimeDelay(xg, yg, beta=[0.,0.];
                           LensModel, LensKwargs=NamedTuple(), z_source=nothing)

    Compute the time-delay surface (in **seconds**) on a grid.

        c · Δt(θ, β) = D_Δt  ·  τ(θ, β)

    where D_Δt is the time-delay distance and τ is the angular Fermat
    potential.  Requires `LensModel` to carry cosmology (`LensedPlane`,
    `MultiLensedPlane`, or `ForwardModel`).

    # Example
    ```julia
    Δt_map = LensTimeDelay(xg, yg, [bx, by]; LensModel=lp, z_source=1.5)
    ```
    """
    function LensTimeDelay(xg::AbstractMatrix, yg::AbstractMatrix,
                            beta=[0.,0.]; LensModel,
                            LensKwargs=NamedTuple(), z_source=nothing)
        cosmo = get_cosmology(LensModel)
        z_lens = get_z_lens(LensModel)
        zs = something(z_source, maybe_z_source(LensModel))

        D_dt = time_delay_distance(cosmo, z_lens, zs)
        tau  = LensFermat(xg, yg, beta;
                          LensModel=LensModel, LensKwargs=LensKwargs, z_source=zs)

        tau_rad2 = tau .* ARCSEC2_TO_RAD2
        return D_dt .* tau_rad2 ./ C_LIGHT_MPC_S
    end

    """
        Δt = LensTimeDelay(sys::ForwardModel; beta=nothing, z_source=nothing, kwargs...)

    Compute the time-delay surface for a complete lens system.
    All parameters are auto-detected from the system.

    # Example
    ```julia
    sys = ForwardModel(lens_plane=lp, source_plane=LightPlane(agn; z=1.5), grid=grid)
    Δt_map = LensTimeDelay(sys)   # everything auto-detected
    ```
    """
    function LensTimeDelay(sys::ForwardModel; beta=nothing,
                            z_source=nothing, kwargs...)
        zs = something(z_source, source_redshift(sys.source_plane))
        bx = something(beta, source_beta(sys.source_plane))
        xg, yg = sys.grid.xg, sys.grid.yg
        return LensTimeDelay(xg, yg, bx;
                             LensModel=sys.lens_plane, z_source=zs, kwargs...)
    end


    # ═══════════════════════════════════════════════════════════════
    #  Point-wise time delay
    # ═══════════════════════════════════════════════════════════════

    """
        Δt = LensTimeDelay(sys::ForwardModel, theta_x::Real, theta_y::Real;
                           beta=nothing, z_source=nothing, kwargs...)

    Compute the time delay at a single image position (in seconds).

    # Example
    ```julia
    images = solve_images(sys, 0.05, -0.03)
    for (tx, ty, mu) in images
        Δt = LensTimeDelay(sys, tx, ty)
        println("Δt = ", Δt / DAY_TO_SEC, " days")
    end
    ```
    """
    function LensTimeDelay(sys::ForwardModel, theta_x::Real, theta_y::Real;
                           beta=nothing, z_source=nothing, kwargs...)
        zs = something(z_source, source_redshift(sys.source_plane))
        bx = something(beta, source_beta(sys.source_plane))

        D_dt = time_delay_distance(sys)
        tau = LensFermat([theta_x;;], [theta_y;;], bx;
                         LensModel=sys.lens_plane, z_source=zs, kwargs...)[1, 1]

        return D_dt * tau * ARCSEC2_TO_RAD2 / C_LIGHT_MPC_S
    end

    """
        delays = LensTimeDelay(sys::ForwardModel,
                                theta_x::AbstractVector, theta_y::AbstractVector;
                                beta=nothing, z_source=nothing, kwargs...)

    Compute time delays at multiple image positions (in seconds).

    # Example
    ```julia
    images = solve_images(sys, 0.05, -0.03)
    tx = [t[1] for t in images]
    ty = [t[2] for t in images]
    delays = LensTimeDelay(sys, tx, ty)
    ```
    """
    function LensTimeDelay(sys::ForwardModel,
                            theta_x::AbstractVector, theta_y::AbstractVector;
                            beta=nothing, z_source=nothing, kwargs...)
        zs = something(z_source, source_redshift(sys.source_plane))
        bx = something(beta, source_beta(sys.source_plane))
        n = length(theta_x)
        @assert length(theta_y) == n "theta_x and theta_y must have the same length"

        D_dt = time_delay_distance(sys)
        delays = Vector{Float64}(undef, n)
        for i in 1:n
            tau = LensFermat([theta_x[i];;], [theta_y[i];;], bx;
                             LensModel=sys.lens_plane, z_source=zs, kwargs...)[1, 1]
            delays[i] = D_dt * tau * ARCSEC2_TO_RAD2 / C_LIGHT_MPC_S
        end
        return delays
    end


    # ═══════════════════════════════════════════════════════════════
    #  Solve images + compute time delays
    # ═══════════════════════════════════════════════════════════════

    """
            images, delays = image_time_delays(sys::ForwardModel, beta_x, beta_y;
                                                z_source=nothing, solver=:nlsolve, kwargs...)

        Solve the lens equation for a point source at `(beta_x, beta_y)` and
        compute the time delay at each image.

        Returns `(images, delays)` where:
        - `images` is a `Vector` of `(theta_x, theta_y, magnification)` tuples
        - `delays` is a `Vector{Float64}` of time delays in seconds

        # Example
        ```julia
        images, delays = image_time_delays(sys, 0.05, -0.03)
        for i in 1:length(delays), j in i+1:length(delays)
            dt = abs(delays[i] - delays[j])
            println("dt = ", dt / DAY_TO_SEC, " days")
        end
        ```
        """
    function image_time_delays(sys::ForwardModel, beta_x::Real, beta_y::Real;
                                z_source=nothing, solver::Symbol=:nlsolve, kwargs...)
        zs = something(z_source, source_redshift(sys.source_plane))
        images = solve_images(sys.lens_plane, beta_x, beta_y; z_source=zs)
        if isempty(images)
            return images, Float64[]
        end
        tx = [t[1] for t in images]
        ty = [t[2] for t in images]
        delays = LensTimeDelay(sys, tx, ty;
                               beta=Float64[beta_x, beta_y],
                               z_source=zs, kwargs...)
        return images, delays
    end

end # module TimeDelay