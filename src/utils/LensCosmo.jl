module LensCosmo

    import Cosmology
    using Cosmology
    export Cosmology
    export angular_diameter_distance, lens_distance_ratio, time_delay_distance

    # ═══════════════════════════════════════════════════════════════
    #  Angular diameter distance
    #  Reference: https://doi.org/10.1051/0004-6361/201424881
    # ═══════════════════════════════════════════════════════════════

    """
        angular_diameter_distance(cosmo::AbstractCosmology, z::Real) → Quantity

    Angular diameter distance from observer (z=0) to redshift `z`, in Mpc.
    """
    function angular_diameter_distance(cosmo::Cosmology.AbstractCosmology, z::Real)
        return Cosmology.angular_diameter_dist(cosmo, Float64(z))
    end

    """
        angular_diameter_distance(cosmo::AbstractCosmology, z1::Real, z2::Real) → Quantity

    Angular diameter distance between redshifts `z1` and `z2`, in Mpc.
    """
    function angular_diameter_distance(cosmo::Cosmology.AbstractCosmology, z1::Real, z2::Real)
        return Cosmology.angular_diameter_dist(cosmo, Float64(z1), Float64(z2))
    end

    # ═══════════════════════════════════════════════════════════════
    #  Lensing distance ratio
    #  beta = theta - (D_ls / D_s) * alpha_phys(theta)
    # ═══════════════════════════════════════════════════════════════

    """
        lens_distance_ratio(cosmo::AbstractCosmology, z_lens::Real, z_source::Real) → Float64

    Compute the cosmological distance ratio D_ls / D_s used in the lens equation:

        beta = theta - (D_ls / D_s) * alpha_phys(theta)

    where D_ls is the angular diameter distance from lens to source,
    and D_s is the angular diameter distance from observer to source.
    """
    function lens_distance_ratio(cosmo::Cosmology.AbstractCosmology, z_lens::Real, z_source::Real)
        D_s  = Cosmology.angular_diameter_dist(cosmo, Float64(z_source))
        D_ls = Cosmology.angular_diameter_dist(cosmo, Float64(z_lens), Float64(z_source))
        return D_ls / D_s
    end

    # ═══════════════════════════════════════════════════════════════
    #  Time-delay distance
    #
    #  D_Δt = (1 + z_lens) · D_l · D_s / D_ls
    #
    #  Multiplied by the Fermat potential τ(θ) (in rad²) and
    #  divided by c, this gives the light travel-time delay.
    # ═══════════════════════════════════════════════════════════════

    """
        time_delay_distance(cosmo::AbstractCosmology, z_lens::Real, z_source::Real) → Float64

    Compute the time-delay distance:

        D_Δt = (1 + z_lens) · D(0, z_lens) · D(0, z_source) / D(z_lens, z_source)

    Returns the distance in Mpc as a plain `Float64`.
    """
    function time_delay_distance(cosmo::Cosmology.AbstractCosmology, z_lens::Real, z_source::Real)
        D_l  = Cosmology.angular_diameter_dist(cosmo, Float64(z_lens))
        D_s  = Cosmology.angular_diameter_dist(cosmo, Float64(z_source))
        D_ls = Cosmology.angular_diameter_dist(cosmo, Float64(z_lens), Float64(z_source))
        result = (1 + z_lens) * D_l * D_s / D_ls
        return result.val   # strip Mpc unit → plain Float64
    end

    end
