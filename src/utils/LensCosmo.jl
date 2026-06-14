module LensCosmo
    using Cosmology

    export angular_diameter_distance, lens_distance_ratio

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

    end
