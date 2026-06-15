module SIE

    using Jens.LensUtils

    function LensCheck(; theta_E::Real, e1::Real, e2::Real,
                         xcentre::Real=0., ycentre::Real=0.)
        para = [theta_E, e1, e2, xcentre, ycentre]
        if all([0., -0.5, -0.5, -100., -100.] .< para) && all([100., 0.5, 0.5, 100., 100.] .> para)
            return
        else
            error("The SIE configuration is out of range!")
        end
    end

    function _to_major_axes(theta_E, e1, e2)
        q, varphi = LensUtils.e2phiq(e1, e2)
        b = theta_E * sqrt((1.0 + q^2) / 2.0)
        q = min(q, 0.9999)
        return b, q, varphi
    end

    function _deflection_ma(xsh, ysh; b, q, s)
        # Deflection (alpha_x, alpha_y) and psi in the major-axis aligned frame (phi = 0)
        psi = sqrt.(q.^2 .* (s.^2 .+ xsh.^2) .+ ysh.^2)
        f_x = b ./ sqrt.(1.0 .- q.^2) .* atan.(sqrt.(1.0 .- q.^2) .* xsh ./ (psi .+ s))
        f_y = b ./ sqrt.(1.0 .- q.^2) .* atanh.(sqrt.(1.0 .- q.^2) .* ysh ./ (psi .+ q.^2 .* s))
        return f_x, f_y, psi
    end


    function LensPotential(xg::AbstractArray, yg::AbstractArray;
              theta_E::Real, e1::Real, e2::Real,
              s::Real=1e-4, xcentre::Real=0., ycentre::Real=0.)
        #=
            Lensing potential psi(theta) for Singular Isothermal Ellipsoid (SIE).

            Reference: Keeton & Kochanek 1998, arXiv:astro-ph/9705194

            Parameters:
            - theta_E:        (circularized) Einstein radius
            - e1, e2:         ellipticity components
            - s:              core radius for numerical stability (default 1e-4)
            - xcentre,ycentre: lens centre
        =#
        b, q, varphi = _to_major_axes(theta_E, e1, e2)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        f_x, f_y, psi = _deflection_ma(xsh, ysh; b=b, q=q, s=s)

        f = @. xsh * f_x + ysh * f_y - 0.5 * b * s * log((psi + s)^2 + (1.0 - q^2) * xsh^2)
        return f
    end

    function LensDerivative(xg::AbstractArray, yg::AbstractArray;
             theta_E::Real, e1::Real, e2::Real,
             s::Real=1e-4, xcentre::Real=0., ycentre::Real=0.)
        #=
            Deflection angle alpha(theta) for Singular Isothermal Ellipsoid (SIE).

            Reference: Keeton & Kochanek 1998, arXiv:astro-ph/9705194

            Parameters:
            - theta_E:        (circularized) Einstein radius
            - e1, e2:         ellipticity components
            - s:              core radius for numerical stability (default 1e-4)
            - xcentre,ycentre: lens centre
        =#
        b, q, varphi = _to_major_axes(theta_E, e1, e2)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        f_x, f_y, _ = _deflection_ma(xsh, ysh; b=b, q=q, s=s)
        f_x, f_y = LensUtils.LensRotation(f_x, f_y, varphi)
        return f_x, f_y
    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
             theta_E::Real, e1::Real, e2::Real,
             s::Real=1e-4, xcentre::Real=0., ycentre::Real=0.,
             diff::Real=1e-10)
        #=
            Hessian of the lensing potential for SIE (finite-difference).

            Computes f_xx, f_xy, f_yy via finite difference of deflection
            in the major-axis frame, then rotates back using the spin-2
            transformation for shear.
        =#
        b, q, varphi = _to_major_axes(theta_E, e1, e2)

        # shift → rotate to major axis
        xsh = xg .- xcentre
        ysh = yg .- ycentre
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        # Hessian in major-axis frame via finite difference of deflection
        f_x, f_y, _   = _deflection_ma(xsh, ysh; b=b, q=q, s=s)
        f_x_dx, _, _  = _deflection_ma(xsh .+ diff, ysh; b=b, q=q, s=s)
        f_x_dy, f_y_dy, _ = _deflection_ma(xsh, ysh .+ diff; b=b, q=q, s=s)

        f_xx = (f_x_dx .- f_x) ./ diff
        f_xy = (f_x_dy .- f_x) ./ diff
        f_yy = (f_y_dy .- f_y) ./ diff

        # rotate shear back to original frame (spin-2)
        kappa  = @. 0.5 * (f_xx + f_yy)
        g1_ma  = @. 0.5 * (f_xx - f_yy)
        g2_ma  = f_xy
        gamma1 = @.   cos(2.0 * varphi) * g1_ma + sin(2.0 * varphi) * g2_ma
        gamma2 = @. - sin(2.0 * varphi) * g1_ma + cos(2.0 * varphi) * g2_ma

        f_xx_out = @. kappa + gamma1
        f_yy_out = @. kappa - gamma1
        f_xy_out = gamma2
        return f_xx_out, f_xy_out, f_yy_out
    end

end
