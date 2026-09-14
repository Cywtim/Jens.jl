"""
    SISreal — SIS with real-domain derivatives

Identical to SIS physically, but uses strictly real-valued math
(no complex numbers) with explicit R=0 handling.  Useful for
environments where complex arithmetic is unavailable.

# Parameters
- `theta_E`: Einstein radius [arcsec]
- `xcentre`, `ycentre`: lens centre [arcsec]

# Example
    lens = SingleModel(SISreal; theta_E=1.2)
"""
module SISreal

    function LensCheck(; theta_E::Real, xcentre::Real=0., ycentre::Real=0.)

        para = [theta_E, xcentre, ycentre]

        if all([0,-100,-100]< para) && all([100,100,100]>para)
            return
        else
            error("The SIS configuration is out of range!")        
        end
    end


    function LensPotential(xg::AbstractArray, yg::AbstractArray;
         theta_E::Real, xcentre::Real=0., ycentre::Real=0.)
        #=
            The mass profile for singular isothermal ellipsoid (SIE)

            mass profile
            \kappa(x, y) = \frac{1}{2} \left(\frac{\theta_{E}}{\sqrt{q x^2 + y^2/q}} \right)
            with
            \theta_{E} is the (circularized) Einstein radius,
            q is the minor/major axis ratio,
            x, y are defined in a coordinate system aligned with the major and minor axis of the lens
        
        =#

        xsh = xg .- xcentre
        ysh = yg .- ycentre

        fermat = theta_E .* sqrt.( xsh.^2 .+ ysh.^2 )
            return fermat
    end


    function LensDerivative(xg::AbstractArray, yg::AbstractArray;
     theta_E::Real, xcentre::Real=0., ycentre::Real=0.)

        xsh = xg .- xcentre
        ysh = yg .- ycentre

        R = @. sqrt(xsh^2 + ysh^2)
        T = promote_type(eltype(R), typeof(theta_E))
        a = @. ifelse(R > 0, theta_E / R, zero(T))

        f_x = @. a * xsh
        f_y = @. a * ysh

        return f_x, f_y
    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
     theta_E::Real, xcentre::Real=0., ycentre::Real=0.)

        xsh = @. xg - xcentre
        ysh = @. yg - ycentre

        R3 = @. (xsh^2 + ysh^2)^1.5   # R^3 = (x^2+y^2)^(3/2)
        T = promote_type(eltype(R3), typeof(theta_E))
        h = @. ifelse(R3 > 0, theta_E / R3, zero(T))

        f_xx = @. ysh * ysh * h
        f_yy = @. xsh * xsh * h
        f_xy = @. -xsh * ysh * h
        return f_xx, f_xy, f_yy

    end

end