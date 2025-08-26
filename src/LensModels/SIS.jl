module SIS
    
    
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

        R = sqrt.( xsh.^2 .+ ysh.^2 )
        a = zeros(size(R))
        r = R[R.>0.]  # in the SIS regime
        a[R.==0.] .= 0
        a[R.>0.] .= theta_E ./ r

        f_x = a .* xsh
        f_y = a .* ysh

        return f_x, f_y
    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
         theta_E::Real, xcentre::Real=0., ycentre::Real=0.)

        xsh = xg - xcentre
        ysh = yg - ycentre

        R .= sqrt.( xsh.^2 .+ ysh.^2 ).^(3.0/2)
        h = zeros(size(R))
        r = R[R.>0.]  # in the SIS regime
        h[R.==0.] .= 0
        h[R.>0.] .= theta_E ./ r

        f_xx = ysh .* ysh .* h
        f_yy = xsh .* xsh .* h
        f_xy = -xsh .* ysh .* h
        return f_xx, f_xy, f_yy

    end

end