
module PointMass

    using  LazyGrids

    include("../LensUtils.jl")

    

    function LensCheck(; theta_E::Real, xcentre::Real=0., ycentre::Real=0.)

        para = [theta_E, xcentre, ycentre]

        if all([0., -100, -100]< para) && all([10, 100, 100]>para)
            return
        else
            error("The NIE configuration is out of range!")        
        end
    end

    # array like input
    function LensMass(xg::AbstractArray, yg::AbstractArray;
        theta_E=.5, xcentre=0., ycentre=0.)

        xsh = xg .- xcentre
        ysh = yg .- ycentre

        a = @. sqrt(xsh^2 + ysh^2)
        a = @. max(10^(-20), a)

        
        f = @. theta_E^2 * log(a)
        return f

    end

    function  LensDerivative(xg::AbstractArray, yg::AbstractArray;
        theta_E=1.5, xcentre=0., ycentre=0.)
        #=
            compute the physical deflection angle of a point mass, given as an
            Einstein radius.
        =#

        xsh = xg .- xcentre
        ysh = yg .- ycentre

        a = @. sqrt(xsh^2 + ysh^2)
        a = @. max(10^(-20), a)
        
        alpha = @. theta_E^2 / a
        f_x = @. alpha * xsh / a
        f_y = @. alpha * ysh / a

        return f_x, f_y

    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
          theta_E::Real=1.5, xcentre::Real=0., ycentre::Real=0., diff::Real= 1e-10)

          xsh = xg .- xcentre
          ysh = yg .- ycentre
  
          a = @. sqrt(xsh^2 + ysh^2)
          a = @. max(10^(-20), a)

          f_xx = @. theta_E^2 * (ysh^2 - xsh^2) / a^4
          f_yy = @. theta_E^2 * (xsh^2 - ysh^2) / a^4
          f_xy = @. -theta_E^2 * 2 * xsh * ysh / a^4
        return f_xx, f_xy, f_yy

    end

end