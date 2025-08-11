
module ShearGamma

    include("../LensUtils.jl")

    function LensCheck(; gamma::Real, psi::Real, xcentre::Real=0.0, ycentre::Real=0.0)

        para = [gamma, psi, xcentre, ycentre]

        if all([-0.5, 0.,-100,-100]< para) && all([-0.5, pi,100,100]>para)
            return
        else
            error("The shear configuration is out of range!")        
        end
    end

    function LensMass(x::AbstractArray, y::AbstractArray; gamma::Real, psi::Real, xcentre::Real=0.0, ycentre::Real=0.0)
        
        xsh = x .- xcentre
        ysh = y .- ycentre

        fermat = 1.0 ./ 2.0 .* (gamma1 .* xsh .^2 .+ 2 .* gamma2 .* xsh .* ysh .- gamma1 .* ysh .^2)

        return fermat
    end

    function LensDerivative(x::AbstractArray, y::AbstractArray; gamma1::Real, psi::Real, xcentre::Real=0.0, ycentre::Real=0.0)
        
        xsh = x .- xcentre
        ysh = y .- ycentre

        f_x = gamma1 .* xsh .+ gamma2 .* ysh
        f_y = gamma2 .* xsh .- gamma1 .* ysh

        return f_x, f_y
    end

    function LensHessian(x::AbstractArray, y::AbstractArray; gamma1::Real, psi::Real, xcentre::Real=0.0, ycentre::Real=0.0, kappa=0.)

        xsh = x .- xcentre
        ysh = y .- ycentre

        f_xx = kappa .+ gamma1
        f_xy = gamma2
        f_yy = kappa .- gamma1
        
        return f_xx, f_xy, f_yy
    end

end