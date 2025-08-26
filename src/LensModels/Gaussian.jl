module Gaussian
    #  https://doi.org/10.1093/mnras/stz1796
    #=
        kappa = kappa0 * exp(-(q^2*xsh^2 + ysh^2)/(2*sigma^2))
    =#

    using SpecialFunctions:erf as erf
    using SpecialFunctions:erfi as erfi
    EulerGamma = MathConstants.eulergamma

    include("../LensUtils.jl")

    function Zeta_z(z::Complex; q::Real=0.9, sigma::Real=0.5)
        lambda = @. exp( - (q^2 * z^2)/(2*sigma^2 *(1-q^2)))
        fir = @. erfi((q*z)/(sigma * sqrt(2*(1-q^2))))
        sec = @. erfi( (q^2*real(z)+ imag(z)*im)/(sigma*sqrt(2*(1-q^2))))
        return lambda .* (fir .- sec)
    end

    function LensPotential(xg::AbstractArray, yg::AbstractArray;
                    kappa0::Real, q::Real, varphi::Real, sigma::Real,
                    xcentre::Real=0., ycentre::Real=0.)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        # in form of erf function
        


        return f
    end

    function LensDerivative(xg::AbstractArray, yg::AbstractArray;
                    kappa0::Real, q::Real, sigma::Real, varphi::Real=0,
                    xcentre::Real=0., ycentre::Real=0.)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        zsh = xsh .+ ysh .* im

        alphaz = @. kappa0 * sigma * sqrt(2*pi/(1-q^2)) * Zeta_z(zsh;q=q, sigma=sigma)

        f_x = real(alphaz)
        f_y = imag(alphaz)

        return f_x, f_y

    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
                    kappa0::Real, q::Real, sigma::Real, varphi::Real=0, 
                    xcentre::Real=0., ycentre::Real=0.)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        
        zsh = xsh .+ ysh .* im
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        kappa = @. kappa0 * exp(-(q^2*xsh^2 + ysh^2)/(2*sigma^2))

        gammaz = @. 1/(1-q^2) * ((1+q^2)*kappa - 2*q*kappa0 + 
                    (sqrt(2*pi)*q^2*kappa0*zsh)/(sigma*sqrt(1-q^2))*Zeta_z(zsh;q=q,sigma=sigma))
        
        gamma1 = real(gammaz)
        gamma2 = imag(gammaz)

        f_xx = kappa + gamma1
        f_yy = kappa - gamma1
        f_xy = gamma2

        return f_xx, f_xy, f_yy
    end

end