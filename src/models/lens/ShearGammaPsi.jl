
"""
    ShearGammaPsi — External Shear (γ, ψ)

Same as Shear but parameterized by amplitude γ and position angle ψ:
    γ₁ = γ cos(2ψ),  γ₂ = γ sin(2ψ)

# Parameters
- `gamma`: shear amplitude
- `psi`: position angle [rad]
- `xcentre`, `ycentre`: centre [arcsec]

# Example
    lens = SingleModel(ShearGammaPsi; gamma=0.05, psi=0.3)
"""
module ShearGammaPsi

    function LensCheck(; gamma::Real, psi::Real, xcentre::Real=0.0, ycentre::Real=0.0)

        para = [gamma, psi, xcentre, ycentre]

        if all([-0.5, 0.,-100,-100]< para) && all([-0.5, pi,100,100]>para)
            return
        else
            error("The shear configuration is out of range!")
        end
    end

    # gamma1 = gamma * cos(2*psi), gamma2 = gamma * sin(2*psi)
    @inline _g1(gamma::Real, psi::Real) = gamma * cos(2 * psi)
    @inline _g2(gamma::Real, psi::Real) = gamma * sin(2 * psi)

    function LensPotential(x::AbstractArray, y::AbstractArray; gamma::Real, psi::Real, xcentre::Real=0.0, ycentre::Real=0.0)
        T = promote_type(eltype(x), eltype(y), typeof(gamma), typeof(psi))
        g1 = T(_g1(gamma, psi))
        g2 = T(_g2(gamma, psi))

        xsh = @. x - xcentre
        ysh = @. y - ycentre

        fermat = @. T(0.5) * (g1 * xsh^2 + T(2) * g2 * xsh * ysh - g1 * ysh^2)

        return fermat
    end

    function LensDerivative(x::AbstractArray, y::AbstractArray; gamma::Real, psi::Real, xcentre::Real=0.0, ycentre::Real=0.0)
        T = promote_type(eltype(x), eltype(y), typeof(gamma), typeof(psi))
        g1 = T(_g1(gamma, psi))
        g2 = T(_g2(gamma, psi))

        xsh = @. x - xcentre
        ysh = @. y - ycentre

        f_x = @. g1 * xsh + g2 * ysh
        f_y = @. g2 * xsh - g1 * ysh

        return f_x, f_y
    end

    function LensHessian(x::AbstractArray, y::AbstractArray; gamma::Real, psi::Real, xcentre::Real=0.0, ycentre::Real=0.0, kappa::Real=0.0)
        T = promote_type(eltype(x), eltype(y), typeof(gamma), typeof(psi), typeof(kappa))
        g1 = T(_g1(gamma, psi))
        g2 = T(_g2(gamma, psi))
        k  = T(kappa)

        f_xx = @. k + g1
        f_xy = @. g2
        f_yy = @. k - g1

        return f_xx, f_xy, f_yy
    end

end