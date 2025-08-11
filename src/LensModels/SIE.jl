module SIE

    include("../LensUtils.jl")

    function LensCheck(; theta_E::Real, e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        para = [theta_E, e1, e2, xcentre, ycentre]

        if all([0,-0.5,0.5,-100,-100]< para) && all([100,0.5,0.5,100,100]>para)
            return
        else
            error("The SIE configuration is out of range!")        
        end
    end

    function  LensMass(xg::AbstractArray, yg::AbstractArray;
         b::Real, q::Real, s::Real=1e-4, xcentre::Real=0., ycentre::Real=0.)
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
        
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        psi = sqrt.(q.^2 .* (s.^2 .+ xsh.^2) .+ ysh.^2)

        q = min.(q, 0.9999)

        f_x, f_y = LensDerivative(xsh, ysh; b=b, s=s, q=q, varphi=0.)
        
        f = xsh .* f_x .+ ysh .* f_y .- b .* s .* 1.0 ./ 2.0 .* log.((psi .+ s).^2 .+ (1.0 .- q.^2) .* xsh.^2)

        return f

    end


    function  LensDerivative(xg::AbstractArray, yg::AbstractArray;
         b::Real=1.5, q::Real=0.8, varphi::Real=pi/6 ,
            s::Real=1e-4, xcentre::Real=0., ycentre::Real=0.)
    #=
    The mass profile for non-singular isothermal ellipsoid (NIE)
    (Keeton and Kochanek 1998, https://arxiv.org/pdf/astro-ph/9705194.pdf)
    mass profile
    \kappa(x, y) = \frac{1}{2} \left(\frac{}{\sqrt{s^2 + q x^2 + y^2/q}} \right)
    with
    \theta_{E} is the (circularized) Einstein radius,
    q is the minor/major axis ratio,
    x, y are defined in a coordinate system aligned with the major and minor axis of the lens

    =#

    xsh = xg .- xcentre
    ysh = yg .- ycentre

    xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

    psi = sqrt.(q.^2 .* (s.^2 .+ xsh.^2) .+ ysh.^2)

    q = min.(q, 0.9999)

    psi = sqrt.(q.^2 .* (s.^2 .+ xsh.^2) .+ ysh.^2)

    f_x = b ./ sqrt.(1.0 .- q.^2) .* atan.(sqrt.(1.0 .- q.^2) .* xsh ./ (psi .+ s))

    f_y = b ./ sqrt.(1.0 .- q.^2) .* atanh.(sqrt.(1.0 .- q.^2) .* ysh ./ (psi .+ q.^2 .* s))

    f_x, f_y = LensUtils.LensRotation(f_x, f_y, varphi)

    return f_x , f_y 

    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
         b::Real=1.5, q::Real=0.8, varphi::Real=pi/6, s::Real=1e-4,
            xcentre::Real=0., ycentre::Real=0., diff::Real= 1e-10)

    # shift
    xsh = xg .- xcentre
    ysh = yg .- ycentre
    # rotate
    xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

    # hessian in major axis
    f_x, f_y = LensDerivative(xsh, ysh; b=b, s=s, q=q, varphi=0.)
    f_x_dx, _ = LensDerivative(xsh .+ diff, ysh;b=b, s=s, q=q, varphi=0.)
    f_x_dy, f_y_dy = LensDerivative(xsh, ysh .+ diff; b=b, s=s, q=q, varphi=0.)

    f_xx = (f_x_dx .- f_x) ./ diff
    f_xy = (f_x_dy .- f_x) ./ diff
    f_yy = (f_y_dy .- f_y) ./ diff

    # rotate back
    kappa = @.  1.0 / 2 * (f_xx + f_yy)
    gamma1_ = @. 1.0 / 2 * (f_xx - f_yy)
    gamma2_ = f_xy
    gamma1 = @. cos(2 * varphi) * gamma1_ - sin(2 * varphi) * gamma2_
    gamma2 = @. sin(2 * varphi) * gamma1_ + cos(2 * varphi) * gamma2_
    f_xx = @. kappa + gamma1
    f_yy = @. kappa - gamma1
    f_xy = gamma2
    return f_xx, f_xy, f_yy

    end

end