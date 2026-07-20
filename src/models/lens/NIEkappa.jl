"""
    NIEkappa — NIE in major-axis coordinates

Core implementation of the non-singular isothermal ellipsoid in its
major-axis frame.  Parameters are already in the b/q/φ convention
(see NIE.jl for the e1/e2 interface).  Use NIE instead unless you
need major-axis parameters directly.

# Parameters
- `b`: Einstein radius in major-axis frame [arcsec]
- `s`: core radius [arcsec]
- `q`: minor-to-major axis ratio (0 < q ≤ 1)
- `varphi`: position angle of major axis [rad]
- `xcentre`, `ycentre`: lens centre [arcsec]

# Reference
Keeton & Kochanek (1998), arXiv:astro-ph/9705194
"""
module NIEkappa

    using Jens.LensUtils
    using Distributions

    function LensCheck(b::Real, s::Real, q::Real, varphi::Real,
                         xcentre::Real=0., ycentre::Real=0.)

        para = [b, s, q, varphi, xcentre, ycentre]

        if all([0., 0., 0., 0., -100, -100]< para) && all([10, 100, 1, 2*pi, 100, 100]>para)
            return
        else
            error("The NIE configuration is out of range!")        
        end
    end

    function LensPotential(xg::AbstractArray, yg::AbstractArray;
          b::Real=1.5, s::Real=0.1, q::Real=0.8,
            varphi::Real=pi/6, xcentre::Real=0., ycentre::Real=0.)
        #=
            The mass distribution of NIE (Keeton and Kochanek 1998, https://arxiv.org/pdf/astro-ph/9705194.pdf)
 
            b is the (circularized) Einstein radius,
            q is the minor/major axis ratio,
            s is the core radius,
            varphi is the angle of major axes,
            xg,yg are mesh of the x,y axeses 
        
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
          b::Real=1.5, s::Real=0.1, q::Real=0.8, 
            varphi::Real=pi/6., xcentre::Real=0., ycentre::Real=0.)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        psi = sqrt.(q.^2. .* (s.^2. .+ xsh.^2.) .+ ysh.^2.)

        q = min.(q, 0.9999)

        psi = sqrt.(q.^2. .* (s.^2. .+ xsh.^2.) .+ ysh.^2.)
        
        f_x = b ./ sqrt.(1.0 .- q.^2.) .* atan.(sqrt.(1.0 .- q.^2.) .* xsh ./ (psi .+ s))

        f_y = b ./ sqrt.(1.0 .- q.^2.) .* atanh.(sqrt.(1.0 .- q.^2.) .* ysh ./ (psi .+ q.^2. .* s))

        f_x, f_y = LensUtils.LensRotation(f_x, f_y, varphi)

        return f_x , f_y 

    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
          b::Real=1.5, s::Real=0.1, q::Real=0.8, varphi::Real=pi/6,
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
        kappa = @.  1.0 / 2. * (f_xx + f_yy)
        gamma1_ = @. 1.0 / 2. * (f_xx - f_yy)
        gamma2_ = f_xy
        gamma1 = @. cos(2. * varphi) * gamma1_ + sin(2. * varphi) * gamma2_
        gamma2 = @. - sin(2. * varphi) * gamma1_ + cos(2. * varphi) * gamma2_
        f_xx = @. kappa + gamma1
        f_yy = @. kappa - gamma1
        f_xy = gamma2
        return f_xx, f_xy, f_yy

    end

end