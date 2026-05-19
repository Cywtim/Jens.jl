module NFW
    
    # 10.48550/arXiv.astro-ph/9602053 ***
    # https://arxiv.org/abs/astro-ph/9611107)
    # https://doi.org/10.1051/0004-6361/202346308  eq(3),
    # https://doi.org/10.1046/j.1365-8711.2003.06276.x  

    max_r_rs = 10
    num_interp = 1000
    interpol = false

    function LensCheck(Rs, alpha_Rs, xcentre=0., ycentre=0.)
        
    end

    function alpha2rho0(alpha_Rs::Real, Rs::Real)

        rho0 = @. alpha_Rs / (4.0 * Rs^2 * (1.0 + log(1.0 / 2.0)))
        return rho0

    end

    function potential(R::Real, Rs::Real, rho0::Real)
        
        r_rs = @. R / Rs
        hx = h.(r_rs)
        p = @. 2 * r_rs * Rs^3 * hx
        return p
    end

    function h(r_rs)
        e = 1e-6
        if r_rs < 1
            rrs = max(e, r_rs)
            a = @. log(rrs / 2.0) + 1 / sqrt(1 - rrs^2) * acosh(1.0 / rrs)
        elseif r_rs == 1
            a = @. 1 + log(1.0 / 2.0)
        else  # r_rs > 1:
            a = @. log(r_rs / 2) + 1 / sqrt(r_rs^2 - 1) * acos(1.0 / r_rs)
        end
        
        return a
    end

    function LensPotential(x, y; Rs, alpha_Rs, xcentre=0., ycentre=0.)

        rho0 = alpha2rho0(alpha_Rs, Rs)
        Rs = max(Rs, 1e-6)

        xsh = @. x - xcentre
        ysh = @. y - ycentre
        R = @. sqrt(xsh^2 + ysh^2)
        f = potential(R, Rs, rho0)
        return f
    end


    function alpha(R, Rs, rho0)
        R = max.(R, 1e-6)
        r_rs = @. R / Rs
        gx = g.(r_rs)
        a = @. 4 * rho0 * Rs * gx / r_rs^2
        return a 

    end

    function g(r_rs)
        c = 1e-6
        if r_rs < 1
            r_rs = max(c, r_rs)
            a = @. log(r_rs / 2.0) + 1 / sqrt(1 - r_rs^2) * acosh(1.0 / r_rs)
        elseif r_rs == 1
            a = @. 1 + log(1.0 / 2.0)
        else  # r_rs > 1:
            a = @. log(r_rs / 2) + 1 / sqrt(r_rs^2 - 1) * acos(1.0 / r_rs)
        end
    end


    function LensDerivative(x, y; Rs, alpha_Rs, xcentre=0., ycentre=0.)

        rho0 = alpha2rho0(alpha_Rs, Rs)
        Rs = max(Rs, 1e-6)

        xsh = @. x - xcentre
        ysh = @. y - ycentre
        R = @. sqrt(xsh^2 + ysh^2)

        a = alpha(R, Rs, rho0)
        f_x = @. a * xsh
        f_y = @. a * ysh

        return f_x, f_y

    end

    function kappa(x, y, Rs, rho0, xcentre=0., ycentre=0.)
        xsh = @. x - xcentre
        ysh = @. y - ycentre
        R = @. sqrt(xsh^2+ysh^2)
        r_rs = @. R / Rs
        Fx = f.(r_rs)
        kappa = @. 2 * rho0 * Rs * Fx
        return kappa
    end

    function gamma(x, y, R, Rs, rho0, xcentre=0., ycentre=0.)

        c = 1e-8
        R = max.(R, c)
        r_rs = @. R / Rs
        gx = g.(r_rs)
        Fx = f.(r_rs)
        a = 2 * rho0 * Rs * (2 * gx / r_rs^2 - Fx)
        shear1 = @. a * (y^2 - x^2) / R^2
        shear2 = @. -a * 2 * (x * y) / R^2

        return shear1, shear2
    end

    function f(r_rs)

        if (r_rs < 1) && (r_rs > 0)
            a = 1 / (r_rs^2 - 1) * (1 - 2 / sqrt(1 - r_rs^2) * atanh(sqrt((1 - r_rs) / (1 + r_rs))))
    
        elseif r_rs == 1
            a = 1.0 / 3
        elseif r_rs > 1
            a = 1 / (r_rs^2 - 1) * ( 1 - 2 / sqrt(r_rs^2 - 1) * atan(sqrt((r_rs - 1) / (1 + r_rs))))

        else  # r_rs == 0:
            c = 1e-8
            a = 1 / (-1) * (1 - 2 / sqrt(1) * atanh(sqrt((1 - c) / (1 + c))))
        end
    end


    function LenHessian(x, y; Rs, alpha_Rs, xcentre=0., ycentre=0.)

        rho0 = alpha2rho0(alpha_Rs, Rs)
        Rs = max(Rs, 1e-6)

        xsh = @. x - xcentre
        ysh = @. y - ycentre
        R = @. sqrt(xsh^2 + ysh^2)
        # kappa
        kappa0 = kappa(R, 0, Rs, rho0)

        # gamma
        gamma1, gamma2 = gamma(xsh,ysh,R, Rs, rho0)
        f_xx = @. kappa0 + gamma1
        f_yy = @. kappa0 - gamma1
        f_xy = @. gamma2
        return f_xx, f_xy, f_yy
 
    end

end