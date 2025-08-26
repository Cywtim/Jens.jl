module EPL

    using   HypergeometricFunctions

    include("../LensUtils.jl")


    # DOI 		https://doi.org/10.1051/0004-6361/201526773 
    function LensCheck(; theta_E::Real, gamma::Real,
         e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        para = [theta_E, gamma, e1, e2, xcentre, ycentre]

        if all([0.,1.5,-0.5,-0.5 -100, -100]< para) && all([10,2.5, 0.5,  0.5, 100, 100]>para)
            return
        else
            error("The EPL configuration is out of range!")        
        end
    end

    function Main2MajorAxes(theta_E::Real, gamma::Real,
        e1::Real, e2::Real)
        t = @. gamma - 1
        varphi, q = LensUtils.e2phiq(e1, e2)
        b = @. theta_E * sqrt(q)
        return b, t, q, varphi
        
    end

    function MajorDerivative(xsh::AbstractArray, ysh::AbstractArray;
                         b::Real , t::Real ,q::Real)

        Zreal = @. q * xsh
        Zimag = ysh
        Z = @. Zreal + Zimag *im
        R = @. abs(Z)
        R = @. max(R, 1e-10)

        _2F1(x) = HypergeometricFunctions.pFq((1., t / 2), (2 - t / 2, ), -(1 - q) / (1 + q) * (x / conj(x)))

        hf = _2F1.(Z)
        omega = @. Z * hf

        alpha = @. 2 / (1 + q) * (b / R)^t * omega
 
        alpha_x = @. real(alpha)
        # np.nan_to_num
        alpha_y = @. imag(alpha)
        # np.nan_to_num

        return alpha_x, alpha_y
    end

    function  LensPotential(xg::AbstractArray, yg::AbstractArray; theta_E::Real, gamma::Real,
        e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        b, t, q, varphi = Main2MajorAxes(theta_E::Real, gamma::Real,
        e1::Real, e2::Real)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        alpha_x, alpha_y = MajorDerivative(xsh, ysh; b, t, q)

        f = @. (xsh * alpha_x + ysh * alpha_y) / (2 - gamma)
        
        return f
    end

    function  LensDerivative(xg::AbstractArray, yg::AbstractArray; theta_E::Real, gamma::Real,
        e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        b, t, q, varphi = Main2MajorAxes(theta_E::Real, gamma::Real,
        e1::Real, e2::Real)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        f_x, f_y = MajorDerivative(xsh, ysh; b, t, q)
        
        return f_x, f_y 
        
    end

    function  LensHessian(xg::AbstractArray, yg::AbstractArray; theta_E::Real, gamma::Real,
        e1::Real, e2::Real, xcentre::Real=0., ycentre::Real=0.)

        b, t, q, varphi = Main2MajorAxes(theta_E::Real, gamma::Real,
        e1::Real, e2::Real)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
        
        xsh, ysh = LensUtils.LensRotation(xsh, ysh, -varphi)

        alpha_x, alpha_y = MajorDerivative(xsh, ysh; b, t, q)

        R = @. sqrt((q * xsh)^2 +  ysh^2)
        R = @. max(R, 1e-10)
        r = @. sqrt(xsh^2 + ysh^2)

        cos = x ./ r
        sin = y ./ r
        cos2 = @. cos * cos * 2 - 1
        sin2 = @. sin * cos * 2

        kappa = @. (2 - t) / 2 * (b / R)^t
        #kappa = np.nan_to_num
        
        gamma_1 = @. (1 - t) * (alpha_x * cos - alpha_y * sin) / r - kappa * cos2
        gamma_2 = @. (1 - t) * (alpha_y * cos + alpha_x * sin) / r - kappa * sin2
        #gamma_1 = np.nan_to_num
        #gamma_2 = np.nan_to_num

        f_xx = kappa .+ gamma_1
        f_yy = kappa .- gamma_1
        f_xy = gamma_2

        return f_xx, f_xy, f_xy, f_yy

    end


end