module NIE

    include("./NIEkappa.jl")
    using ...LensUtils: e2phiq


    function LensCheck(; theta_E::Real, s_scale::Real, e1::Real, e2::Real,
                         xcentre::Real=0., ycentre::Real=0.)

        para = [theta_E, s_scale, e1, e2, xcentre, ycentre]

        if all([0., 0., -0.5, -0.5, -100, -100]< para) && all([10, 100, 0.5, 0.5, 100, 100]>para)
            return
        else
            error("The NIE configuration is out of range!")        
        end
    end

    function Main2MajorAxes(theta_E::Real, s_scale::Real, e1::Real, e2::Real)
        q, varphi = LensUtils.e2phiq(e1, e2)
        theta_E = @. theta_E / (sqrt((1.0 + q^2) / (2.0 * q)))
        b = @. theta_E * sqrt((1 + q^2) / 2)
        s = @. s_scale * sqrt((1 + q^2) / (2*q^2))
        q = min.(q, 0.9999)
        return Dict(:b=>b, :s=>s, :q=>q, :varphi=>varphi)
    end

    function LensPotential(xg::AbstractArray, yg::AbstractArray;
             theta_E::Real, s_scale::Real, e1::Real, e2::Real,
                xcentre::Real=0., ycentre::Real=0.)
        #=
            The mass distribution of NIE (Keeton and Kochanek 1998, https://arxiv.org/pdf/astro-ph/9705194.pdf)
 
            theta_E is the Einstein radius,
            s_scale is the core radius,
            e1, e2 are 
            xg,yg are mesh of the x,y axeses 
        
        =#

        para_ma = Main2MajorAxes(theta_E, s_scale, e1, e2)

        f = NIEkappa.LensPotential(xg, yg; xcentre=xcentre, ycentre=ycentre, para_ma...)
        
        return f

    end

    function  LensDerivative(xg::AbstractArray, yg::AbstractArray;
                    theta_E::Real, s_scale::Real, e1::Real, e2::Real,
                    xcentre::Real=0., ycentre::Real=0.)

        para_ma = Main2MajorAxes(theta_E, s_scale, e1, e2)

        f_x , f_y = NIEkappa.LensDerivative(xg, yg; xcentre=xcentre, ycentre=ycentre, para_ma...)
        
        return f_x , f_y 

    end

    function LensHessian(xg::AbstractArray, yg::AbstractArray;
        theta_E::Real, s_scale::Real, e1::Real, e2::Real,
        xcentre::Real=0., ycentre::Real=0., diff::Real= 1e-10)

        para_ma = Main2MajorAxes(theta_E, s_scale, e1, e2)

        f_xx, f_xy, f_yy = NIEkappa.LensHessian(xg, yg; xcentre=xcentre, ycentre=ycentre, diff=diff, para_ma...)
        
        return f_xx, f_xy, f_yy

    end

end