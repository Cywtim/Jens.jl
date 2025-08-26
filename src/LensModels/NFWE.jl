module NFWE

    include("./NFW.jl")
    include("../LensUtils.jl")

    function LensPotential(x, y; Rs, alpha_Rs, e1, e2, xcentre=0., ycentre=0.)

        xsh, ysh = LensUtils.EllipticalDistortion(x, y; e1, e2, xcentre, ycentre)
        f = NFW.LensPotential(xsh, ysh; Rs, alpha_Rs, xcentre=0., ycentre=0.)
        return f
    end

    function LensDerivative(x, y; Rs, alpha_Rs, e1, e2, xcentre=0., ycentre=0.)

        xsh, ysh = LensUtils.EllipticalDistortion(x, y; e1, e2, xcentre, ycentre)
        f_x, f_y = NFW.LensDerivative(xsh, ysh; Rs, alpha_Rs, xcentre=0, ycentre=0)
        return f_x, f_y
    end

    function LenHessian(x, y; Rs, alpha_Rs, e1, e2, xcentre=0., ycentre=0.)

        xsh, ysh = LensUtils.EllipticalDistortion(x, y; e1, e2, xcentre, ycentre)
        f_xx, f_xy, f_yy = NFW.LenHessian(xsh, ysh; Rs, alpha_Rs, xcentre=0., ycentre=0.)
        return f_xx, f_xy, f_yy
    end
end