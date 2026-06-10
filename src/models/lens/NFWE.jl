module NFWE

    include("./NFW.jl")
    using Jens.LensUtils

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
                                                                        

    function LensHessian(x, y; Rs, alpha_Rs, e1, e2,
                          xcentre=0., ycentre=0., diff::Real=1e-10)
        f_x, f_y = LensDerivative(x, y; Rs, alpha_Rs, e1, e2, xcentre, ycentre)
        f_x_dx, _      = LensDerivative(x .+ diff, y;      Rs, alpha_Rs, e1, e2, xcentre, ycentre)
        f_x_dy, f_y_dy = LensDerivative(x,      y .+ diff; Rs, alpha_Rs, e1, e2, xcentre, ycentre)

        f_xx = (f_x_dx .- f_x) ./ diff
        f_xy = (f_x_dy .- f_x) ./ diff
        f_yy = (f_y_dy .- f_y) ./ diff

        return f_xx, f_xy, f_yy
    end

end
