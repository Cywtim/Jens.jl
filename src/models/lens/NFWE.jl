"""
    NFWE — Elliptical Navarro-Frenk-White

Elliptical extension of NFW via coordinate distortion.  Wraps NFW.jl:
replaces circular radius R with an elliptical radius via the
EllipticalDistortion transform.

# Parameters
- `Rs`: NFW scale radius [arcsec]
- `alpha_Rs`: deflection scale at Rs [arcsec]
- `e1`, `e2`: ellipticity components
- `xcentre`, `ycentre`: lens centre [arcsec]

# Reference
Golse & Kneib (2002), arXiv:astro-ph/0112138

# Example
    lens = SingleModel(NFWE; Rs=5.0, alpha_Rs=0.5, e1=0.15, e2=0.0)
"""
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
