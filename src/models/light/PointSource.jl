"""
    PointSource(; flux=1.0, beta_x=0.0, beta_y=0.0)

A point source at source-plane position `(beta_x, beta_y)` [arcsec].

Use `LensGenerator.add_point` to render it through a lens model.

# Example
    ps = PointSource(flux=100.0, beta_x=0.3, beta_y=-0.1)
    img = add_point(li, lens, ps.flux, ps.beta_x, ps.beta_y; psf=...)
"""
struct PointSource
    flux::Float64
    beta_x::Float64
    beta_y::Float64
end

function PointSource(; flux::Real=1.0, beta_x::Real=0.0, beta_y::Real=0.0)
    return PointSource(Float64(flux), Float64(beta_x), Float64(beta_y))
end