"""
    PointSource(; flux=1.0, beta_x=0.0, beta_y=0.0)

Backward-compat alias for `PointImage`.  Use `PointImage` in new code
and `render_lens` instead of `add_point`.

# Example
    agn = PointSource(flux=100.0, beta_x=0.3, beta_y=-0.1)
    img = render_lens(agn, li, lens; psf=...)
"""
const PointSource = PointImage