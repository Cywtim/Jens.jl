# ═══════════════════════════════════════════════════════════════
#  AbstractLight — unified light-source type hierarchy
#
#  Analogous to AbstractLens: any light component (extended, point,
#  composite) shares the same render_lens interface.
#
#      AbstractLight
#       ├── ExtendedSource{P}    ← Function + baked NamedTuple params
#       ├── PointImage           ← {flux, beta_x, beta_y}
#       └── CompositeImage{T}    ← Tuple of AbstractLight items
#
#  See LensGenerator.render_lens for the rendering pipeline.
# ═══════════════════════════════════════════════════════════════

export AbstractLight, ExtendedSource, PointImage, CompositeImage
export evaluate_source


# ═══════════════════════════════════════════════════════════════
abstract type AbstractLight end


# ═══════════════════════════════════════════════════════════════
#  ExtendedSource — a surface-brightness profile on the source plane
#
#      host = ExtendedSource(SersicLight.SersicSpheric,
#                            (amp=1.0, Rsersic=0.5, n=4.0,
#                             xcentre=0.0, ycentre=0.0))
#
#  `profile` is any callable f(x, y; kwargs...) → array.
#  `params` are baked into the struct (no Dict needed).
# ═══════════════════════════════════════════════════════════════

struct ExtendedSource{F<:Function, P<:NamedTuple} <: AbstractLight
    profile::F
    params::P
end

function ExtendedSource(profile::Function; kwargs...)
    nt = (; (Symbol(k) => float(v) for (k, v) in kwargs)...)
    return ExtendedSource{typeof(profile), typeof(nt)}(profile, nt)
end


# ═══════════════════════════════════════════════════════════════
#  PointImage — a point source at source-plane position
#
#      agn = PointImage(flux=100.0, beta_x=0.3, beta_y=0.0)
#
#  `flux` is the total integrated flux.  Magnification is
#  applied by render_lens during ray-tracing.
# ═══════════════════════════════════════════════════════════════

struct PointImage <: AbstractLight
    flux::Float64
    beta_x::Float64
    beta_y::Float64
end

function PointImage(; flux::Real=1.0, beta_x::Real=0.0, beta_y::Real=0.0)
    return PointImage(float(flux), float(beta_x), float(beta_y))
end


# ═══════════════════════════════════════════════════════════════
#  CompositeImage — ordered tuple of AbstractLight sources
#
#      src = CompositeImage(host, agn, ring)
#
#  render_lens sums the contribution of each source in order.
#  The CompositeImage helper wraps varargs in a Tuple.
# ═══════════════════════════════════════════════════════════════

struct CompositeImage{T<:Tuple} <: AbstractLight
    sources::T
end

CompositeImage(srcs::AbstractLight...) = CompositeImage(srcs)


# ═══════════════════════════════════════════════════════════════
#  evaluate_source — evaluate light profile on a coordinate grid
#
#  Useful for testing or pre-render inspection.  Only meaningful
#  for ExtendedSource (returns flux array); PointImage and
#  CompositeImage throw MethodError.
# ═══════════════════════════════════════════════════════════════

function evaluate_source(src::ExtendedSource, x, y)
    return src.profile(x, y; src.params...)
end


# ═══════════════════════════════════════════════════════════════
#  _render_source — LensBase dispatch extension
#
#  Extended by LightModel so that LensRayShooting can accept
#  AbstractLight structs in addition to bare Functions.
# ═══════════════════════════════════════════════════════════════

import ..LensBase: _render_source

function _render_source(src::ExtendedSource, x, y; kwargs...)
    return src.profile(x, y; src.params...)
end