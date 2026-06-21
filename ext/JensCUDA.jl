# JensCUDA — GPU extension for Jens.jl
#
# Loaded automatically when the user does `using CUDA` after `using Jens`.
# Provides:
#   - GridGPU struct
#   - GenGrid_GPU constructor
#   - render_lens(src::ExtendedSource, grid::GridGPU, ...)
#
# CPU functionality in Jens.jl has zero CUDA dependency.

module JensCUDA

using CUDA
using Jens
using Jens.LensUtils: ndgrid
using Jens.LensGenerator: Grid, GenGrid
using Jens.LensBase: lens_derivative
using Jens.LensPSF
using Jens.LightModel: ExtendedSource, evaluate_source
import Jens.LensGenerator: render_lens

export GridGPU, GenGrid_GPU

# ═══════════════════════════════════════════════════════════════
#  GridGPU — GPU observation grid
#
#  Mirrors the CPU Grid struct but with CuArray{Float32} coordinates.
#  All lens models (SIS, SIE, MGE, CombinedLens, ...) work on GridGPU
#  without modification — they dispatch on AbstractArray.
# ═══════════════════════════════════════════════════════════════

struct GridGPU{A<:AbstractMatrix{Float32}}
    pix_n::Int
    pix_size::Float32
    xg::A
    yg::A
end

"""
    grid = GenGrid_GPU(; pix_n=256, pix_size=0.09f0)

Build a GPU-resident observation grid. The coordinate arrays
`xg`, `yg` are `CuArray{Float32}` ready for lens equation
evaluation on GPU.

# Example
```julia
using CUDA, Jens

grid = GenGrid_GPU(pix_n=256, pix_size=0.09f0)
cl = CombinedLens(SIS => (theta_E=0.8f0, xcentre=0f0, ycentre=0f0))
img = render_lens(source, grid, cl)  # runs entirely on GPU
```
"""
function GenGrid_GPU(; pix_n::Int=256, pix_size::Real=0.09f0)
    half = Float32(div(pix_n, 2) * pix_size)
    x = range(-half, half; length=pix_n + 1)
    xv = CUDA.cu(collect(x))
    xg, yg = ndgrid(xv, xv)
    return GridGPU(pix_n, Float32(pix_size), xg, yg)
end

# ═══════════════════════════════════════════════════════════════
#  render_lens for GridGPU — GPU ray-tracing pipeline
#
#  Identical logic to the CPU version, but dispatches on GridGPU
#  so the CuArray coordinates flow through lens_derivative →
#  source evaluation → optional PSF, all on GPU.
# ═══════════════════════════════════════════════════════════════

function render_lens(src::ExtendedSource, grid::GridGPU,
                     lens_model; psf=nothing, z_source=nothing, kwargs...)

    xg, yg = grid.xg, grid.yg
    pixel_scale = grid.pix_size

    alphax, alphay = lens_derivative(lens_model, xg, yg; z_source=z_source, kwargs...)
    betax = xg .- alphax
    betay = yg .- alphay

    result = evaluate_source(src, betax, betay)

    if psf !== nothing
        result = LensPSF.conv_psf(result, psf, pixel_scale)
    end
    return result
end

end # module JensCUDA
