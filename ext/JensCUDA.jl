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
using CUDA.CUFFT
using Jens
using Jens.LensUtils: ndgrid
using Jens.LensGenerator: Grid, GenGrid
using Jens.LensBase: lens_derivative
using Jens.LensPSF
using Jens.LightModel: ExtendedSource, evaluate_source
import Jens.LensGenerator: render_lens
import Jens.LensPSF: conv_psf, AbstractPSF, make_kernel, _default_half

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
        result = conv_psf(result, psf, pixel_scale)
    end
    return result
end

# ═══════════════════════════════════════════════════════════════
#  GPU-native PSF convolution — FFT-based
#
#  Overrides the CPU fallback in LensPSF.conv_psf for CuArray
#  images. Uses CUFFT for element-wise frequency-domain
#  multiplication, avoiding host↔device transfers.
# ═══════════════════════════════════════════════════════════════

"""
    _fft_conv_same(image::CuArray{T,2}, kernel::AbstractMatrix) → CuArray{T,2}

FFT-based "same" convolution. Pads both arrays, FFTs, multiplies,
IFFTs, and crops to the original image size. The kernel is assumed
to be centred (odd size).
"""
function _fft_conv_same(image::CuArray{T,2}, kernel::AbstractMatrix) where T
    M, N = size(image)
    K = size(kernel, 1)  # assume square kernel
    @assert K == size(kernel, 2) && isodd(K) "kernel must be square and odd-sized"

    # Full-convolution padded size
    P_h, P_w = M + K - 1, N + K - 1

    # Pad image to padded size
    img_pad = CUDA.zeros(T, P_h, P_w)
    img_pad[1:M, 1:N] .= image

    # Pad flipped kernel to same size, placed at top-left
    k_flip_cpu = reverse(reverse(kernel; dims=1); dims=2)
    k_flip = CuArray{T}(k_flip_cpu)
    kern_pad = CUDA.zeros(Complex{T}, P_h, P_w)
    kern_pad[1:K, 1:K] .= k_flip

    # FFT → multiply → IFFT
    F_img  = fft(img_pad)
    F_kern = fft(kern_pad)
    full   = real(ifft(F_img .* F_kern))

    # Crop "same" region: kernel centre → (kc, kc) where kc = (K+1)÷2
    kc = (K + 1) ÷ 2
    return full[kc:kc+M-1, kc:kc+N-1]
end

# Override conv_psf for CuArray images — GPU-native FFT path
function conv_psf(image::CuArray{T,2}, psf::AbstractPSF,
                  pixel_scale::Real; half::Int=0) where T
    h = half > 0 ? half : _default_half(psf, pixel_scale)
    kernel = make_kernel(psf, pixel_scale, h)
    return _fft_conv_same(image, kernel)
end

end # module JensCUDA
