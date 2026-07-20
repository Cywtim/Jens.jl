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
import Jens.LensMask: _coords
import Jens.LensShapelet: ShapeletBasis, n_basis,
    evaluate_basis, build_design!, shapelet_2d, _source_plane

export GridGPU, GenGrid_GPU

# ═══════════════════════════════════════════════════════════════
#  GridGPU — GPU observation grid
#
#  Mirrors the CPU Grid struct but with CuArray{Float32} coordinates.
#  All lens models (SIS, SIE, MGE, CombinedLens, ...) work on GridGPU
#  without modification — they dispatch on AbstractArray.
# ═══════════════════════════════════════════════════════════════

struct GridGPU{T<:AbstractFloat, A<:AbstractMatrix{T}}
    pix_n::Int
    pix_size::T
    xg::A
    yg::A
end

"""
    grid = GenGrid_GPU(; pix_n=256, pix_size=JFloat(0.09))

Build a GPU-resident observation grid.  Precision follows `Jens.JFloat`
(default Float32).  Pass `pix_size=0.09` for Float64.
"""
function GenGrid_GPU(; pix_n::Int=256, pix_size::Real=Jens.JFloat(0.09))
    T = typeof(pix_size)
    half = T(div(pix_n, 2) * pix_size)
    x = range(-half, half; length=pix_n + 1)
    xv = CUDA.cu(collect(x))
    xg, yg = ndgrid(xv, xv)
    return GridGPU(pix_n, T(pix_size), xg, yg)
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

# ═══════════════════════════════════════════════════════════════
#  Inject into parent namespace: fills Jens.gpu_grid placeholder
# ═══════════════════════════════════════════════════════════════
Jens.gpu_grid(; pix_n=256, pix_size::Real=Jens.JFloat(0.09)) =
    GenGrid_GPU(; pix_n, pix_size=Jens.JFloat(pix_size))

# ═══════════════════════════════════════════════════════════════
#  GPU randn_like — overload for CuArray
# ═══════════════════════════════════════════════════════════════
import Jens.LensUtils: randn_like
randn_like(x::CuArray, dims::Int...) = CUDA.randn(eltype(x), dims...)
randn_like(x::CuArray)              = CUDA.randn(eltype(x), size(x)...)

# ═══════════════════════════════════════════════════════════════
#  LensMask: GridGPU dispatch
# ═══════════════════════════════════════════════════════════════
_coords(grid::GridGPU) = grid.xg, grid.yg

# ═══════════════════════════════════════════════════════════════
#  GPU Hermite polynomials
# ═══════════════════════════════════════════════════════════════

"GPU Hermite: H_n(x) evaluated for all elements in parallel."
function _hermite_vec_gpu(n::Int, x::CuArray{<:AbstractFloat})
    T = eltype(x)
    n == 0 && return CUDA.ones(T, size(x))
    n == 1 && return T(2) .* x
    h_prev = CUDA.ones(T, size(x))
    h_curr = T(2) .* x
    for k in 2:n
        h_next = @. T(2) * x * h_curr - T(2(k - 1)) * h_prev
        h_prev = h_curr
        h_curr = h_next
    end
    return h_curr
end

# ═══════════════════════════════════════════════════════════════
#  Shapelet basis functions — GPU dispatch
# ═══════════════════════════════════════════════════════════════

function shapelet_2d(n1::Int, n2::Int, beta::Real,
                     x::CuArray{<:AbstractFloat}, y::CuArray{<:AbstractFloat})
    T = promote_type(typeof(beta), eltype(x), eltype(y))
    xb = x ./ T(beta)
    yb = y ./ T(beta)
    norm_inv = T(1) / sqrt(T(2)^(n1 + n2) * T(pi) *
                factorial(n1) * factorial(n2) * T(beta)^2)
    hx = _hermite_vec_gpu(n1, xb)
    hy = _hermite_vec_gpu(n2, yb)
    return @. norm_inv * hx * hy * exp(-(xb^2 + yb^2) / 2)
end

# ═══════════════════════════════════════════════════════════════
#  evaluate_basis — GPU dispatch
# ═══════════════════════════════════════════════════════════════

function evaluate_basis(basis::ShapeletBasis, x::CuArray, y::CuArray)
    N = n_basis(basis)
    M = length(x)
    A = CUDA.zeros(eltype(x), M, N)

    col = 1
    for n in 0:basis.n_max
        for n1 in 0:n
            n2 = n - n1
            A[:, col] .= vec(shapelet_2d(n1, n2, basis.beta, x, y))
            col += 1
        end
    end
    return A
end

# ═══════════════════════════════════════════════════════════════
#  build_design! — GPU dispatch
# ═══════════════════════════════════════════════════════════════

function build_design!(A_conv::CuArray, A_flat::CuArray,
                        sys, basis::ShapeletBasis, data, mask;
                        xc_src::Real=0.0, yc_src::Real=0.0)
    xg = sys.grid.xg
    yg = sys.grid.yg
    nx, ny = size(xg)
    N = n_basis(basis)

    # ── Step 1–2: ray-trace + centre shift (GPU broadcasting) ──
    betax, betay = _source_plane(sys, xg, yg)

    if xc_src != 0.0 || yc_src != 0.0
        T = eltype(xg)
        betax = betax .- T(xc_src)
        betay = betay .- T(yc_src)
    end

    # ── Step 3: evaluate basis on GPU ──
    # _source_plane returns Float64 (lens params are Float64), but we want
    # the grid's precision for the design matrix. Convert explicitly.
    T_grid = eltype(A_flat)
    betax32 = T_grid.(betax)
    betay32 = T_grid.(betay)
    A_tmp = evaluate_basis(basis, vec(betax32), vec(betay32))
    A_flat .= A_tmp

    # ── Step 4: PSF convolve each column (GPU FFT path) ──
    psf = sys.psf
    pixel_scale = eltype(xg)(sys.grid.pix_size)

    for col in 1:N
        img_col = reshape(view(A_flat, :, col), nx, ny)
        if psf !== nothing
            img_col = conv_psf(img_col, psf, pixel_scale)
        end
        A_conv[:, col] .= vec(img_col)
    end

    # ── Step 5: mask (materialize for cuBLAS compatibility) ──
    if mask !== nothing
        mask_cpu = mask isa CuArray ? Array(mask) : mask
        mask_idx = findall(vec(mask_cpu))
        A_masked = A_conv[mask_idx, :]   # copy, not view — GPU needs contiguous
    else
        mask_idx = collect(1:nx * ny)
        A_masked = A_conv
    end

    return A_masked, mask_idx
end

end # module JensCUDA
