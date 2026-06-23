module LensPSF

    # ═══════════════════════════════════════════════════════════════
    #  AbstractPSF — unified PSF interface
    #
    #  All PSF types implement:
    #    make_kernel(psf, pixel_scale, half) -> Matrix{Float64}
    #
    #  where:
    #    pixel_scale : image pixel scale [arcsec/pixel]
    #    half        : half-size of the output kernel [pixels]
    #  Returns a normalised (sum=1), centred kernel of size
    #  (2*half+1) × (2*half+1).
    # ═══════════════════════════════════════════════════════════════

    using PSFModels, ImageFiltering, Optim, SpecialFunctions

    export AbstractPSF
    export GaussianPSF, MoffatPSF, AiryDiskPSF, KernelPSF
    export make_kernel, conv_psf
    export render_point!, evaluate_psf_at

    # ═══════════════════════════════════════════════════════════════
    #  Abstract type
    # ═══════════════════════════════════════════════════════════════

    "Abstract supertype for all PSF models."
    abstract type AbstractPSF end

    # ═══════════════════════════════════════════════════════════════
    #  Interface
    # ═══════════════════════════════════════════════════════════════

    """
        K = make_kernel(psf::AbstractPSF, pixel_scale, half)

    Build a normalised (sum=1), centred PSF kernel of size
    `(2*half+1) × (2*half+1)` at the given `pixel_scale` [arcsec/pixel].
    """
    function make_kernel end

    """
        img = conv_psf(image, psf::AbstractPSF, pixel_scale; half)

    Convolve `image` with the PSF kernel. `half` defaults to
    `ceil(Int, 5 * fwhm_pix(psf, pixel_scale))`.
    """
    function conv_psf(image::AbstractMatrix, psf::AbstractPSF, pixel_scale::Real; half::Int=0)
        h = half > 0 ? half : _default_half(psf, pixel_scale)
        kernel = make_kernel(psf, pixel_scale, h)
        T = eltype(image)
        # ImageFiltering.imfilter does not support GPU kernels natively
        # (factorkernel triggers scalar indexing). Convolve on CPU and
        # copy back — kernel is tiny, image convolution is cheap relative
        # to the lens equation.
        img_cpu = Array(image)
        k_cpu = convert(Matrix{T}, kernel)
        result = imfilter(img_cpu, centered(k_cpu))
        out = similar(image, T, size(result))
        copyto!(out, result)
        return out
    end

    # ═══════════════════════════════════════════════════════════════
    #  Gaussian PSF
    # ═══════════════════════════════════════════════════════════════

    """
        GaussianPSF(; fwhm=5.0)

    Circular Gaussian PSF parametrised by FWHM [pixels].
    """
    struct GaussianPSF{T<:Real} <: AbstractPSF
        fwhm::T
    end
    GaussianPSF(fwhm::Real) = GaussianPSF{typeof(fwhm)}(fwhm)
    GaussianPSF(; fwhm::Real=5.0) = GaussianPSF(fwhm)

    function make_kernel(psf::GaussianPSF, pixel_scale::Real, half::Int)
        fwhm_pix = psf.fwhm / pixel_scale
        xs = -half:half
        ys = -half:half
        k = [PSFModels.gaussian(px, py; fwhm=fwhm_pix, x=0, y=0) for px in xs, py in ys]
        return k ./ sum(k)
    end

    function _default_half(psf::GaussianPSF, pixel_scale)
        return ceil(Int, 3.0 * psf.fwhm / pixel_scale)
    end

    # ═══════════════════════════════════════════════════════════════
    #  Moffat PSF
    # ═══════════════════════════════════════════════════════════════

    """
        MoffatPSF(; fwhm=5.0, alpha=3.0)

    Moffat profile PSF. Larger `alpha` → lighter wings.
    """
    struct MoffatPSF{T<:Real} <: AbstractPSF
        fwhm::T
        alpha::T
    end
    MoffatPSF(fwhm::Real, alpha::Real) = MoffatPSF{promote_type(typeof(fwhm), typeof(alpha))}(fwhm, alpha)
    MoffatPSF(; fwhm::Real=5.0, alpha::Real=3.0) = MoffatPSF(fwhm, alpha)

    function make_kernel(psf::MoffatPSF, pixel_scale::Real, half::Int)
        fwhm_pix = psf.fwhm / pixel_scale
        xs = -half:half
        ys = -half:half
        k = [PSFModels.moffat(px, py; fwhm=fwhm_pix, alpha=psf.alpha, x=0, y=0) for px in xs, py in ys]
        return k ./ sum(k)
    end

    function _default_half(psf::MoffatPSF, pixel_scale)
        return ceil(Int, 5.0 * psf.fwhm / pixel_scale)  # Moffat wings
    end

    # ═══════════════════════════════════════════════════════════════
    #  Airy disk PSF
    # ═══════════════════════════════════════════════════════════════

    """
        AiryDiskPSF(; fwhm=5.0)

    Diffraction-limited Airy disk PSF.
    """
    struct AiryDiskPSF{T<:Real} <: AbstractPSF
        fwhm::T
    end
    AiryDiskPSF(fwhm::Real) = AiryDiskPSF{typeof(fwhm)}(fwhm)
    AiryDiskPSF(; fwhm::Real=5.0) = AiryDiskPSF(fwhm)

    function make_kernel(psf::AiryDiskPSF, pixel_scale::Real, half::Int)
        fwhm_pix = psf.fwhm / pixel_scale
        xs = -half:half
        ys = -half:half
        k = [PSFModels.airydisk(px, py; fwhm=fwhm_pix, x=0, y=0) for px in xs, py in ys]
        return k ./ sum(k)
    end

    function _default_half(psf::AiryDiskPSF, pixel_scale)
        return ceil(Int, 5.0 * psf.fwhm / pixel_scale)  # Airy rings
    end

    # ═══════════════════════════════════════════════════════════════
    #  KernelPSF — user-provided PSF
    # ═══════════════════════════════════════════════════════════════

    """
        KernelPSF(kernel, pixel_scale)

    User-provided PSF from a precomputed kernel matrix and its
    intrinsic pixel scale [arcsec/pixel].

    If the kernel's pixel scale differs from the target image,
    `make_kernel` will resample (linear interpolation).
    """
    struct KernelPSF{T<:Real} <: AbstractPSF
        kernel::Matrix{T}
        pixel_scale::T   # [arcsec/pixel] — intrinsic scale of the kernel
    end

    function make_kernel(psf::KernelPSF, pixel_scale::Real, half::Int)
        k = copy(psf.kernel)
        k ./= sum(k)  # normalise copy

        if abs(pixel_scale - psf.pixel_scale) < 1e-6
            # Same scale — centre-crop or zero-pad to (2h+1)×(2h+1)
            return _resize_centred(k, half)
        end

        # Different scale — resample
        scale_factor = psf.pixel_scale / pixel_scale
        return _resample_kernel(k, scale_factor, half)
    end

    function _default_half(psf::KernelPSF, pixel_scale)
        return size(psf.kernel, 1) ÷ 2
    end

    "Centre-crop or zero-pad a kernel to the target half-size."
    function _resize_centred(kernel::AbstractMatrix, target_half::Int)
        sz = size(kernel, 1)
        cur_half = sz ÷ 2
        out_sz = 2 * target_half + 1
        out = zeros(eltype(kernel), out_sz, out_sz)

        copy_half = min(cur_half, target_half)
        c0 = sz ÷ 2 + 1
        o0 = out_sz ÷ 2 + 1

        rng_in  = (c0-copy_half):(c0+copy_half)
        rng_out = (o0-copy_half):(o0+copy_half)
        out[rng_out, rng_out] .= kernel[rng_in, rng_in]
        return out ./ sum(out)
    end

    "Resample a kernel by a given scale factor (bilinear interpolation)."
    function _resample_kernel(kernel::AbstractMatrix, scale_factor::Real, target_half::Int)
        sz_in = size(kernel, 1)
        out_sz = 2 * target_half + 1
        T = eltype(kernel)

        # Build coordinate grids for output → input mapping
        c_out = out_sz ÷ 2 + 1
        c_in  = sz_in ÷ 2 + 1

        out = zeros(T, out_sz, out_sz)
        for j_out in 1:out_sz, i_out in 1:out_sz
            # Map output pixel centre → input fractional coordinate
            x_in = c_in + (i_out - c_out) / scale_factor
            y_in = c_in + (j_out - c_out) / scale_factor

            # Bilinear interpolation
            ix0 = floor(Int, x_in)
            iy0 = floor(Int, y_in)
            ix1 = ix0 + 1
            iy1 = iy0 + 1

            if ix0 < 1 || ix1 > sz_in || iy0 < 1 || iy1 > sz_in
                out[i_out, j_out] = zero(T)
                continue
            end

            wx = x_in - ix0
            wy = y_in - iy0

            out[i_out, j_out] =
                (1-wx)*(1-wy) * kernel[ix0, iy0] +
                wx*(1-wy)     * kernel[ix1, iy0] +
                (1-wx)*wy     * kernel[ix0, iy1] +
                wx*wy         * kernel[ix1, iy1]
        end

        return out ./ sum(out)
    end

    # ═══════════════════════════════════════════════════════════════
    #  Legacy API — convenience constructors (direct kernel, no struct)
    # ═══════════════════════════════════════════════════════════════

    _fwhm_half(fwhm, scale=4.0) = ceil(Int, scale * fwhm / 2)

    function _make_grid(half::Int)
        return -half:half, -half:half
    end

    function gaussian_kernel(; fwhm::Real=5.0, half::Int=0)
        h = half > 0 ? half : _fwhm_half(fwhm, 3.0)
        xs, ys = _make_grid(h)
        k = [PSFModels.gaussian(px, py; fwhm) for px in xs, py in ys]
        return k ./ sum(k)
    end

    function moffat_kernel(; fwhm::Real=5.0, alpha::Real=3.0, half::Int=0)
        h = half > 0 ? half : _fwhm_half(fwhm, 5.0)
        xs, ys = _make_grid(h)
        k = [PSFModels.moffat(px, py; fwhm, alpha) for px in xs, py in ys]
        return k ./ sum(k)
    end

    function airydisk_kernel(; fwhm::Real=5.0, half::Int=0)
        h = half > 0 ? half : _fwhm_half(fwhm, 5.0)
        xs, ys = _make_grid(h)
        k = [PSFModels.airydisk(px, py; fwhm) for px in xs, py in ys]
        return k ./ sum(k)
    end

    # ═══════════════════════════════════════════════════════════════
    #  FitPSF — unchanged from original
    # ═══════════════════════════════════════════════════════════════

    export FitPSF, ApplyPSF

    function ApplyPSF(image::AbstractMatrix, kernel::AbstractMatrix)
        return imfilter(image, centered(kernel))
    end

    function FitPSF(data::AbstractMatrix{<:Real};
            model::Symbol = :gaussian,
            fwhm0::Real = 3.0,
            alpha0::Real = 3.0,
        )

        model in (:gaussian, :moffat, :airy) ||
            error("model must be :gaussian, :moffat, or :airy, got :$model")

        ny, nx = size(data)
        xs = collect(Float64, axes(data, 2))
        ys = collect(Float64, axes(data, 1))

        amp0  = maximum(data) - minimum(data)
        x0, y0 = nx / 2, ny / 2
        bkg0  = minimum(data)

        p0 = (model == :moffat ?
             [fwhm0, alpha0, amp0, x0, y0, bkg0] :
             [fwhm0, amp0, x0, y0, bkg0])

        function loss(p)
            s = 0.0
            if model == :gaussian
                σ  = p[1] / 2.355f0
                a, cx, cy, bg = p[2], p[3], p[4], p[5]
                for j in eachindex(ys), i in eachindex(xs)
                    d = a * exp(-((xs[i]-cx)^2 + (ys[j]-cy)^2) / (2σ^2)) + bg - data[j,i]
                    s += d * d
                end
            elseif model == :moffat
                γ   = p[1] / (2 * sqrt(2^(1/p[2]) - 1))
                α, a, cx, cy, bg = p[2], p[3], p[4], p[5], p[6]
                for j in eachindex(ys), i in eachindex(xs)
                    r2 = (xs[i]-cx)^2 + (ys[j]-cy)^2
                    d  = a / (1 + r2 / γ^2)^α + bg - data[j,i]
                    s += d * d
                end
            else
                fwhm, a, cx, cy, bg = p[1], p[2], p[3], p[4], p[5]
                r₀ = fwhm / 3.24
                for j in eachindex(ys), i in eachindex(xs)
                    r = sqrt((xs[i]-cx)^2 + (ys[j]-cy)^2) / r₀ * 3.8317
                    v = iszero(r) ? 1.0 : (2 * besselj1(r) / r)^2
                    d = a * v + bg - data[j,i]
                    s += d * d
                end
            end
            return s
        end

        result = optimize(loss, p0, NelderMead())

        if !Optim.converged(result)
            @warn "FitPSF did not converge after $(Optim.iterations(result)) iters"
        end

        p_best = Optim.minimizer(result)
        if model == :moffat
            fwhm, alpha, amp, xc, yc, bkg = p_best
        else
            fwhm, amp, xc, yc, bkg = p_best[1], p_best[2], p_best[3], p_best[4], p_best[5]
            alpha = nothing
        end

        kernel = if model == :gaussian
            Float64[PSFModels.gaussian(ix, iy; fwhm, x=xc, y=yc) for iy in ys, ix in xs]
        elseif model == :moffat
            Float64[PSFModels.moffat(ix, iy; fwhm, alpha, x=xc, y=yc) for iy in ys, ix in xs]
        else
            Float64[PSFModels.airydisk(ix, iy; fwhm, x=xc, y=yc) for iy in ys, ix in xs]
        end
        kernel ./= sum(kernel)

        model_img = if model == :gaussian
            Float64[PSFModels.gaussian(ix, iy; fwhm, amp, x=xc, y=yc, bkg) for iy in ys, ix in xs]
        elseif model == :moffat
            Float64[PSFModels.moffat(ix, iy; fwhm, alpha, amp, x=xc, y=yc, bkg) for iy in ys, ix in xs]
        else
            Float64[PSFModels.airydisk(ix, iy; fwhm, amp, x=xc, y=yc, bkg) for iy in ys, ix in xs]
        end
        residual = data .- model_img

        return (;
            model, fwhm, amp, x=xc, y=yc, alpha, bkg,
            kernel, residual,
            loss      = Optim.minimum(result),
            converged = Optim.converged(result),
        )
    end

    # ═══════════════════════════════════════════════════════════════
    #  Point Source Sub-Pixel PSF Rendering
    # ═══════════════════════════════════════════════════════════════

    """
        evaluate_psf_at(psf::AbstractPSF, dx, dy; pixel_scale, half)

    Evaluate the PSF at a specific offset `(dx, dy)` in coarse-pixel units,
    returning a normalised kernel matrix of size `(2*half+1) × (2*half+1)`.

    Used by the `:shift` method of `render_point!`.
    """
    function evaluate_psf_at(psf::AbstractPSF, dx::Real, dy::Real;
                              pixel_scale::Real=0.04, half::Int=7)
        kernel = make_kernel(psf, pixel_scale, half)
        sz = 2*half + 1
        out = zeros(eltype(kernel), sz, sz)
        for j in 1:sz, i in 1:sz
            px = (i - half - 1) - dx
            py = (j - half - 1) - dy
            out[j, i] = _interpolate_kernel(kernel, px, py, half)
        end
        s = sum(out)
        return iszero(s) ? out : out ./ s
    end

    "Bilinear interpolation within the kernel at fractional pixel offset."
    function _interpolate_kernel(kernel, px, py, half)
        sz = size(kernel, 1)
        T = eltype(kernel)
        ix0 = floor(Int, px + half + 1)
        iy0 = floor(Int, py + half + 1)
        ix1, iy1 = ix0 + 1, iy0 + 1
        if ix0 < 1 || ix1 > sz || iy0 < 1 || iy1 > sz
            return zero(T)
        end
        wx = px + half + 1 - ix0
        wy = py + half + 1 - iy0
        return (1-wx)*(1-wy) * kernel[iy0, ix0] +
               wx*(1-wy)     * kernel[iy1, ix0] +
               (1-wx)*wy     * kernel[iy0, ix1] +
               wx*wy         * kernel[iy1, ix1]
    end

    """
        render_point!(image, psf, x_src, y_src, flux;
                       pixel_scale=0.04, half=7, method=:supersample, n_sub=5)

    Render a point source at 1-indexed pixel position `(x_src, y_src)` onto
    `image` (modified in-place).  `x_src` and `y_src` are in **pixel coordinates**
    (1-indexed, sub-pixel via fractional part).  `pixel_scale` [arcsec/pixel]
    is used only for the PSF kernel scale.

    **Methods**
    - `:supersample` — Build fine kernel at `pixel_scale/n_sub`,
      sum-pool into coarse pixels.  **Required for undersampled data**
      (e.g. WFC3 FWHM ≈ 1.3 px).  Default.
    - `:shift` — Evaluate PSF at each pixel centre offset, normalise.
      Fast preview.  Only reliable when FWHM ≫ 2 px.

    **Reference**: strong-lensing-code-dev skill, point-source-psf.md
    """
    function render_point!(image::AbstractMatrix, psf::AbstractPSF,
                           x_src::Real, y_src::Real, flux::Real;
                           pixel_scale::Real=0.04, half::Int=7,
                           method::Symbol=:supersample, n_sub::Int=5)

        ny_img, nx_img = size(image)

        # x_src, y_src are 1-indexed pixel coordinates (fractional allowed)
        ipx = floor(Int, x_src)         # integer pixel containing source
        ipy = floor(Int, y_src)
        T = eltype(image)
        fpx = T(x_src) - ipx      # fractional offset ∈ [0, 1)
        fpy = T(y_src) - ipy

        # Window origin in 1-indexed image pixel coordinates
        ci0 = ipx - half
        cj0 = ipy - half
        sz = 2*half + 1

        if method == :supersample
            # Build fine kernel at pixel_scale/n_sub
            fine_scale = T(pixel_scale) / n_sub
            half_fine = (half + 1) * n_sub       # PADDED for sub-pixel offsets
            fk = make_kernel(psf, fine_scale, half_fine)
            ksz = size(fk, 1)
            fc = half_fine + 1                    # 1-indexed fine centre

            for cj in 0:(sz-1), ci in 0:(sz-1)
                fi0 = ceil(Int, fc + (ci - half - 0.5 - fpx) * n_sub)
                fi1 = floor(Int, fc + (ci - half + 0.5 - fpx) * n_sub - 1e-12)
                fj0 = ceil(Int, fc + (cj - half - 0.5 - fpy) * n_sub)
                fj1 = floor(Int, fc + (cj - half + 0.5 - fpy) * n_sub - 1e-12)
                fi0 = max(1, min(ksz, fi0))
                fi1 = max(1, min(ksz, fi1))
                fj0 = max(1, min(ksz, fj0))
                fj1 = max(1, min(ksz, fj1))

                if fi1 >= fi0 && fj1 >= fj0
                    val = sum(fk[fj0:fj1, fi0:fi1])
                    ix = ci0 + ci
                    iy = cj0 + cj
                    if 1 <= ix <= nx_img && 1 <= iy <= ny_img
                        image[iy, ix] += val * flux
                    end
                end
            end
        else
            # Shift method: displaced PSF at coarse pixel centres
            kernel = evaluate_psf_at(psf, fpx, fpy; pixel_scale, half)
            for cj in 1:sz, ci in 1:sz
                ix = ci0 + ci - 1
                iy = cj0 + cj - 1
                if 1 <= ix <= nx_img && 1 <= iy <= ny_img
                    image[iy, ix] += kernel[cj, ci] * flux
                end
            end
        end
        return image
    end

end