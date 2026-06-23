module WFC3

    # ═══════════════════════════════════════════════════════════════
    #  WFC3/UVIS PSF — Fourier optics model
    #
    #  Generates a monochromatic PSF for HST/WFC3 by constructing
    #  the telescope pupil function (circular aperture + central
    #  obscuration + spider supports) and computing the PSF via FFT.
    #
    #  Pipeline:
    #    pupil(u,v) ──▶ FFT ──▶ |amplitude|² ──▶ rebin ──▶ PSF
    #
    #  References:
    #    Krist et al. (2011) — WFC3 Instrument Handbook
    #    Biretta et al. (2024) — WFC3 Data Handbook
    # ═══════════════════════════════════════════════════════════════

    using FFTW

    import Jens.LensPSF: AbstractPSF, make_kernel, _default_half

    export wfc3_psf, WFC3_UVIS_PSF, WFC3_PSF_UVIS_F606W

    # ═══════════════════════════════════════════════════════════════
    #  HST + WFC3 physical constants
    # ═══════════════════════════════════════════════════════════════

    const D_HST       = 2.4          # primary mirror diameter [m]
    const EPSILON_HST = 0.33         # secondary obscuration ratio R_inner / R_outer
    const WFC3_PIXEL  = 0.04         # UVIS pixel scale [arcsec/pixel]

    # Spider vane geometry — 4 vanes, equally spaced by default.
    # For WFC3 the spider angles have a small rotational offset
    # relative to the detector axes.
    const N_SPIDERS      = 4
    const SPIDER_ANGLE_0 = 0.0       # offset of first vane [rad] (0 = aligned with x-axis)
    const SPIDER_WIDTH   = 0.025     # vane width relative to D (≈6 cm)

    # ═══════════════════════════════════════════════════════════════
    #  Pupil function construction
    # ═══════════════════════════════════════════════════════════════

    """
        _pupil_grid(n::Int) -> (xs, ys, rs)

    Build centred coordinate grids on [-D/2, D/2] for an n×n pupil array.
    """
    function _pupil_grid(n::Int)
        half_D = D_HST / 2.0
        edges = range(-half_D, half_D; length=n+1)
        centres = [(edges[i] + edges[i+1]) / 2.0 for i in 1:n]
        xs = [x for x in centres, _ in centres]
        ys = [y for _ in centres, y in centres]
        rs = @. sqrt(xs^2 + ys^2)
        return xs, ys, rs
    end

    """
        P = _circular_aperture(rs, R_outer, R_inner)

    Binary mask: 1 inside the annular aperture, 0 elsewhere.
    """
    function _circular_aperture(rs::AbstractMatrix, R_outer::Float64, R_inner::Float64)
        P = zeros(Float64, size(rs))
        @. P[(rs <= R_outer) & (rs >= R_inner)] = 1.0
        return P
    end

    """
        P .*= _spider_mask(xs, ys; n_spiders, angle0, width)

    Multiply the pupil by a spider support mask (0 = obstructed).
    Each vane is a rectangular strip of given `width` radiating from
    the secondary to the primary edge.
    """
    function _spider_mask(xs::AbstractMatrix, ys::AbstractMatrix;
                          n_spiders::Int=N_SPIDERS,
                          angle0::Float64=SPIDER_ANGLE_0,
                          width::Float64=SPIDER_WIDTH * D_HST)
        mask = ones(Float64, size(xs))
        half_w = width / 2.0
        for k in 0:(n_spiders-1)
            theta = angle0 + k * pi / n_spiders
            # Project (x,y) onto the perpendicular direction of the vane
            perp = @. -xs * sin(theta) + ys * cos(theta)
            # The vane extends from R_inner to R_outer in the radial direction
            proj = @. xs * cos(theta) + ys * sin(theta)
            R_inner = EPSILON_HST * D_HST / 2.0
            R_outer = D_HST / 2.0
            @. mask[(abs(perp) <= half_w) & (proj >= R_inner) & (proj <= R_outer)] = 0.0
        end
        return mask
    end

    # ═══════════════════════════════════════════════════════════════
    #  WFC3_UVIS_PSF struct (implements AbstractPSF)
    # ═══════════════════════════════════════════════════════════════

    """
        WFC3_UVIS_PSF(; lambda_eff=606e-9, oversample=5)

    Monochromatic WFC3/UVIS PSF model via Fourier optics.
    Implements `make_kernel` for `AbstractPSF` compatibility.
    """
    struct WFC3_UVIS_PSF <: AbstractPSF
        lambda_eff::Float64    # effective wavelength [m]
        oversample::Int        # sub-pixel oversampling
    end

    function WFC3_UVIS_PSF(; lambda_eff::Real=606e-9, oversample::Int=5)
        return WFC3_UVIS_PSF(Float64(lambda_eff), oversample)
    end

    function make_kernel(psf::WFC3_UVIS_PSF, pixel_scale::Float64, half::Int)
        npix = 2 * half + 1
        kernel, _, _ = _generate_wfc3_kernel(
            psf.lambda_eff, npix, pixel_scale, psf.oversample
        )
        return kernel
    end

    # default_half: 6σ coverage (~3× FWHM) to capture diffraction wings + spiders
    #    FWHM ≈ λ/D for HST, converted to arcsec via 206265 rad⁻¹
    _default_half(psf::WFC3_UVIS_PSF, pixel_scale) = begin
        fwhm_rad  = psf.lambda_eff / D_HST         # diffraction limit [rad]
        fwhm_arc  = fwhm_rad * 206265.0             # [arcsec]
        ceil(Int, 3.0 * fwhm_arc / pixel_scale)     # 6σ ≈ 3×FWHM → pixels
    end

    # ═══════════════════════════════════════════════════════════════
    #  Raw PSF generation (internal, shared by wfc3_psf + make_kernel)
    # ═══════════════════════════════════════════════════════════════

    """
        psf, x_pix, y_pix = wfc3_psf(; kwargs...)

    Generate a monochromatic WFC3/UVIS PSF via Fourier optics.
    Convenience wrapper around `_generate_wfc3_kernel`.

    # Arguments
    - `lambda_eff::Float64=606e-9`: effective wavelength [m] (F606W ≈ 606 nm)
    - `npix::Int=51`: output PSF stamp size [pixels] (odd)
    - `npix_pupil::Int=511`: pupil-plane array size
    - `oversample::Int=5`: sub-pixel oversampling factor
    - `pixel_scale::Float64=0.04`: detector pixel scale [arcsec/pixel]

    # Returns
    - `psf::Matrix{Float64}`: normalised PSF (sum = 1)
    - `x_pix::Vector{Float64}`: x pixel centres [arcsec]
    - `y_pix::Vector{Float64}`: y pixel centres [arcsec]
    """
    function wfc3_psf(;
            lambda_eff::Float64 = 606e-9,
            npix::Int           = 51,
            npix_pupil::Int     = 511,
            oversample::Int     = 5,
            pixel_scale::Float64 = WFC3_PIXEL,
        )
        return _generate_wfc3_kernel(lambda_eff, npix, pixel_scale, oversample; npix_pupil=npix_pupil)
    end

    function _generate_wfc3_kernel(
            lambda_eff::Float64, npix::Int, pixel_scale::Float64, oversample::Int;
            npix_pupil::Int = 511,
        )

        @assert isodd(npix) "npix must be odd (for a central pixel)"
        @assert isodd(npix_pupil) "npix_pupil must be odd"

        # ── 1. Pupil function ─────────────────────────────────────
        xs, ys, rs = _pupil_grid(npix_pupil)
        R_outer = D_HST / 2.0
        R_inner = EPSILON_HST * R_outer

        pupil = _circular_aperture(rs, R_outer, R_inner)
        pupil .*= _spider_mask(xs, ys)

        # Normalize pupil so total transmitted power = 1
        pupil ./= sqrt(sum(abs2, pupil))

        # ── 2. Zero-pad for sub-pixel PSF sampling ────────────────
        pad_size = npix_pupil * oversample
        pupil_padded = zeros(ComplexF64, pad_size, pad_size)
        off = (pad_size - npix_pupil) ÷ 2
        pupil_padded[off+1:off+npix_pupil, off+1:off+npix_pupil] .= pupil

        # ── 3. FFT → complex amplitude → intensity ────────────────
        amplitude = fftshift(fft(pupil_padded))
        psf_fine  = abs2.(amplitude)
        psf_fine ./= sum(psf_fine)  # normalise

        # ── 4. Image plane pixel scale ────────────────────────────
        #    Δθ_fine = λ / (oversample · D)
        dtheta_fine_rad = lambda_eff / (oversample * D_HST)
        dtheta_fine_arcsec = dtheta_fine_rad * (180.0 / pi) * 3600.0

        # Rebin factor to match detector pixel scale
        rebin_raw = pixel_scale / dtheta_fine_arcsec
        rebin = max(1, round(Int, rebin_raw))

        # Actual pixel scale we achieve (may differ slightly from target)
        actual_pscale = dtheta_fine_arcsec * rebin

        # ── 5. Rebin (average pooling) ────────────────────────────
        # Crop psf_fine to an integer multiple of rebin
        fine_size = size(psf_fine, 1)
        crop_to = (fine_size ÷ rebin) * rebin
        crop_start = (fine_size - crop_to) ÷ 2 + 1
        crop_end   = crop_start + crop_to - 1
        psf_cropped = psf_fine[crop_start:crop_end, crop_start:crop_end]

        # Average pooling
        n_bins = crop_to ÷ rebin
        psf_binned = zeros(Float64, n_bins, n_bins)
        for j in 1:n_bins, i in 1:n_bins
            rj = (j-1)*rebin+1 : j*rebin
            ri = (i-1)*rebin+1 : i*rebin
            psf_binned[j, i] = sum(psf_cropped[ri, rj]) / (rebin^2)
        end

        # ── 6. Crop to desired output size ────────────────────────
        if n_bins >= npix
            c0 = n_bins ÷ 2
            h  = npix ÷ 2
            psf = psf_binned[c0-h+1:c0+h+1, c0-h+1:c0+h+1]
        else
            # Pad with zeros if binned PSF is smaller than requested
            psf = zeros(Float64, npix, npix)
            off = (npix - n_bins) ÷ 2
            psf[off+1:off+n_bins, off+1:off+n_bins] .= psf_binned
        end
        psf ./= sum(psf)

        # ── 7. Coordinate axes ────────────────────────────────────
        half = npix ÷ 2
        x_pix = [(-half + i) * actual_pscale for i in 0:(npix-1)]
        y_pix = copy(x_pix)

        return psf, x_pix, y_pix
    end

    # ═══════════════════════════════════════════════════════════════
    #  Convenience presets
    # ═══════════════════════════════════════════════════════════════

    """
        WFC3_PSF_UVIS_F606W(; npix=51) -> (psf, x_pix, y_pix)

    Pre-configured WFC3/UVIS PSF for the F606W broad-band filter
    (effective wavelength ≈ 606 nm).
    """
    function WFC3_PSF_UVIS_F606W(; npix::Int=51)
        return wfc3_psf(; lambda_eff=606e-9, npix=npix)
    end

    function WFC3_PSF_UVIS_F814W(; npix::Int=51)
        return wfc3_psf(; lambda_eff=814e-9, npix=npix)
    end

    function WFC3_PSF_UVIS_F438W(; npix::Int=51)
        return wfc3_psf(; lambda_eff=438e-9, npix=npix)
    end

    function WFC3_PSF_UVIS_F555W(; npix::Int=51)
        return wfc3_psf(; lambda_eff=555e-9, npix=npix)
    end

    function WFC3_PSF_UVIS_F275W(; npix::Int=51)
        return wfc3_psf(; lambda_eff=275e-9, npix=npix)
    end

    function WFC3_PSF_UVIS_F336W(; npix::Int=51)
        return wfc3_psf(; lambda_eff=336e-9, npix=npix)
    end

    function WFC3_PSF_UVIS_F475W(; npix::Int=51)
        return wfc3_psf(; lambda_eff=475e-9, npix=npix)
    end

end # module WFC3