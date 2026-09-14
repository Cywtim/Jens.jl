# ═══════════════════════════════════════════════════════════════
#  LensObservation — container for observed telescope data
#
#  Observation bundles everything needed to evaluate a forward model
#  against real data: pixel values, coordinate grid, noise model,
#  pixel mask, PSF, and the original FITS header.
#
#  Typical workflow:
#     obs  = read_fits("HE0435-1223.fits")
#     sys  = ForwardModel(lens_plane=..., source_plane=...,
#                          grid=obs.grid, psf=obs.psf)
#     logp = masked_logp(sys, obs.data, obs.noise.sigma, obs.mask)
# ═══════════════════════════════════════════════════════════════

module LensObservation

    using Jens.LensGenerator: Grid, GenGrid
    using Jens.LensNoise: LensNoise, GaussNoise, PoissNoise
    using Jens.LensPSF: AbstractPSF

    export Observation, with_psf, with_noise, with_mask
    export pixel_scale, filter_name, telescope, instrument
    export photflam, photplam, photfnu, pivot_wavelength
    export sky_to_pixel, pixel_to_sky

    # ═══════════════════════════════════════════════════════════════
    #  Observation struct
    # ═══════════════════════════════════════════════════════════════

    """
        Observation{T,N,P}(; data, grid, noise, mask, psf, ...)

    Complete observation container bridging raw telescope data and
    the `ForwardModel` fitting pipeline.

    # Type parameters
    - `T<:AbstractFloat`: element type of `data` and `grid` (e.g. `Float32`)
    - `N<:LensNoise`: concrete noise model type (e.g. `GaussNoise{Float32}`)
    - `P`: concrete PSF type, or `Nothing` when no PSF is attached

    # Fields
    - `data::Matrix{T}`          : observed pixel values (e⁻/s or ADU/s)
    - `grid::Grid{T}`            : coordinate grid (contains pix_size, xg, yg)
    - `noise::N`                 : noise model (`GaussNoise` or `PoissNoise`)
    - `mask::BitMatrix`          : `true` = include pixel in χ²
    - `psf::P`                   : PSF model (`nothing` = no convolution)
    - `exposure_time::Float64`   : exposure time [seconds]
    - `header::Dict{String, Any}`: full FITS header for metadata access
    - `target::String`           : label for this extraction (\"full\", \"roi\", custom)
    - `center::Union{Tuple{Int,Int}, Nothing}` : ROI centre [pixels]; `nothing` = full image
    - `radius::Union{Int, Nothing}`            : ROI half-side [pixels]; `nothing` = full image
    - `source_path::String`                    : original FITS file path (\"\" if mock)

    # Example
    ```julia
    noise = GaussNoise(0.05)           # or PoissNoise(9939.6)
    mask  = circular_mask(grid, 2.5)

    obs = Observation(;
        data          = fits_data,
        grid          = grid,
        noise         = noise,
        mask          = mask,
        psf           = WFC3_UVIS_PSF(),
        exposure_time = 9939.6,
        header        = Dict("FILTER" => "F160W", "TELESCOP" => "HST"),
    )

    sys = ForwardModel(; lens_plane=..., source_plane=...,
                         grid=obs.grid, psf=obs.psf)
    logp = masked_logp(sys, obs.data, obs.noise.σ, obs.mask)
    ```
    """
    struct Observation{T<:AbstractFloat, N<:LensNoise, P}
        data::Matrix{T}
        grid::Grid{T}
        noise::N
        mask::BitMatrix
        psf::P
        exposure_time::Float64
        header::Dict{String, Any}
        target::String
        center::Union{Tuple{Int,Int}, Nothing}   # ROI centre [pixels], nothing = full image
        radius::Union{Int, Nothing}              # ROI half-side [pixels], nothing = full image
        source_path::String                      # original FITS file path ("" if mock)

        # Inner constructor with validation
        function Observation{T,N,P}(;
            data::Matrix{T},
            grid::Grid{T},
            noise::N,
            mask::BitMatrix,
            psf::P=nothing,
            exposure_time::Real=0.0,
            header::Dict{String, Any}=Dict{String, Any}(),
            target::String="full",
            center::Union{Tuple{Int,Int}, Nothing}=nothing,
            radius::Union{Int, Nothing}=nothing,
            source_path::String="",
        ) where {T<:AbstractFloat, N<:LensNoise, P}
            # Validate data-mask shapes match
            size(data) == size(mask) || throw(ArgumentError(
                "data size $(size(data)) ≠ mask size $(size(mask))"))
            # Warn if data doesn't match grid dimensions
            grid_pix = grid.pix_n + 1   # Grid has pix_n + 1 points
            if size(data, 1) != grid_pix || size(data, 2) != grid_pix
                @warn "data size $(size(data)) ≠ grid size ($(grid_pix), $(grid_pix))"
            end
            return new{T,N,P}(data, grid, noise, mask, psf,
                          Float64(exposure_time), header, target,
                          center, radius, source_path)
        end
    end

    # Convenience outer constructor: infer T, N, P from arguments
    function Observation(;
        data::Matrix{T},
        grid::Grid,
        noise::LensNoise,
        mask::BitMatrix,
        psf=nothing,
        exposure_time::Real=0.0,
        header::Dict{String, Any}=Dict{String, Any}(),
        target::String="full",
        center::Union{Tuple{Int,Int}, Nothing}=nothing,
        radius::Union{Int, Nothing}=nothing,
        source_path::String="",
    ) where {T<:AbstractFloat}
        return Observation{T, typeof(noise), typeof(psf)}(;
            data=data, grid=grid, noise=noise, mask=mask,
            psf=psf, exposure_time=exposure_time, header=header,
            target=target, center=center, radius=radius,
            source_path=source_path)
    end

    # ═══════════════════════════════════════════════════════════════
    #  Convenience: all-true mask from data shape
    # ═══════════════════════════════════════════════════════════════

    """
        all_true_mask(data_or_shape)

    Return a `BitMatrix` of `true` values matching the given array
    or shape.  Useful as a default mask when no pixel needs excluding.

    # Example
        mask = all_true_mask(obs.data)
        mask = all_true_mask((256, 256))
    """
    function all_true_mask(data::AbstractMatrix)
        return trues(size(data))
    end
    function all_true_mask(shape::Tuple{Int,Int})
        return trues(shape)
    end

    # ═══════════════════════════════════════════════════════════════
    #  Accessor methods — read from grid or header
    # ═══════════════════════════════════════════════════════════════

    """
        pixel_scale(obs::Observation) -> Float64

    Pixel scale in arcseconds per pixel (from `obs.grid.pix_size`).
    """
    pixel_scale(obs::Observation) = Float64(obs.grid.pix_size)

    """
        filter_name(obs::Observation) -> String

    Filter name from the FITS header (e.g., `\"F160W\"`).
    Returns `\"unknown\"` if not present.
    """
    filter_name(obs::Observation) = string(get(obs.header, "FILTER", "unknown"))

    """
        telescope(obs::Observation) -> String

    Telescope name from the FITS header (e.g., `\"HST\"`).
    Returns `\"unknown\"` if not present.
    """
    telescope(obs::Observation) = string(get(obs.header, "TELESCOP", "unknown"))

    """
        instrument(obs::Observation) -> String

    Instrument name from the FITS header (e.g., `\"WFC3\"`).
    Returns `\"unknown\"` if not present.
    """
    instrument(obs::Observation) = string(
        get(obs.header, "INSTRUME",     # HST convention
        get(obs.header, "INSTRUMENT", "unknown")))

    """
        photflam(obs::Observation) -> Float64

    Inverse sensitivity [erg/cm²/s/Å per e⁻/s] from the FITS header.
    Returns 0.0 if not present.
    """
    photflam(obs::Observation) = Float64(get(obs.header, "PHOTFLAM", 0.0))

    """
        photplam(obs::Observation) -> Float64

    Pivot wavelength [Å] from the FITS header.
    Returns 0.0 if not present.
    """
    photplam(obs::Observation) = Float64(get(obs.header, "PHOTPLAM", 0.0))

    """
        photfnu(obs::Observation) -> Float64

    Inverse sensitivity [Jy per e⁻/s] from the FITS header.
    Returns 0.0 if not present.
    """
    photfnu(obs::Observation) = Float64(get(obs.header, "PHOTFNU", 0.0))

    """
        pivot_wavelength(obs::Observation) -> Float64

    Alias for `photplam(obs)`.
    """
    pivot_wavelength(obs::Observation) = photplam(obs)

    # ═══════════════════════════════════════════════════════════════
    #  WCS coordinate transforms
    #
    #  Uses the CD matrix + CRPIX/CRVAL from the FITS header.
    #  Linear (tangent-plane) approximation — accurate for small
    #  fields (typical strong lens: a few arcsec).
    # ═══════════════════════════════════════════════════════════════

    """
        _has_wcs(obs::Observation) -> Bool

    Check whether the header contains the minimum WCS keywords.
    """
    function _has_wcs(obs::Observation)
        return all(k -> haskey(obs.header, k),
                   ["CD1_1", "CD2_2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2"])
    end

    """
        x, y = sky_to_pixel(obs::Observation, ra, dec)

    Convert sky coordinates [degrees] to pixel coordinates [1-indexed]
    using the CD matrix from the FITS header.

    Throws an error if the header lacks WCS keywords.
    """
    function sky_to_pixel(obs::Observation, ra::Real, dec::Real)
        _has_wcs(obs) || error(
            "Observation header lacks WCS keywords (CD1_1, CD2_2, CRPIX*, CRVAL*)")

        cd11   = Float64(obs.header["CD1_1"])
        cd12   = Float64(get(obs.header, "CD1_2", 0.0))
        cd21   = Float64(get(obs.header, "CD2_1", 0.0))
        cd22   = Float64(obs.header["CD2_2"])
        crpix1 = Float64(obs.header["CRPIX1"])
        crpix2 = Float64(obs.header["CRPIX2"])
        crval1 = Float64(obs.header["CRVAL1"])
        crval2 = Float64(obs.header["CRVAL2"])

        dra  = ra  - crval1
        ddec = dec - crval2

        # CD⁻¹ · Δsky
        det = cd11 * cd22 - cd12 * cd21
        x = crpix1 + ( cd22 * dra - cd12 * ddec) / det
        y = crpix2 + (-cd21 * dra + cd11 * ddec) / det
        return x, y
    end

    """
        ra, dec = pixel_to_sky(obs::Observation, x, y)

    Convert pixel coordinates [1-indexed] to sky coordinates [degrees]
    using the CD matrix from the FITS header.

    Throws an error if the header lacks WCS keywords.
    """
    function pixel_to_sky(obs::Observation, x::Real, y::Real)
        _has_wcs(obs) || error(
            "Observation header lacks WCS keywords (CD1_1, CD2_2, CRPIX*, CRVAL*)")

        cd11   = Float64(obs.header["CD1_1"])
        cd12   = Float64(get(obs.header, "CD1_2", 0.0))
        cd21   = Float64(get(obs.header, "CD2_1", 0.0))
        cd22   = Float64(obs.header["CD2_2"])
        crpix1 = Float64(obs.header["CRPIX1"])
        crpix2 = Float64(obs.header["CRPIX2"])
        crval1 = Float64(obs.header["CRVAL1"])
        crval2 = Float64(obs.header["CRVAL2"])

        dx = x - crpix1
        dy = y - crpix2

        ra  = crval1 + cd11 * dx + cd12 * dy
        dec = crval2 + cd21 * dx + cd22 * dy
        return ra, dec
    end

    # ═══════════════════════════════════════════════════════════════
    #  Immutable update helpers — return a new Observation with one
    #  field replaced.  Avoids manual reconstruction when attaching
    #  a PSF, swapping the noise model, or updating the mask.
    # ═══════════════════════════════════════════════════════════════

    """
        obs′ = with_psf(obs, psf)

    Return a new `Observation` with `psf` attached, copying all other
    fields from `obs`.  Since `Observation` is immutable, this is the
    idiomatic way to add a PSF after `read_fits`.

    # Example
    ```julia
    obs = read_fits("data.fits")             # psf = nothing
    obs = with_psf(obs, WFC3_UVIS_PSF())     # now obs.psf is set
    ```
    """
    function with_psf(obs::Observation{T,N}, psf) where {T,N}
        return Observation{T, N, typeof(psf)}(;
            data=obs.data, grid=obs.grid, noise=obs.noise, mask=obs.mask,
            psf=psf, exposure_time=obs.exposure_time, header=obs.header,
            target=obs.target, center=obs.center, radius=obs.radius,
            source_path=obs.source_path)
    end

    """
        obs′ = with_noise(obs, noise)

    Return a new `Observation` with `noise` replaced.
    """
    function with_noise(obs::Observation{T}, noise::LensNoise) where {T}
        return Observation{T, typeof(noise), typeof(obs.psf)}(;
            data=obs.data, grid=obs.grid, noise=noise, mask=obs.mask,
            psf=obs.psf, exposure_time=obs.exposure_time, header=obs.header,
            target=obs.target, center=obs.center, radius=obs.radius,
            source_path=obs.source_path)
    end

    """
        obs′ = with_mask(obs, mask)

    Return a new `Observation` with `mask` replaced.  The new mask
    must have the same shape as `obs.data`.
    """
    function with_mask(obs::Observation{T,N,P}, mask::BitMatrix) where {T,N,P}
        size(mask) == size(obs.data) || throw(ArgumentError(
            "mask size $(size(mask)) ≠ data size $(size(obs.data))"))
        return Observation{T,N,P}(;
            data=obs.data, grid=obs.grid, noise=obs.noise, mask=mask,
            psf=obs.psf, exposure_time=obs.exposure_time, header=obs.header,
            target=obs.target, center=obs.center, radius=obs.radius,
            source_path=obs.source_path)
    end

    # ═══════════════════════════════════════════════════════════════
    #  Convenience: build a quick all-true Observation for testing
    # ═══════════════════════════════════════════════════════════════

    """
        obs = mock_observation(; pix_n, pix_size, sigma)

    Create a minimal `Observation` with zero data and all-true mask
    for testing and prototyping.  Not for use with real data.
    """
    function mock_observation(;
            pix_n::Int=256,
            pix_size::Real=Float32(0.09),
            sigma::Real=0.05,
        )
        grid = GenGrid(pix_n=pix_n, pix_size=Float32(pix_size))
        T = eltype(grid.xg)
        data = zeros(T, pix_n + 1, pix_n + 1)
        mask = trues(pix_n + 1, pix_n + 1)
        noise = GaussNoise(sigma)
        return Observation(;
            data=data, grid=grid, noise=noise, mask=mask,
            exposure_time=0.0, header=Dict{String, Any}(),
            target="mock")
    end

end # module LensObservation