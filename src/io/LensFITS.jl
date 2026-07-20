# ═══════════════════════════════════════════════════════════════
#  LensFITS — FITS I/O for real telescope data
#
#  Reads FITS files from HST/WFC3 (and other instruments) and
#  produces `Observation` objects ready for lens modeling.
#
#  Pipeline:
#     obs = read_fits("HE0435-1223.fits")
#     sys = ForwardModel(; lens_plane=..., source_plane=...,
#                          grid=obs.grid, psf=obs.psf)
#     logp = masked_logp(sys, obs.data, obs.noise.σ, obs.mask)
# ═══════════════════════════════════════════════════════════════

module LensFITS

    using FITSIO
    using Jens.LensGenerator: Grid, GenGrid
    using Jens.LensNoise: LensNoise, GaussNoise, GaussPoissNoise, estimate_sigma
    using Jens.LensObservation: Observation, all_true_mask
    using Jens: JFloat

    export read_fits, read_header, write_fits, write_model

    # ═══════════════════════════════════════════════════════════════
    #  Internal: find the first HDU containing image data
    # ═══════════════════════════════════════════════════════════════

    """
        _find_science_hdu(f::FITS) -> (hdu_index, hdu)

    Scan all HDUs and return the first one with `NAXIS >= 2`.
    Throws an error if no image HDU is found.
    """
    function _find_science_hdu(f::FITS)
        for i in 1:length(f)
            hdr = FITSIO.read_header(f[i])
            try
                naxis = hdr["NAXIS"]
                if naxis >= 2
                    return i, f[i]
                end
            catch
            end
        end
        close(f)
        error("No image HDU (NAXIS >= 2) found in file")
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: extract all non-structural keywords into a Dict
    # ═══════════════════════════════════════════════════════════════

    const _SKIP_KEYS = Set(["SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2",
                             "NAXIS3", "EXTEND", "COMMENT", "HISTORY", "",
                             "BSCALE", "BZERO", "BLANK", "DATAMIN", "DATAMAX",
                             "CHECKSUM", "DATASUM"])

    function _collect_header(hdu)
        hdr = FITSIO.read_header(hdu)
        d = Dict{String, Any}()
        for key in keys(hdr)
            ks = string(key)
            ks in _SKIP_KEYS && continue
            try
                d[ks] = hdr[key]
            catch
                # skip unreadable values
            end
        end
        return d
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: derive pixel scale [arcsec/pixel] from WCS
    # ═══════════════════════════════════════════════════════════════

    """
        _pixel_scale_from_wcs(header) -> Float64

    Compute pixel scale in arcsec/pixel from the CD matrix or
    CDELT keywords in the header.  Returns 0.0 if neither is found.
    """
    function _pixel_scale_from_wcs(header::Dict)
        # Try CD matrix first: pix_scale = sqrt(|det(CD)|) * 3600 [arcsec/pix]
        cd11 = Float64(get(header, "CD1_1", 0.0))
        cd12 = Float64(get(header, "CD1_2", 0.0))
        cd21 = Float64(get(header, "CD2_1", 0.0))
        cd22 = Float64(get(header, "CD2_2", 0.0))
        det_cd = cd11 * cd22 - cd12 * cd21

        if abs(det_cd) > 1e-30
            return sqrt(abs(det_cd)) * 3600.0
        end

        # Fallback: CDELT keywords (older FITS convention)
        cdelt1 = Float64(get(header, "CDELT1", 0.0))
        if abs(cdelt1) > 1e-30
            return abs(cdelt1) * 3600.0
        end

        # Last resort: single CD1_1
        if abs(cd11) > 1e-30
            return abs(cd11) * 3600.0
        end

        return 0.0
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: keyword lookup with fallback chain
    # ═══════════════════════════════════════════════════════════════

    """
        _kw_get(header, keys, default=nothing)

    Look up `keys` in the header Dict.  `keys` can be:
    - a single `String` or `Symbol`
    - a `Tuple` of strings/symbols (tried in order, first match wins)

    Returns `default` if no key is found.
    """
    function _kw_get(header::Dict, keys, default=nothing)
        if keys isa String || keys isa Symbol
            return get(header, string(keys), default)
        else
            for k in keys
                haskey(header, string(k)) && return header[string(k)]
            end
            return default
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: resolve noise model from header + data
    # ═══════════════════════════════════════════════════════════════

    """
        _resolve_noise(header, data, noise, noise_kw, noise_method) -> LensNoise

    Noise resolution (in priority order):
    1. `noise` is a concrete `LensNoise` → use directly
    2. `noise == :gauss_poiss` → `GaussPoissNoise(sigma_gauss, EXPTIME)`
       sigma_gauss from header kw or blind estimate
    3. `noise == :auto` → `GaussNoise(σ)` from header kw or blind estimate
    4. `noise == :none` → `GaussNoise(0.0)`
    """
    function _resolve_noise(header::Dict, data::AbstractMatrix,
                            noise, noise_kw, noise_method::Symbol,
                            noise_kwargs)
        # Level 1: explicit noise model passed by user
        if noise isa LensNoise
            return noise
        end

        # :none — uninitialised noise
        if noise == :none
            return GaussNoise(0.0)
        end

        # ── resolve sigma: header kw → blind estimate ──
        sigma = nothing
        if noise_kw !== nothing
            val = _kw_get(header, noise_kw, nothing)
            if val !== nothing
                sigma = Float64(val)
            end
        end
        if sigma === nothing
            sigma = Float64(estimate_sigma(data; method=noise_method,
                                           noise_kwargs...))
            sigma = max(sigma, 1e-12)
        end

        # Level 2: combined Gaussian + Poisson
        if noise == :gauss_poiss
            exp_time = Float64(get(header, "EXPTIME", 1.0))
            return GaussPoissNoise(sigma, exp_time)
        end

        # Level 3: pure Gaussian (:auto)
        if noise == :auto
            return GaussNoise(sigma)
        end

        # Fallback (should not reach here)
        return GaussNoise(0.0)
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: normalise header — remap custom keywords to standard names
    # ═══════════════════════════════════════════════════════════════

    """
        _normalise_header!(header; kw_mapping...)

    For each `standard_key => custom_key` pair, if `custom_key` exists
    in `header` and differs from `standard_key`, copy its value to
    `standard_key`.  This allows downstream code (e.g. `LensObservation`
    accessors) to always use standard keyword names.
    """
    function _normalise_header!(header::Dict; kw_pairs...)
        for (std_key, custom_key) in kw_pairs
            sk = string(std_key)
            ck = string(custom_key)
            if ck != sk && haskey(header, ck)
                header[sk] = header[ck]
            end
        end
        return header
    end


    # ═══════════════════════════════════════════════════════════════
    #  read_fits  —  main entry point
    # ═══════════════════════════════════════════════════════════════

    """
        obs = read_fits(path; noise=:auto, noise_kw="SIGMA", ...)

    Read a FITS file and return an `Observation` object ready for
    lens modeling.

    # Noise control

    Three-level resolution (in priority order):

    1. **Explicit** — pass a noise model directly:
       `noise = GaussNoise(0.05)` or `noise = PoissNoise(300.0)`

    2. **Header keyword** — read noise σ from a FITS header keyword:
       `noise_kw = "SIGMA"`       → reads `header["SIGMA"]`
       `noise_kw = ("MY_NOISE", "SIGMA", "RMS")`  → tries each in order

    3. **Blind estimate** — MAD estimate from pixel data (default):
       `noise = :auto`  (default)

    Pass `noise = :none` to leave noise uninitialised.

    # ROI / Grid

    Extract a sub-region instead of using the full image:

    - `center = (990.0, 902.0)` — ROI centre in FITS pixel coordinates.
      Defaults to image centre if omitted.
    - `radius = 128` — half-side of the square ROI [pixels].
      Extracted region is (2×radius + 1) × (2×radius + 1) pixels;
      grid is pix_n = 2×radius.

    CRPIX keywords are adjusted to preserve WCS for the sub-region.

    - `target = "lens_core"` — label stored as `obs.target` (also written
      to `JENS_TARGET` in the FITS header for round-trip).  Auto-infers
      `"full"` or `"roi"` if omitted.

    # Keyword remapping

    Use `kw_*` arguments when your FITS file uses non-standard keywords:

    | argument | default | effect |
    |----------|---------|--------|
    | `kw_exptime`  | `\"EXPTIME\"`  | keyword for exposure time [s] |
    | `kw_filter`   | `\"FILTER\"`   | keyword for filter name |
    | `kw_photflam` | `\"PHOTFLAM\"` | keyword for inverse sensitivity |
    | `kw_photplam` | `\"PHOTPLAM\"` | keyword for pivot wavelength |

    # Overrides

    - `pix_size = 0.09` — override WCS-derived pixel scale [arcsec/pix]
    - `exptime  = 1000.0` — override header exposure time [s]

    # Example

    ```julia
    # Standard HST data (full image, no crop)
    obs = read_fits(\"img/HE0435-1223.fits\")

    # Sub-region: 257×257 pixels centred on the lens
    obs = read_fits(\"img/HE0435-1223.fits\";
        center = (990.0, 902.0), radius = 128)

    # Named sub-region (label survives write_fits → read_fits round-trip)
    obs = read_fits(\"img/HE0435-1223.fits\";
        center = (990.0, 902.0), radius = 64,
        target = \"agn_core\")

    # Custom pipeline with non-standard keywords
    obs = read_fits(\"data.fits\";
        noise       = :auto,
        noise_kw    = (\"MY_SIGMA\", \"SIGMA\"),
        kw_filter   = \"FLT\",
        kw_exptime  = \"TEXP\",
    )

    # Fully manual
    obs = read_fits(\"data.fits\";
        noise    = GaussNoise(0.03),
        pix_size = 0.08,
        exptime  = 1200.0,
    )
    ```
    """
    function read_fits(path::String;
            # ── Noise ──
            noise::Union{LensNoise, Symbol} = :auto,
            noise_kw = "SIGMA",            # String or Tuple of strings
            noise_method::Symbol = :mad,
            noise_kwargs = (),             # extra kwargs for estimate_sigma
            # ── Overrides ──
            pix_size::Union{Real, Nothing} = nothing,
            exptime::Union{Real, Nothing}  = nothing,
            # ── ROI / Grid ──
            center::Union{Tuple{Real,Real}, Nothing} = nothing,  # ROI centre [pixels]
            radius::Union{Int, Nothing} = nothing,   # half-side of ROI [pixels]; pix_n = 2×radius
            target::Union{String, Nothing} = nothing,  # label ("full"/"roi"/custom)
            # ── Keyword remapping ──
            kw_exptime  = "EXPTIME",
            kw_filter   = "FILTER",
            kw_photflam = "PHOTFLAM",
            kw_photplam = "PHOTPLAM",
        )

        if !endswith(path, ".fits") && !endswith(path, ".fits.gz") &&
           !endswith(path, ".fit") && !endswith(path, ".FITS")
            path = path * ".fits"
        end

        f = FITS(path)

        # ── 1. Read primary header (metadata) ──
        header = _collect_header(f[1])

        # ── 2. Find and read science data ──
        sci_idx, sci_hdu = _find_science_hdu(f)
        data_raw = read(sci_hdu)
        data = JFloat.(data_raw)

        # Merge science HDU header (may override primary)
        merge!(header, _collect_header(f[sci_idx]))
        close(f)

        # ── 3. Normalise header (remap custom keywords → standard names) ──
        _normalise_header!(header;
            kw_pairs = (FILTER   = kw_filter,
                        EXPTIME  = kw_exptime,
                        PHOTFLAM = kw_photflam,
                        PHOTPLAM = kw_photplam))

        # ── 4. Derive pixel scale ──
        if pix_size !== nothing
            pix_size_arcsec = Float64(pix_size)
        else
            pix_size_arcsec = _pixel_scale_from_wcs(header)
            if pix_size_arcsec < 1e-10
                @warn "No WCS CD matrix or CDELT found in header. " *
                      "Defaulting to 0.04 arcsec/pix. " *
                      "Pass `pix_size=...` to override."
                pix_size_arcsec = 0.04
            end
        end

        # ── 5. Region of Interest (ROI) extraction ──
        #     data from FITSIO: size = (NAXIS1, NAXIS2) = (cols, rows)
        nx_data, ny_data = size(data)
        _center = nothing   # will be populated if ROI is used
        _radius = nothing

        if radius !== nothing
            # ── Custom ROI: square of side (2×radius + 1) pixels ──
            roi_side = 2 * radius + 1
            pix_n    = 2 * radius

            if center === nothing
                cx, cy = div(nx_data, 2), div(ny_data, 2)
            else
                cx = Int(round(center[1]))
                cy = Int(round(center[2]))
            end

            half_lo = radius
            half_hi = radius   # symmetric: roi_side = half_lo + 1 + half_hi

            x1 = cx - half_lo
            x2 = cx + half_hi
            y1 = cy - half_lo
            y2 = cy + half_hi

            # Clamp to image bounds and warn
            x1c, x2c = max(1, x1), min(nx_data, x2)
            y1c, y2c = max(1, y1), min(ny_data, y2)
            if (x1c != x1) || (x2c != x2) || (y1c != y1) || (y2c != y2)
                @warn "ROI ($x1:$x2, $y1:$y2) clipped to image bounds → ($x1c:$x2c, $y1c:$y2c)"
            end

            data = data[x1c:x2c, y1c:y2c]
            _center = (cx, cy)
            _radius = radius

            # Adjust CRPIX for sub-region offset
            for key in ("CRPIX1", "CRPIX2")
                if haskey(header, key)
                    offset = (key == "CRPIX1" ? x1c - 1 : y1c - 1)
                    header[key] = Float64(header[key]) - offset
                end
            end
        else
            # ── Full image (no auto-crop) ──
            pix_n = min(nx_data, ny_data) - 1
            if nx_data != ny_data
                @warn "FITS data is rectangular ($(nx_data)×$(ny_data)). " *
                      "Grid will use the smaller side (pix_n=$pix_n). " *
                      "Pass `center` + `radius` to extract a sub-region."
            end
        end

        # ── 6. Build coordinate Grid ──
        pix_size_f32 = JFloat(pix_size_arcsec)
        grid = GenGrid(pix_n=pix_n, pix_size=pix_size_f32)

        # ── 7. Resolve noise model ──
        noise_model = _resolve_noise(header, data, noise, noise_kw,
                                     noise_method, noise_kwargs)

        # ── 8. Extract exposure time ──
        if exptime !== nothing
            exposure_time = Float64(exptime)
        else
            exposure_time = Float64(get(header, "EXPTIME", 0.0))
        end

        # ── 9. Resolve target label ──
        # Priority: user-provided > already in header > auto-infer.
        if target !== nothing
            _target = target
        elseif haskey(header, "JENS_TARGET")
            _target = string(header["JENS_TARGET"])
        elseif radius !== nothing
            _target = "roi"
        else
            _target = "full"
        end
        # Also store in header for round-trip via write_fits
        header["JENS_TARGET"] = _target

        # ── 10. Build Observation ──
        mask = all_true_mask(data)

        return Observation(;
            data=data, grid=grid, noise=noise_model, mask=mask,
            psf=nothing, exposure_time=exposure_time, header=header,
            target=_target, center=_center, radius=_radius,
            source_path=path)
    end

    # ═══════════════════════════════════════════════════════════════
    #  read_header  —  metadata-only
    # ═══════════════════════════════════════════════════════════════

    """
        header = read_header(path) -> Dict{String, Any}

    Read only the FITS header without loading pixel data.
    Useful for inspecting WCS, photometry, and observation metadata.

    # Example
    ```julia
    hdr = read_header("img/HE0435-1223.fits")
    println(hdr["FILTER"])      # "F160W"
    println(hdr["EXPTIME"])     # 9939.59
    ```
    """
    function read_header(path::String)
        if !endswith(path, ".fits") && !endswith(path, ".fits.gz") &&
           !endswith(path, ".fit") && !endswith(path, ".FITS")
            path = path * ".fits"
        end
        f = FITS(path)
        header = Dict{String, Any}()
        # Collect from all HDUs (WCS keywords may be in extension, not primary)
        for i in 1:length(f)
            merge!(header, _collect_header(f[i]))
        end
        close(f)
        return header
    end

    # ═══════════════════════════════════════════════════════════════
    #  write_fits  —  write image to FITS
    # ═══════════════════════════════════════════════════════════════

    """
        write_fits(path, image; header, overwrite)

    Write an image array to a FITS file.

    # Arguments
    - `path::String`: output file path.
    - `image::AbstractMatrix`: pixel data to write.
    - `header::Dict{String, Any}`: FITS header keywords (optional).
    - `overwrite::Bool=true`: overwrite existing file.

    # Example
    ```julia
    write_fits("model.fits", model_image; header=obs.header)
    ```
    """
    function write_fits(path::String, image::AbstractMatrix;
                         header::Dict{String, Any}=Dict{String, Any}(),
                         overwrite::Bool=true)
        if !endswith(path, ".fits")
            path = path * ".fits"
        end
        overwrite && isfile(path) && rm(path; force=true)

        FITS(path, "w") do f
            # Write data as Float32 (matches JFloat)
            data_out = Float32.(image)
            write(f, data_out)
            hdu = f[1]

            # Write header keywords
            for (key, val) in header
                try
                    write_key(hdu, string(key), val)
                catch
                    # skip keywords that FITSIO rejects
                end
            end
        end
        return nothing
    end

    # ═══════════════════════════════════════════════════════════════
    #  write_model  —  write model image with observation metadata
    # ═══════════════════════════════════════════════════════════════

    """
        write_model(path, model; obs, residual)

    Write a model image to FITS, preserving the WCS and metadata
    from the original `Observation`.

    # Arguments
    - `path::String`: output file path.
    - `model::AbstractMatrix`: rendered model image (same shape as `obs.data`).
    - `obs::Observation`: original observation (for header inheritance).
    - `residual::AbstractMatrix`: optional residual `data - model`.

    # Example
    ```julia
    sys = build_system(params)
    model = render(sys)
    residual = obs.data .- model

    write_model("result.fits", model; obs=obs, residual=residual)
    ```
    """
    function write_model(path::String, model::AbstractMatrix;
                          obs::Observation,
                          residual::Union{AbstractMatrix, Nothing}=nothing)
        if !endswith(path, ".fits")
            path = path * ".fits"
        end
        isfile(path) && rm(path; force=true)

        # Inherit observation header
        out_header = copy(obs.header)
        out_header["HISTORY"] = "Model generated by Jens.jl LensFITS.write_model"

        if residual === nothing
            # Single HDU: model image + original WCS
            write_fits(path, model; header=out_header, overwrite=true)
        else
            # Multi-HDU: model (ext 1) + residual (ext 2)
            FITS(path, "w") do f
                write(f, Float32.(model))
                hdu1 = f[1]
                for (key, val) in out_header
                    try write_key(hdu1, string(key), val) catch; end
                end

                # Extension: residual
                res_out = Float32.(residual)
                write(f, res_out)
                hdu2 = f[2]
                try write_key(hdu2, "EXTNAME", "RESIDUAL") catch; end
                try write_key(hdu2, "BUNIT", "ELECTRONS/S") catch; end
            end
        end
        return nothing
    end

end # module LensFITS