# ═══════════════════════════════════════════════════════════════
#  LensFITSIO — declarative FITS reader for any telescope
#
#  Instead of hardcoding instrument-specific logic, the user
#  describes *how* their FITS file is organised via a `FITSRecipe`
#  config struct.  Presets exist for common instruments but every
#  field can be overridden at call time.
#
#  Basic usage:
#     obs = read_fits("image.fits")
#
#  With a recipe:
#     obs = read_fits("image.fits"; recipe=HST_WFC3())
#
#  Fully custom:
#     obs = read_fits("weird.fits"; recipe=FITSRecipe(
#         hdu_sci = "SCI",
#         noise   = :from_err,
#         wcs     = :pcd,
#     ))
# ═══════════════════════════════════════════════════════════════

module LensFITSIO

    using FITSIO
    using LinearAlgebra: det
    using Jens.LensGenerator: Grid, GenGrid
    using Jens.LensNoise: LensNoise, GaussNoise, GaussPoissNoise, estimate_sigma
    using Jens.LensObservation: Observation, all_true_mask
    using Jens.LensPSF: AbstractPSF
    using Jens: JFloat

    export FITSRecipe, read_fits, read_header
    export HST_WFC3, GenericGround

    # ═══════════════════════════════════════════════════════════════
    #  FITSRecipe — declarative read configuration
    # ═══════════════════════════════════════════════════════════════

    """
        FITSRecipe(; hdu_sci, hdu_err, wcs, noise, ...)

    Declarative recipe that describes how to interpret a FITS file.
    Every field can be overridden at `read_fits` call time.

    # Fields

    | field | type | default | meaning |
    |-------|------|---------|---------|
    | `hdu_sci` | `Int`, `String`, or `Function` | `:auto` | Which HDU holds the science image |
    | `hdu_err` | `Int`, `String`, `Function`, or `nothing` | `nothing` | Which HDU holds the per-pixel error map |
    | `wcs` | `Symbol` or `Function` | `:auto` | How to derive pixel scale + CD matrix |
    | `noise` | `Symbol` or `LensNoise` | `:auto` | Noise model construction strategy |
    | `noise_keyword` | `String`, `Tuple`, or `nothing` | `nothing` | Header keyword(s) for noise σ |
    | `noise_method` | `Symbol` | `:mad` | Blind estimation method for `estimate_sigma` |
    | `unit` | `Symbol` or `Function` | `:auto` | Unit normalisation strategy |
    | `metadata` | `Dict{String,String}` | see below | Map standard keys → FITS header keys |
    | `psf` | `Symbol` | `:none` | PSF resolution (`:none` / `:auto`) |

    # `hdu_sci` — science HDU selector

    - `Int`         → use `f[N]` directly
    - `String`      → scan for `EXTNAME == hdu_sci`
    - `Function`    → `(f::FITS) -> (idx::Int, hdu)`
    - `:auto`       → first HDU with `NAXIS >= 2` (FITSIO default)

    # `wcs` — pixel-scale + world-coordinate resolution

    - `:auto`       → try CD matrix → PC+CDELT → CDELT only
    - `:cd`         → CD matrix only (`CDi_j`)
    - `:pcd`        → PC matrix × CDELT (`PCi_j` + `CDELTi`)
    - `:cdelt`      → diagonal only, no rotation (`CDELT1`)
    - `:none`       → skip; user must provide `pix_size=` kwarg
    - `Function`    → `(header::Dict) -> (pix_scale::Float64, cd::Matrix{Float64}, crpix, crval)`

    # `noise` — noise model strategy

    - `:auto`         → ERR extension if available → header keyword → blind estimate
    - `:gauss`        → `GaussNoise(σ)` with σ from keyword or blind
    - `:gauss_poiss`  → `GaussPoissNoise(σ, exp_time)` from keyword + header
    - `:from_err`     → use per-pixel σ from ERR extension
    - `:blind`        → always blind estimate (MAD, clipped, border)
    - concrete `LensNoise` → use directly

    # `unit` — pixel value normalisation

    - `:auto`   → inspect `BUNIT`; convert `MJy/sr` → `e⁻/s` if possible,
                  otherwise pass through
    - `:none`   → no transformation
    - `Function`→ `(data, header) -> normalized_data`

    # `metadata` — keyword name mapping

    Map *standard* key → *actual header key* in your FITS file.
    Default mapping:
      `"TELESCOP" => "TELESCOP"`, `"INSTRUME" => "INSTRUME"`,
      `"FILTER"   => "FILTER"`,   `"EXPTIME"  => "EXPTIME"`

    # Example

    ```julia
    # JWST NIRCam: SCI in EXTNAME="SCI", ERR in "ERR", PC+CDELT WCS
    recipe = FITSRecipe(
        hdu_sci  = "SCI",
        hdu_err  = "ERR",
        wcs      = :pcd,
        noise    = :from_err,
        metadata = Dict("EXPTIME" => "EFFEXPTM"),
    )

    # Ground-based: single HDU, no WCS, noise from border pixels
    recipe = FITSRecipe(
        wcs          = :none,
        noise        = :blind,
        noise_method = :border,
        metadata     = Dict("EXPTIME" => "TEXPTIME", "FILTER" => "FILTBAND"),
    )
    ```
    """
    @kwdef struct FITSRecipe
        hdu_sci::Any  = :auto     # Int | String | Function | :auto
        hdu_err::Any  = nothing   # Int | String | Function | nothing
        wcs::Any      = :auto     # :auto | :cd | :pcd | :cdelt | :none | Function
        noise::Any    = :auto     # :auto | :gauss | :gauss_poiss | :from_err | :blind | LensNoise
        noise_keyword = nothing   # String | Tuple | nothing
        noise_method::Symbol = :mad
        unit::Any     = :auto     # :auto | :none | Function
        metadata::Dict{String, String} = Dict(
            "TELESCOP" => "TELESCOP",
            "INSTRUME" => "INSTRUME",
            "FILTER"   => "FILTER",
            "EXPTIME"  => "EXPTIME",
        )
        psf::Symbol   = :none     # :none | :auto
    end

    # ═══════════════════════════════════════════════════════════════
    #  Presets — convenience constructors for common instruments
    # ═══════════════════════════════════════════════════════════════

    """
        HST_WFC3()

    Preset for HST/WFC3 drizzled images (`_drz.fits`).

    - Single SCI HDU (first image)
    - CD matrix WCS
    - Gaussian + Poisson noise (from `SIGMA` keyword or blind)
    - PHOTFLAM / PHOTPLAM photometry keywords
    """
    function HST_WFC3()
        return FITSRecipe(;
            hdu_sci       = :auto,
            wcs           = :cd,
            noise         = :gauss_poiss,
            noise_keyword = "SIGMA",
            metadata      = Dict(
                "TELESCOP"  => "TELESCOP",
                "INSTRUME"  => "INSTRUME",
                "FILTER"    => "FILTER",
                "EXPTIME"   => "EXPTIME",
                "PHOTFLAM"  => "PHOTFLAM",
                "PHOTPLAM"  => "PHOTPLAM",
            ),
        )
    end

    """
        GenericGround()

    Fallback preset for ground-based or unknown instruments.

    - First image HDU
    - Tries CD → PC+CDELT → CDELT → fallback
    - Blind noise estimation (MAD)
    - No photometric calibration
    """
    function GenericGround()
        return FITSRecipe(;
            hdu_sci  = :auto,
            wcs      = :auto,
            noise    = :blind,
            noise_method = :mad,
            metadata = Dict(
                "TELESCOP" => "TELESCOP",
                "INSTRUME" => "INSTRUME",
                "FILTER"   => "FILTER",
                "EXPTIME"  => "EXPTIME",
            ),
        )
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: header collection
    # ═══════════════════════════════════════════════════════════════

    const _SKIP_KEYS = Set([
        "SIMPLE", "BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3",
        "EXTEND", "COMMENT", "HISTORY", "",
        "BSCALE", "BZERO", "BLANK", "DATAMIN", "DATAMAX",
        "CHECKSUM", "DATASUM",
    ])

    function _collect_header(hdu)
        hdr = FITSIO.read_header(hdu)
        d = Dict{String, Any}()
        for key in keys(hdr)
            ks = string(key)
            ks in _SKIP_KEYS && continue
            try
                d[ks] = hdr[key]
            catch
            end
        end
        return d
    end

    function _collect_all_headers(f::FITS)
        header = Dict{String, Any}()
        for i in 1:length(f)
            merge!(header, _collect_header(f[i]))
        end
        return header
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: HDU resolution
    # ═══════════════════════════════════════════════════════════════

    function _resolve_hdu(f::FITS, selector)
        if selector isa Integer
            # Direct index: f[N]
            return (selector, f[selector])
        elseif selector isa String
            # Scan for EXTNAME match
            for i in 1:length(f)
                hdr = FITSIO.read_header(f[i])
                try
                    if string(hdr["EXTNAME"]) == selector
                        return (i, f[i])
                    end
                catch
                end
            end
            error("No HDU with EXTNAME == \"$selector\" found")
        elseif selector isa Function
            return selector(f)
        elseif selector == :auto
            # First HDU with NAXIS >= 2
            for i in 1:length(f)
                hdr = FITSIO.read_header(f[i])
                try
                    if hdr["NAXIS"] >= 2
                        return (i, f[i])
                    end
                catch
                end
            end
            error("No image HDU (NAXIS >= 2) found in file")
        else
            error("Unknown hdu_sci selector: $(repr(selector))")
        end
    end

    function _try_read_err(f::FITS, recipe::FITSRecipe)
        recipe.hdu_err === nothing && return nothing
        try
            _, err_hdu = _resolve_hdu(f, recipe.hdu_err)
            return JFloat.(read(err_hdu))
        catch e
            @warn "Failed to read ERR extension: $e"
            return nothing
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: WCS resolution
    # ═══════════════════════════════════════════════════════════════

    """
        _WCSInfo(scale, cd, crpix, crval)

    Parsed WCS information bundle.
    - `scale`: pixel scale [arcsec/pixel]
    - `cd`: 2×2 CD matrix [deg/pixel]
    - `crpix`: reference pixel (crpix1, crpix2)
    - `crval`: reference sky position [deg] (crval1, crval2)
    """
    struct _WCSInfo
        scale::Float64
        cd::Matrix{Float64}
        crpix::Tuple{Float64, Float64}
        crval::Tuple{Float64, Float64}
    end

    function _resolve_wcs(header::Dict, recipe::FITSRecipe)
        if recipe.wcs isa Function
            scale, cd, crpix, crval = recipe.wcs(header)
            return _WCSInfo(Float64(scale), Float64.(cd),
                            (Float64(crpix[1]), Float64(crpix[2])),
                            (Float64(crval[1]), Float64(crval[2])))
        end

        wcs_mode = recipe.wcs
        if wcs_mode == :auto
            # Try CD → PC+CDELT → CDELT
            wcs = _try_wcs_cd(header)
            wcs !== nothing && return wcs
            wcs = _try_wcs_pcd(header)
            wcs !== nothing && return wcs
            wcs = _try_wcs_cdelt(header)
            wcs !== nothing && return wcs
            @warn "No WCS found in header. Use `wcs=:none` + `pix_size=` or provide a custom WCS function."
            return _WCSInfo(0.0, [1.0 0.0; 0.0 1.0], (0.0, 0.0), (0.0, 0.0))
        elseif wcs_mode == :cd
            wcs = _try_wcs_cd(header)
            wcs !== nothing && return wcs
            error("CD matrix WCS requested but no CDi_j keywords found in header")
        elseif wcs_mode == :pcd
            wcs = _try_wcs_pcd(header)
            wcs !== nothing && return wcs
            error("PC+CDELT WCS requested but no PCi_j+CDELTi keywords found in header")
        elseif wcs_mode == :cdelt
            wcs = _try_wcs_cdelt(header)
            wcs !== nothing && return wcs
            error("CDELT WCS requested but no CDELT1 keyword found in header")
        elseif wcs_mode == :none
            return _WCSInfo(0.0, [1.0 0.0; 0.0 1.0], (0.0, 0.0), (0.0, 0.0))
        else
            error("Unknown wcs mode: $(repr(wcs_mode)). Use :auto, :cd, :pcd, :cdelt, :none, or a Function.")
        end
    end

    # ── Individual WCS parsers ──

    function _try_wcs_cd(header::Dict)
        haskey(header, "CD1_1") || haskey(header, "CD2_2") || return nothing
        cd = [
            Float64(get(header, "CD1_1", 0.0))  Float64(get(header, "CD1_2", 0.0));
            Float64(get(header, "CD2_1", 0.0))  Float64(get(header, "CD2_2", 0.0))
        ]
        abs(det(cd)) < 1e-30 && return nothing
        scale = sqrt(abs(det(cd))) * 3600.0
        crpix = (Float64(get(header, "CRPIX1", 0.0)), Float64(get(header, "CRPIX2", 0.0)))
        crval = (Float64(get(header, "CRVAL1", 0.0)), Float64(get(header, "CRVAL2", 0.0)))
        return _WCSInfo(scale, cd, crpix, crval)
    end

    function _try_wcs_pcd(header::Dict)
        # PC_i_j × diag(CDELT_i) = CD_i_j
        haskey(header, "CDELT1") || return nothing
        cdelt1 = Float64(get(header, "CDELT1", 0.0))
        cdelt2 = Float64(get(header, "CDELT2", 0.0))
        pc11 = Float64(get(header, "PC1_1", 1.0))
        pc12 = Float64(get(header, "PC1_2", 0.0))
        pc21 = Float64(get(header, "PC2_1", 0.0))
        pc22 = Float64(get(header, "PC2_2", 1.0))
        cd = [pc11*cdelt1  pc12*cdelt1;
              pc21*cdelt2  pc22*cdelt2]
        abs(det(cd)) < 1e-30 && return nothing
        scale = sqrt(abs(det(cd))) * 3600.0
        crpix = (Float64(get(header, "CRPIX1", 0.0)), Float64(get(header, "CRPIX2", 0.0)))
        crval = (Float64(get(header, "CRVAL1", 0.0)), Float64(get(header, "CRVAL2", 0.0)))
        return _WCSInfo(scale, cd, crpix, crval)
    end

    function _try_wcs_cdelt(header::Dict)
        haskey(header, "CDELT1") || return nothing
        cdelt1 = Float64(get(header, "CDELT1", 0.0))
        abs(cdelt1) < 1e-30 && return nothing
        scale = abs(cdelt1) * 3600.0
        cd = [cdelt1 0.0; 0.0 cdelt1]
        crpix = (Float64(get(header, "CRPIX1", 0.0)), Float64(get(header, "CRPIX2", 0.0)))
        crval = (Float64(get(header, "CRVAL1", 0.0)), Float64(get(header, "CRVAL2", 0.0)))
        return _WCSInfo(scale, cd, crpix, crval)
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: unit normalisation
    # ═══════════════════════════════════════════════════════════════

    function _normalise_units(data::AbstractMatrix, header::Dict, recipe::FITSRecipe)
        unit_mode = recipe.unit
        if unit_mode isa Function
            return unit_mode(data, header)
        elseif unit_mode == :auto
            bunit = string(get(header, "BUNIT", ""))
            if occursin(r"MJy.*sr"i, bunit)
                # MJy/sr → keep as-is for now (surface brightness)
                # Conversion to e⁻/s requires pixel scale + gain + filter info
                @warn "BUNIT = \"$bunit\" (surface brightness). Values kept as-is; " *
                      "ensure your forward model produces matching units."
                return data
            elseif occursin(r"ELECTRON"i, bunit)
                return data  # already in e⁻/s or e⁻
            elseif occursin(r"ADU"i, bunit) || occursin(r"DN"i, bunit)
                @warn "BUNIT = \"$bunit\" (ADU/DN). Values kept as-is; " *
                      "convert to e⁻/s manually if needed."
                return data
            else
                return data  # unknown unit — pass through
            end
        elseif unit_mode == :none
            return data
        else
            error("Unknown unit mode: $(repr(unit_mode)). Use :auto, :none, or a Function.")
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: ROI extraction
    # ═══════════════════════════════════════════════════════════════

    function _extract_roi(data::AbstractMatrix{T}, wcs::_WCSInfo, header::Dict, recipe::FITSRecipe;
                           center=nothing, radius=nothing) where T
        header = copy(header)  # avoid mutating caller's dict (CRPIX adjustment below)
        nx_data, ny_data = size(data)

        roi_center = nothing
        roi_radius = nothing
        pix_n = min(nx_data, ny_data) - 1

        if radius !== nothing
            roi_radius = radius
            if center === nothing
                cx, cy = div(nx_data, 2), div(ny_data, 2)
            else
                cx = Int(round(center[1]))
                cy = Int(round(center[2]))
            end
            roi_center = (cx, cy)

            x1 = cx - radius
            x2 = cx + radius
            y1 = cy - radius
            y2 = cy + radius

            x1c, x2c = max(1, x1), min(nx_data, x2)
            y1c, y2c = max(1, y1), min(ny_data, y2)

            if (x1c != x1) || (x2c != x2) || (y1c != y1) || (y2c != y2)
                @warn "ROI ($x1:$x2, $y1:$y2) clipped to image bounds → ($x1c:$x2c, $y1c:$y2c)"
            end

            data = data[x1c:x2c, y1c:y2c]
            pix_n = 2 * radius

            # Adjust CRPIX for sub-region offset
            for (key, offset) in [("CRPIX1", x1c - 1), ("CRPIX2", y1c - 1)]
                if haskey(header, key)
                    header[key] = Float64(header[key]) - offset
                end
            end
        else
            if nx_data != ny_data
                @warn "FITS data is rectangular ($(nx_data)×$(ny_data)). " *
                      "Grid will use the smaller side (pix_n=$pix_n). " *
                      "Pass `center` + `radius` to extract a sub-region."
            end
        end

        return data, pix_n, roi_center, roi_radius
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: noise model construction
    # ═══════════════════════════════════════════════════════════════

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

    function _build_noise(data::AbstractMatrix, err_map, header::Dict, recipe::FITSRecipe)
        noise_mode = recipe.noise

        # Level 0: explicit LensNoise
        if noise_mode isa LensNoise
            return noise_mode
        end

        # ── Resolve σ (may be needed by multiple strategies) ──
        sigma = nothing
        if recipe.noise_keyword !== nothing
            val = _kw_get(header, recipe.noise_keyword, nothing)
            if val !== nothing
                sigma = Float64(val)
            end
        end
        if sigma === nothing && noise_mode in (:auto, :gauss, :gauss_poiss, :blind)
            sigma = Float64(estimate_sigma(data; method=recipe.noise_method))
            sigma = max(sigma, 1e-12)
        end

        # ── Dispatch by strategy ──
        if noise_mode == :blind
            return GaussNoise(sigma)

        elseif noise_mode == :gauss
            return GaussNoise(sigma)

        elseif noise_mode == :gauss_poiss
            exp_key = get(recipe.metadata, "EXPTIME", "EXPTIME")
            exp_val = get(header, exp_key, nothing)
            if exp_val === nothing
                @warn "EXPTIME (key \"$exp_key\") not found in header. " *
                      "GaussPoiss noise requires exposure time. " *
                      "Falling back to Gauss-only noise."
                return GaussNoise(sigma)
            end
            return GaussPoissNoise(sigma, Float64(exp_val))

        elseif noise_mode == :from_err
            if err_map !== nothing
                # ERR extension provides per-pixel σ — use Gaussian with
                # per-pixel variance (stored as extra info, not directly LensNoise)
                # For now: take median of ERR map as global σ
                sigma_err = Float64(median(abs.(err_map)))
                return GaussNoise(max(sigma_err, 1e-12))
            else
                @warn "noise=:from_err but no ERR extension found. Falling back to blind estimate."
                return GaussNoise(sigma)
            end

        elseif noise_mode == :auto
            # Priority: ERR extension → header keyword → blind
            if err_map !== nothing
                sigma_err = Float64(median(abs.(err_map)))
                return GaussNoise(max(sigma_err, 1e-12))
            else
                return GaussNoise(sigma)
            end

        else
            error("Unknown noise mode: $(repr(noise_mode))")
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: metadata normalisation
    # ═══════════════════════════════════════════════════════════════

    function _normalise_metadata!(header::Dict, recipe::FITSRecipe)
        for (std_key, file_key) in recipe.metadata
            sk = string(std_key)
            fk = string(file_key)
            if fk != sk && haskey(header, fk)
                header[sk] = header[fk]
            end
        end
        return header
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: PSF resolution
    # ═══════════════════════════════════════════════════════════════

    function _resolve_psf(header::Dict, recipe::FITSRecipe)
        if recipe.psf == :none
            return nothing
        elseif recipe.psf == :auto
            # TODO: match TELESCOP + FILTER → known PSF models
            @warn "psf=:auto is not yet implemented. Returning nothing."
            return nothing
        else
            error("Unknown psf mode: $(repr(recipe.psf))")
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  Internal: merge recipe overrides
    # ═══════════════════════════════════════════════════════════════

    function _override_recipe(recipe::FITSRecipe; kwargs...)
        isempty(kwargs) && return recipe
        pairs = Dict{Symbol, Any}()
        for fn in fieldnames(FITSRecipe)
            pairs[fn] = haskey(kwargs, fn) ? kwargs[fn] : getfield(recipe, fn)
        end
        return FITSRecipe(; pairs...)
    end

    # ═══════════════════════════════════════════════════════════════
    #  read_fits  —  main entry point
    # ═══════════════════════════════════════════════════════════════

    """
        obs = read_fits(path; recipe=FITSRecipe(), pix_size=nothing, ...)

    Read a FITS file into an `Observation`.  The `recipe` describes
    how to interpret the file; every recipe field can be overridden
    by passing the corresponding keyword argument directly.

    # Arguments

    - `path::String`: path to the FITS file
    - `recipe::FITSRecipe`: configuration for how to read the file

    # Convenience overrides (applied on top of recipe)

    | kwarg | overrides | type |
    |-------|-----------|------|
    | `pix_size` | recipe.wcs (forces `:none` + manual scale) | `Real` or `nothing` |
    | `exptime` | exposure time in header | `Real` or `nothing` |
    | `center` | ROI centre [pixels] | `Tuple{Real,Real}` or `nothing` |
    | `radius` | ROI half-side [pixels] | `Int` or `nothing` |
    | `target` | label for this extraction | `String` or `nothing` |

    # Example

    ```julia
    # Quick: HST drizzled image
    obs = read_fits("image_drz.fits"; recipe=HST_WFC3())

    # Custom: ground-based, no WCS, noise from border
    obs = read_fits("keck.fits"; recipe=FITSRecipe(
        wcs          = :none,
        noise        = :blind,
        noise_method = :border,
    ), pix_size=0.04)

    # Override just one field of a preset
    obs = read_fits("image.fits"; recipe=HST_WFC3(), noise=:blind)
    ```
    """
    function read_fits(path::String;
            recipe::FITSRecipe = GenericGround(),
            # ── Convenience overrides ──
            pix_size::Union{Real, Nothing} = nothing,
            exptime::Union{Real, Nothing}  = nothing,
            center::Union{Tuple{Real,Real}, Nothing} = nothing,
            radius::Union{Int, Nothing} = nothing,
            target::Union{String, Nothing} = nothing,
            # ── Recipe field overrides (any FITSRecipe field) ──
            kwargs...,
        )

        # Merge kwargs into recipe (allows e.g. noise=:blind at call site)
        if !isempty(kwargs)
            recipe = _override_recipe(recipe; kwargs...)
        end

        # ── Resolve file extension ──
        if !endswith(path, ".fits") && !endswith(path, ".fits.gz") &&
           !endswith(path, ".fit") && !endswith(path, ".FITS")
            # Only append if the path really looks like a typo, not a deliberate path
            if !isfile(path) && isfile(path * ".fits")
                path = path * ".fits"
            elseif !isfile(path)
                error("File not found: $path (also tried $path.fits)")
            end
        end
        isfile(path) || error("File not found: $path")

        f = FITS(path)

        # ── 1. Collect all headers ──
        header = _collect_all_headers(f)

        # ── 2. Normalise metadata: map file keywords → standard names ──
        _normalise_metadata!(header, recipe)

        # ── 3. Resolve science HDU ──
        sci_idx, sci_hdu = _resolve_hdu(f, recipe.hdu_sci)
        data_raw = read(sci_hdu)
        data = JFloat.(data_raw)

        # ── 4. Read ERR extension (if configured) ──
        err_map = _try_read_err(f, recipe)

        close(f)

        # ── 5. WCS → pixel scale ──
        if pix_size !== nothing
            pixel_scale = Float64(pix_size)
            wcs = _WCSInfo(pixel_scale, [pixel_scale/3600.0 0.0; 0.0 pixel_scale/3600.0],
                           (0.0, 0.0), (0.0, 0.0))
        else
            wcs = _resolve_wcs(header, recipe)
            pixel_scale = wcs.scale
            if pixel_scale < 1e-10
                @warn "Could not determine pixel scale from WCS. " *
                      "Defaulting to 0.04 arcsec/pix. Pass `pix_size=...` to override."
                pixel_scale = 0.04
                wcs = _WCSInfo(pixel_scale, [pixel_scale/3600.0 0.0; 0.0 pixel_scale/3600.0],
                               (0.0, 0.0), (0.0, 0.0))
            end
        end

        # ── 6. Unit normalisation ──
        data = _normalise_units(data, header, recipe)

        # ── 7. ROI extraction ──
        data, pix_n, roi_center, roi_radius = _extract_roi(
            data, wcs, header, recipe; center=center, radius=radius)

        # ── 8. Build coordinate Grid ──
        grid = GenGrid(pix_n=pix_n, pix_size=JFloat(pixel_scale))

        # ── 9. Noise model ──
        noise_model = _build_noise(data, err_map, header, recipe)

        # ── 10. Exposure time ──
        if exptime !== nothing
            exposure_time = Float64(exptime)
        else
            exp_key = get(recipe.metadata, "EXPTIME", "EXPTIME")
            exposure_time = Float64(get(header, exp_key, 0.0))
        end

        # ── 11. Target label ──
        if target !== nothing
            _target = target
        elseif haskey(header, "JENS_TARGET")
            _target = string(header["JENS_TARGET"])
        elseif radius !== nothing
            _target = "roi"
        else
            _target = "full"
        end
        header["JENS_TARGET"] = _target

        # ── 12. PSF ──
        psf = _resolve_psf(header, recipe)

        # ── 13. Mask ──
        mask = all_true_mask(data)

        return Observation(;
            data=data, grid=grid, noise=noise_model, mask=mask,
            psf=psf, exposure_time=exposure_time, header=header,
            target=_target, center=roi_center, radius=roi_radius,
            source_path=path)
    end

    # ═══════════════════════════════════════════════════════════════
    #  read_header  —  metadata-only
    # ═══════════════════════════════════════════════════════════════

    """
        header = read_header(path) -> Dict{String, Any}

    Read only the FITS header without loading pixel data.
    """
    function read_header(path::String)
        if !endswith(path, ".fits") && !endswith(path, ".fits.gz") &&
           !endswith(path, ".fit") && !endswith(path, ".FITS")
            if !isfile(path) && isfile(path * ".fits")
                path = path * ".fits"
            end
        end
        isfile(path) || error("File not found: $path")
        f = FITS(path)
        header = Dict{String, Any}()
        for i in 1:length(f)
            merge!(header, _collect_header(f[i]))
        end
        close(f)
        return header
    end

end # module LensFITSIO