# ═══════════════════════════════════════════════════════════════
#  LensNoise — noise models for simulation & MCMC likelihood
#
#  Noise types (new):
#    GaussNoise(σ)          Gaussian noise with fixed σ
#    PoissNoise(exp_time)   Gaussian-approximated Poisson noise
#
#  Functions (preserved):
#    GaussianNoise(image, sigma_bkd)   → noise array
#    PoissonNoise(image, exp_time)     → noise array  
#    BackgroundNoise(img; ...)         → (bkg_map, bkg_rms)
#
#  Functions (new):
#    add_noise(image, noise)           → image + noise sample
#    log_likelihood(data, model, n)    → log p(data | model)
#    log_likelihood_gpu(data, model, n)→ GPU-friendly variant
#    estimate_sigma(data)              → robust noise estimation
# ═══════════════════════════════════════════════════════════════

module LensNoise

using Statistics, Random, Distributions

export GaussNoise, PoissNoise, GaussPoissNoise, LensNoise
export GaussianNoise, PoissonNoise, BackgroundNoise
export add_noise, log_likelihood, log_likelihood_gpu
export estimate_sigma, per_pixel_variance

# ═══════════════════════════════════════════════════════════════
#  Noise model types (new API)
# ═══════════════════════════════════════════════════════════════

abstract type LensNoise end

"""
    GaussNoise(σ::Real)

Homoskedastic Gaussian noise with standard deviation `σ`.

    noise = GaussNoise(0.05)
    noisy = add_noise(data, noise)
    logp  = log_likelihood(data, model, noise)
"""
struct GaussNoise{T<:Real} <: LensNoise
    σ::T
    # Inner constructor: auto-promote to Float64
    function GaussNoise(σ::Real)
        return new{Float64}(Float64(σ))
    end
end

"""
    PoissNoise(exp_time::Real)

Poisson noise in the Gaussian approximation: σ² = |image| / exp_time.

For low-count regimes (λ < 10), the Gaussian approximation breaks down;
use true Poisson noise via `Distributions.Poisson` instead.

    noise = PoissNoise(300.0)   # 300 second exposure
    noisy = add_noise(data, noise)
    logp  = log_likelihood(data, model, noise)
"""
struct PoissNoise{T<:Real} <: LensNoise
    exp_time::T
    # Inner constructor: auto-promote to Float64
    function PoissNoise(exp_time::Real)
        return new{Float64}(Float64(exp_time))
    end
end

"""
    GaussPoissNoise(sigma_gauss::Real, exp_time::Real)

Combined Gaussian + Poisson noise model for real astronomical images.

    sigma2[i] = sigma_gauss² + |model[i]| / exp_time

The Gaussian term covers read noise, dark current, and sky background
(all approximately constant across pixels).  The Poisson term is the
photon shot noise from the source itself, using the model flux (not
the data) for the variance — correct for MCMC likelihood evaluation.

# Example
    noise = GaussPoissNoise(0.001, 9939.6)
    noisy = add_noise(data, noise)
    logp  = log_likelihood(data, model, noise)
"""
struct GaussPoissNoise{T<:Real} <: LensNoise
    sigma_gauss::T  # Gaussian component σ [same units as data, e.g. e⁻/s]
    exp_time::T     # exposure time [s]
    function GaussPoissNoise(sigma_gauss::Real, exp_time::Real)
        return new{Float64}(Float64(sigma_gauss), Float64(exp_time))
    end
end


# ═══════════════════════════════════════════════════════════════
#  add_noise  —  image + noise
# ═══════════════════════════════════════════════════════════════

"""
    noisy = add_noise(image, noise::LensNoise)

Add a noise realisation to `image`.  Works on CPU `Array` and GPU `CuArray`.

# Example
    noisy = add_noise(truth_img, GaussNoise(0.05))
    noisy = add_noise(truth_img, PoissNoise(100.0))
"""
function add_noise(image::AbstractArray, gn::GaussNoise)
    return image .+ _randn_like(image) .* gn.σ
end

function add_noise(image::AbstractArray, pn::PoissNoise)
    # σ² = |μ| / exp_time  (Gaussian approximation of Poisson)
    σ = @. sqrt(max(abs(image), 0)) / sqrt(pn.exp_time)
    return image .+ _randn_like(image) .* σ
end

function add_noise(image::AbstractArray, np::GaussPoissNoise)
    # sigma2[i] = sigma_gauss² + |image[i]| / exp_time
    sigma2_g = np.sigma_gauss^2
    inv_t = 1 / np.exp_time
    sigma = @. sqrt(sigma2_g + max(abs(image), 0) * inv_t)
    return image .+ _randn_like(image) .* sigma
end


# ═══════════════════════════════════════════════════════════════
#  log_likelihood  —  for MCMC
# ═══════════════════════════════════════════════════════════════

"""
    lp = log_likelihood(data, model, noise::LensNoise)

Log-likelihood  log p(data | model)  under the noise model.
Drops the normalisation constant  -½N log(2π).

# Example
    noise = GaussNoise(0.05)
    logp  = log_likelihood(data_noisy, model_img, noise)
"""
function log_likelihood(data::AbstractArray, model::AbstractArray,
                        gn::GaussNoise)
    σ² = gn.σ^2
    return -sum((data .- model).^2) / (2σ²)
end

function log_likelihood(data::AbstractArray, model::AbstractArray,
                        pn::PoissNoise)
    # Heteroskedastic: Var_i = |model_i| / exp_time
    # Use model (not data) for variance — correct for MCMC
    inv_t = 1 / pn.exp_time
    # Guard against zero model flux (degenerate variance)
    residual² = (data .- model).^2
    var_model = @. max(abs(model) * inv_t, 1e-20)
    return -sum(residual² ./ var_model) / 2
end

function log_likelihood(data::AbstractArray, model::AbstractArray,
                        np::GaussPoissNoise)
    # Var_i = sigma_gauss² + |model_i| / exp_time
    sigma2_g = np.sigma_gauss^2
    inv_t = 1 / np.exp_time
    residual2 = (data .- model).^2
    var_model = @. sigma2_g + max(abs(model) * inv_t, 0.0)
    return -sum(residual2 ./ var_model) / 2
end

"""
    lp = log_likelihood_gpu(data, model, noise::GaussNoise)

GPU-friendly log-likelihood.  Avoids scalar indexing; broadcasts entirely
on device when `data` and `model` are `CuArray`.

    # Works on both CPU and GPU:
    logp = log_likelihood_gpu(data_gpu, model_gpu, GaussNoise(0.05f0))
"""
function log_likelihood_gpu(data::AbstractArray, model::AbstractArray,
                             gn::GaussNoise)
    T = eltype(data)
    σ² = T(gn.σ^2)
    diff² = (data .- model).^2
    return -T(0.5) * sum(diff²) / σ²
end

function log_likelihood_gpu(data::AbstractArray, model::AbstractArray,
                             np::GaussPoissNoise)
    T = eltype(data)
    sigma2_g = T(np.sigma_gauss^2)
    inv_t = T(1 / np.exp_time)
    diff2 = (data .- model).^2
    var_model = @. sigma2_g + max(abs(model) * inv_t, T(0))
    return -T(0.5) * sum(diff2 ./ var_model)
end

# ═══════════════════════════════════════════════════════════════
#  per_pixel_variance  —  helper for masked_logp in LensSystem
# ═══════════════════════════════════════════════════════════════

"""
    var = per_pixel_variance(model, noise::LensNoise)

Return per-pixel variance array for a given noise model.
Used by `masked_logp` / `masked_chi2` to compute weighted χ²
without knowing the noise model internals.

# Example
    var = per_pixel_variance(model_img, GaussNoise(0.05))
    var = per_pixel_variance(model_img, GaussPoissNoise(0.001, 9939.6))
"""
function per_pixel_variance(model::AbstractArray, gn::GaussNoise)
    T = eltype(model)
    return fill(T(gn.σ^2), size(model))
end

function per_pixel_variance(model::AbstractArray, pn::PoissNoise)
    inv_t = 1 / pn.exp_time
    return @. max(abs(model) * inv_t, eltype(model)(1e-20))
end

function per_pixel_variance(model::AbstractArray, np::GaussPoissNoise)
    T = eltype(model)
    sigma2_g = T(np.sigma_gauss^2)
    inv_t = T(1 / np.exp_time)
    return @. sigma2_g + max(abs(model) * inv_t, T(0))
end


# ═══════════════════════════════════════════════════════════════
#  estimate_sigma  —  robust noise estimation from data
# ═══════════════════════════════════════════════════════════════

"""
    σ = estimate_sigma(data; method=:mad, kwargs...)

Robust noise estimation from pixel values.

# Methods

| method       | description | extra kwargs |
|--------------|-------------|--------------|
| `:mad`       | Median Absolute Deviation (default, robust to signal) | — |
| `:clipped`   | σ-clipping: iteratively reject outliers | `clip_sigma=3.0`, `max_iter=5` |
| `:border`    | estimate from image border pixels (assumes blank sky) | `border_width=10` |
| `:masked`    | estimate from pixels where `mask == true` | `mask::BitMatrix` |
| `:fraction`  | fraction-of-median (fast, crude) | `fraction=0.1` |

# Examples
    sigma = estimate_sigma(data; method=:mad)
    sigma = estimate_sigma(data; method=:clipped, clip_sigma=3.0)
    sigma = estimate_sigma(data; method=:border, border_width=15)
    sigma = estimate_sigma(data; method=:masked, mask=bg_mask)
"""
function estimate_sigma(data::AbstractArray; method::Symbol=:mad, kwargs...)
    if method == :mad
        return _sigma_mad(data)
    elseif method == :clipped
        return _sigma_clipped(data; kwargs...)
    elseif method == :border
        return _sigma_border(data; kwargs...)
    elseif method == :masked
        return _sigma_masked(data; kwargs...)
    elseif method == :fraction
        return _sigma_fraction(data; kwargs...)
    else
        error("Unknown method: $method. Use :mad, :clipped, :border, :masked, or :fraction.")
    end
end

# ── Individual estimators ──

"""
    _sigma_mad(data) -> Float64

Median Absolute Deviation: σ = 1.4826 × median(|data - median(data)|).
"""
function _sigma_mad(data::AbstractArray)
    mu_val = median(data)
    mad_val = median(abs.(data .- mu_val))
    sigma = mad_val * 1.4826
    return max(Float64(sigma), 1e-12)
end

"""
    _sigma_clipped(data; clip_sigma=3.0, max_iter=5) -> Float64

Sigma-clipped standard deviation: iteratively compute σ, discard
pixels beyond `clip_sigma × σ` from the median, and recompute.
Useful when the image contains bright sources (stars, lensed arcs)
that would otherwise bias the noise estimate upward.
"""
function _sigma_clipped(data::AbstractArray;
                         clip_sigma::Real=3.0,
                         max_iter::Int=5)
    arr = vec(Float64.(data))
    mu_val = median(arr)
    # Start with MAD (robust), not std (biased by bright sources)
    sigma = _sigma_mad(data)

    for _ in 1:max_iter
        keep = abs.(arr .- mu_val) .<= clip_sigma * sigma
        if sum(keep) < 10
            break   # too few pixels left
        end
        arr = arr[keep]
        mu_val = median(arr)
        sigma_new = std(arr)
        # Converged?
        if abs(sigma_new - sigma) / max(sigma, 1e-12) < 0.01
            sigma = sigma_new
            break
        end
        sigma = sigma_new
    end
    return max(sigma, 1e-12)
end

"""
    _sigma_border(data; border_width=10) -> Float64

Estimate noise from image border pixels only.  Assumes the borders
contain blank sky (common for HST postage stamps centred on a lens).

Pixels within `border_width` of any image edge are used.
"""
function _sigma_border(data::AbstractArray;
                        border_width::Int=10)
    nx, ny = size(data)
    bw = border_width

    # Build a mask for border pixels
    mask = fill(false, nx, ny)
    mask[1:bw, :] .= true              # top
    mask[end-bw+1:end, :] .= true      # bottom
    mask[:, 1:bw] .= true              # left
    mask[:, end-bw+1:end] .= true      # right

    vals = Float64.(data[mask])
    sigma = _sigma_mad(vals)
    # If MAD ≈ 0 (extremely uniform data, e.g. drizzled sky),
    # fall back to full-image MAD with a warning.
    if sigma < 1e-10
        @warn "Border pixels too uniform (drizzled data?). Falling back to full-image MAD."
        sigma = _sigma_mad(data)
    end
    return max(sigma, 1e-12)
end

"""
    _sigma_masked(data; mask::BitMatrix) -> Float64

Estimate noise from pixels where `mask` is `true`.  The user is
responsible for defining a mask that covers only background regions.
"""
function _sigma_masked(data::AbstractArray;
                        mask::BitMatrix)
    size(data) == size(mask) || throw(ArgumentError(
        "data size $(size(data)) ≠ mask size $(size(mask))"))
    vals = Float64.(data[mask])
    sigma = _sigma_mad(vals)
    # If MAD ≈ 0 (extremely uniform data, e.g. drizzled sky),
    # fall back to full-image MAD with a warning.
    if sigma < 1e-10
        @warn "Border pixels too uniform (drizzled data?). Falling back to full-image MAD."
        sigma = _sigma_mad(data)
    end
    return max(sigma, 1e-12)
end

"""
    _sigma_fraction(data; fraction=0.1) -> Float64

Crude estimate: σ = fraction × median(|data|).
Fast but assumes the data mean is near zero.
"""
function _sigma_fraction(data::AbstractArray;
                          fraction::Real=0.1)
    sigma = Float64(fraction) * median(abs.(data))
    return max(sigma, 1e-12)
end


# ═══════════════════════════════════════════════════════════════
#  Legacy functions  (preserved — unchanged API)
# ═══════════════════════════════════════════════════════════════

"""
    noise_arr = GaussianNoise(image::AbstractArray, sigma_bkd::Real)

Legacy: return a noise *array* (not image + noise).
Prefer `add_noise(image, GaussNoise(sigma))` for new code.
"""
function GaussianNoise(image::AbstractArray, sigma_bkd::Real)
    return sigma_bkd .* randn(Float64, size(image))
end

"""
    noise_arr = PoissonNoise(image::AbstractArray, exp_time::Real)

Legacy: return a noise *array* (not image + noise).
Gaussian approximation of Poisson shot noise: σ² = |image|/exp_time.
Prefer `add_noise(image, PoissNoise(exp_time))` for new code.
"""
function PoissonNoise(image::AbstractArray, exp_time::Real)
    sigma = @. sqrt(abs(image) / exp_time)
    return randn(Float64, size(image)) .* sigma
end

"""
    bkg, bkg_rms = BackgroundNoise(img; kw...)

Legacy: sigma-clipping + background estimation.  Requires AstroLib.
"""
function BackgroundNoise(img::AbstractArray;
    clipara::Dict=Dict(:fill=>NaN, :center=>median(img),
                        :std=>std(img, corrected=false)),
    sigma::Real=1, boxsize::Real=50, filtersize::Real=5)

    clipped = AstroLib.sigma_clip(img, sigma; clipara...)
    bkg, bkg_rms = AstroLib.estimate_background(
        clipped, boxsize, filter_size=filtersize)
    return bkg, bkg_rms
end


# ═══════════════════════════════════════════════════════════════
#  Internal: GPU-compatible randn
# ═══════════════════════════════════════════════════════════════

function _randn_like(image::AbstractArray{T}) where T
    if image isa Array
        return randn(Float64, size(image))
    else
        # GPU path: generate on CPU, transfer to device
        noise_cpu = randn(Float64, size(image))
        noise_dev = similar(image, Float64)
        copyto!(noise_dev, noise_cpu)
        return noise_dev
    end
end

end # module