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

export GaussNoise, PoissNoise
export GaussianNoise, PoissonNoise, BackgroundNoise
export add_noise, log_likelihood, log_likelihood_gpu
export estimate_sigma

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
end
# Outer constructor: auto-promote to Float64
GaussNoise(σ::Real) = GaussNoise{Float64}(Float64(σ))

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
end
# Outer constructor: auto-promote to Float64
PoissNoise(exp_time::Real) = PoissNoise{Float64}(Float64(exp_time))


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


# ═══════════════════════════════════════════════════════════════
#  estimate_sigma  —  robust noise estimation from data
# ═══════════════════════════════════════════════════════════════

"""
    σ = estimate_sigma(data; method=:mad, fraction=0.1)

Robust noise estimation from pixel values.  Two methods:

- `:mad` — Median Absolute Deviation  (default, robust to signal)
  `σ = 1.4826 × median(|data - median(data)|)`

- `:fraction` — fraction-of-median (fast, crude)
  `σ = fraction × median(|data|)`

# Example
    sigma = estimate_sigma(data_noisy; method=:mad)
    noise = GaussNoise(sigma)
"""
function estimate_sigma(data::AbstractArray; method::Symbol=:mad,
                         fraction::Real=0.1)
    if method == :mad
        # MAD → σ for Gaussian noise
        μ = median(data)
        mad = median(abs.(data .- μ))
        σ = mad * 1.4826
        return max(float(σ), 1e-12)
    elseif method == :fraction
        σ = fraction * median(abs.(data))
        return max(float(σ), 1e-12)
    else
        error("Unknown method: $method. Use :mad or :fraction.")
    end
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