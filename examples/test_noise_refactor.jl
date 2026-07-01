#!/usr/bin/env julia
# Quick verification: LensNoise old + new API
using Jens.LensNoise, Statistics, Random, Test

Random.seed!(42)

# ── 1. Old API still works ──────────────────────────────────
img = rand(64, 64)

noise_g = GaussianNoise(img, 0.05)
@test size(noise_g) == (64, 64)
@test abs(mean(noise_g)) < 0.01  # zero mean
println("✓ GaussianNoise (legacy): mean=$(round(mean(noise_g), digits=5)), σ≈$(round(std(noise_g), digits=4))")

noise_p = PoissonNoise(img, 100.0)
@test size(noise_p) == (64, 64)
println("✓ PoissonNoise  (legacy): mean=$(round(mean(noise_p), digits=5)), σ≈$(round(std(noise_p), digits=4))")

# ── 2. New API: add_noise ────────────────────────────────────
gn = GaussNoise(0.05)
noisy_g = add_noise(img, gn)
@test size(noisy_g) == (64, 64)
@test noisy_g != img  # actually modified
println("✓ add_noise + GaussNoise: max|diff|=$(round(maximum(abs.(noisy_g .- img)), digits=4))")

pn = PoissNoise(100.0)
noisy_p = add_noise(img, pn)
@test size(noisy_p) == (64, 64)
println("✓ add_noise + PoissNoise: max|diff|=$(round(maximum(abs.(noisy_p .- img)), digits=4))")

# ── 3. log_likelihood consistency ────────────────────────────
model_good = img .+ 0.01            # close
model_bad  = img .+ 0.1             # far
lp_good = log_likelihood(img, model_good, GaussNoise(0.05))
lp_bad  = log_likelihood(img, model_bad,  GaussNoise(0.05))
@test lp_good > lp_bad   # closer model → higher logp
@test lp_good < 0
println("✓ log_likelihood GaussNoise: lp(good)=$(round(lp_good, digits=1)), lp(bad)=$(round(lp_bad, digits=1))")

lp_p = log_likelihood(img, model_good, PoissNoise(100.0))
@test lp_p < 0
println("✓ log_likelihood PoissNoise: lp=$(round(lp_p, digits=1))")

# ── 4. log_likelihood_gpu (CPU path) ──────────────────────────
lp_gpu = log_likelihood_gpu(img, model_good, GaussNoise(0.05))
@test abs(lp_gpu - lp_good) < 1e-10  # should match regular version
println("✓ log_likelihood_gpu matches regular: Δ=$(round(abs(lp_gpu - lp_good), digits=12))")

# ── 5. estimate_sigma ─────────────────────────────────────────
truth = 0.05
const_img = ones(64, 64)          # constant → MAD only sees noise
noisy_data = const_img .+ truth .* randn(size(const_img))
σ_mad = estimate_sigma(noisy_data; method=:mad)
σ_frac = estimate_sigma(noisy_data; method=:fraction, fraction=0.1)
println("✓ estimate_sigma(:mad)     : $(round(σ_mad, digits=4))  (truth=$truth)")
println("✓ estimate_sigma(:fraction): $(round(σ_frac, digits=4))  (truth=$truth)")
@test 0.03 < σ_mad < 0.07

# ── 6. Type stability ─────────────────────────────────────────
@test GaussNoise(0.05) isa LensNoise
@test PoissNoise(100.0) isa LensNoise
@test GaussNoise(0.05f0) isa LensNoise  # Float32
println("✓ Type hierarchy correct (GaussNoise <: LensNoise, PoissNoise <: LensNoise)")

println("\n✅ All tests passed.")