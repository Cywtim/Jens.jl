# ═══════════════════════════════════════════════════════════════
#  SamplingUtil — generic chain-processing utilities
#
#  Extends `chain`, `burnin`, `chain_stats` to work with:
#    • Raw Matrix{Float64} (n_params × n_samples)
#    • MCMCChains.Chains (when MCMCChains is loaded)
#
#  LensMH.MHResult is handled by LensMH itself.
# ═══════════════════════════════════════════════════════════════

module SamplingUtil

    using Statistics

    export chain, burnin, chain_stats

    # ══════════════════════════════════════════════════════════
    #  Raw Matrix  (n_params × n_samples)
    # ══════════════════════════════════════════════════════════

    """
        S = chain(M::Matrix{Float64}; burn=0)

    Extract posterior from a `(n_params, n_samples)` matrix.
    """
    function chain(M::Matrix{Float64}; burn::Int=0)
        burn = max(0, min(burn, size(M, 2) - 1))
        return M[:, burn+1:end]
    end

    """
        S = burnin(M::Matrix{Float64}, n::Int)

    Drop first `n` warm-up samples.
    """
    burnin(M::Matrix{Float64}, n::Int) = chain(M; burn=n)

    """
        (means, stds) = chain_stats(M::Matrix{Float64}; burn=0)

    Posterior mean and standard deviation per parameter.
    """
    function chain_stats(M::Matrix{Float64}; burn::Int=0)
        S = chain(M; burn=burn)
        m = vec(mean(S; dims=2))
        s = vec(std(S; dims=2))
        return m, s
    end

    # ══════════════════════════════════════════════════════════
    #  MCMCChains.Chains  (loaded conditionally)
    #
    #  These become available when the user does `using MCMCChains`.
    #  Julia will dispatch to them only when MCMCChains is loaded.
    # ══════════════════════════════════════════════════════════

    # function chain(chn::MCMCChains.Chains; burn::Int=0)
    #     arr = Array(chn)  # (n_iter, n_params, n_chains)
    #     # Squeeze single-chain and transpose → (n_params, n_iter)
    #     M = dropdims(arr; dims=3)''
    #     return chain(M; burn=burn)
    # end

    # function chain_stats(chn::MCMCChains.Chains; burn::Int=0)
    #     return chain_stats(chain(chn; burn=burn))
    # end

    # burnin(chn::MCMCChains.Chains, n::Int) = chain(chn; burn=n)

end # module SamplingUtil