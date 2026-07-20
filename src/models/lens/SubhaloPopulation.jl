"""
    SubhaloPopulation — Efficient Batch Subhalo Rendering

Renders all subhalos in a single GPU kernel via broadcast fusion,
avoiding the per-subhalo kernel-launch overhead.  All subhalos use
the PseudoJaffe profile with per-pixel accumulation.

# Performance
- 1 subhalo   → same as single PseudoJaffe
- 50 subhalos → 2 kernel launches on GPU (vs ~250 before)
- CPU: O(1) grid traversals (vs O(N) before), cache-friendly pixel-first order

# Parameters
- `host`: the main lens (SIE, EPL, CombinedLens, etc.)
- `theta_Es::Vector`: Einstein radii for N subhalos [arcsec]
- `r_ts::Vector`: tidal radii [arcsec]
- `xs::Vector`, `ys::Vector`: centre positions [arcsec]

# Usage
    pop = SubhaloPop(host_lens, theta_Es, r_ts, xs, ys)
    ax, ay = lens_derivative(pop, xg, yg)

# Example
    host = SingleModel(SIE; theta_E=1.2, e1=0.1, e2=0.0)
    pop = SubhaloPop(host, [0.05, 0.03], [0.3, 0.2], [0.5, -0.3], [0.2, -0.4])
"""
# ═══════════════════════════════════════════════════════════════
#  SubhaloPopulation — single-kernel batch rendering via broadcast fusion
#
#  Core idea: instead of looping over subhalos on the CPU and
#  launching one kernel per subhalo, move the per-subhalo loop
#  INSIDE the broadcast function.  CUDA.jl compiles f.(args...)
#  into a single GPU kernel with the inner loop inlined.
#
#  Before (N subhalos):
#    for i in 1:N
#        _sub_deflection!(ax, ay, x, y, ...)  # ~5 kernel launches each
#    end
#    → ~5N kernel launches, N x grid re-reads
#
#  After (N subhalos):
#    ax_sub = _sub_ax.(x, y, Ref(θE), Ref(rt), Ref(xs), Ref(ys), n)  # 1 kernel
#    ay_sub = _sub_ay.(x, y, Ref(θE), Ref(rt), Ref(xs), Ref(ys), n)  # 1 kernel
#    → 2 kernel launches, grid read exactly once per direction
#
#  Ref() wraps each vector as a 0-dimensional "scalar" in broadcast,
#  so the vectors are passed whole into each per-pixel invocation.
# ═══════════════════════════════════════════════════════════════

module SubhaloPopulation

    import Jens.JFloat
    import Jens.LensBase: AbstractLens, lens_derivative, lens_hessian, lens_potential, lens_check

    export SubhaloPop

    # ═══════════════════════════════════════════════════════════
    #  Struct
    # ═══════════════════════════════════════════════════════════

    """
        SubhaloPop(host_lens, theta_Es, r_ts, xs, ys)

    Batch subhalo container.  `host_lens` is the main lens
    (SIE, EPL, CombinedLens, etc.).  Subhalos are PseudoJaffe
    profiles with parameters given as plain Vectors.

    All subhalo parameters must have the same length N.

    Parameters are converted to `T` (default `JFloat` = Float32) at
    construction for GPU performance and consistency with `SingleModel`.
    Pass `T=Float64` for high-precision CPU work.
    """
    struct SubhaloPop{L<:AbstractLens, V<:AbstractVector{<:Real}} <: AbstractLens
        host::L
        theta_E::V
        r_t::V
        xs::V
        ys::V
        n::Int   # number of subhalos
    end

    function SubhaloPop(host::AbstractLens,
                        theta_E::AbstractVector{<:Real},
                        r_t::AbstractVector{<:Real},
                        xs::AbstractVector{<:Real},
                        ys::AbstractVector{<:Real};
                        T::Type{<:AbstractFloat}=JFloat)
        n = length(theta_E)
        @assert length(r_t) == length(xs) == length(ys) == n
        return SubhaloPop(host,
            T.(theta_E), T.(r_t),
            T.(xs), T.(ys), n)
    end

    function lens_check(pop::SubhaloPop; kwargs...)
        lens_check(pop.host; kwargs...)
        @assert all(pop.theta_E .> 0) "all theta_E must be > 0"
        @assert all(pop.r_t .> 0)     "all r_t must be > 0"
    end

    # ═══════════════════════════════════════════════════════════
    #  Per-pixel subhalo accumulators  (broadcast-fused kernels)
    #
    #  Each function computes the total contribution of ALL
    #  subhalos at a single pixel (x, y).  The inner for loop
    #  is inlined by the compiler into the broadcast kernel.
    #
    #  PseudoJaffe radial formulae:
    #    α(R) = θ_E · (R + r_t − √(R²+r_t²)) / R
    #    dα/dR = θ_E · (r_t²/(R²·D) − r_t/R²)    where D = √(R²+r_t²)
    #    ψ(R) = θ_E · [R − D + r_t·log((D+r_t)/(2r_t))]
    # ═══════════════════════════════════════════════════════════

    # ── Deflection x-component ──────────────────────────────────

    @inline function _sub_ax(xv, yv, theta_Es, r_ts, xs, ys, n::Int)
        T = eltype(xv)
        s = zero(T)
        for i in 1:n
            dx = xv - T(xs[i])
            dy = yv - T(ys[i])
            R2 = dx*dx + dy*dy
            R  = max(sqrt(R2), eps(T))
            r  = T(r_ts[i])
            D  = sqrt(R2 + r*r)
            s += T(theta_Es[i]) * (R + r - D) / (R * R) * dx
        end
        return s
    end

    # ── Deflection y-component ──────────────────────────────────

    @inline function _sub_ay(xv, yv, theta_Es, r_ts, xs, ys, n::Int)
        T = eltype(xv)
        s = zero(T)
        for i in 1:n
            dx = xv - T(xs[i])
            dy = yv - T(ys[i])
            R2 = dx*dx + dy*dy
            R  = max(sqrt(R2), eps(T))
            r  = T(r_ts[i])
            D  = sqrt(R2 + r*r)
            s += T(theta_Es[i]) * (R + r - D) / (R * R) * dy
        end
        return s
    end

    # ── Hessian xx-component ────────────────────────────────────
    #  f_xx = Σ (dα/dR · cos²φ + α/R · sin²φ)

    @inline function _sub_hxx(xv, yv, theta_Es, r_ts, xs, ys, n::Int)
        T = eltype(xv)
        s = zero(T)
        for i in 1:n
            dx = xv - T(xs[i])
            dy = yv - T(ys[i])
            R2 = dx*dx + dy*dy
            R  = max(sqrt(R2), eps(T))
            r  = T(r_ts[i])
            tE = T(theta_Es[i])
            D  = sqrt(R2 + r*r)
            a     = tE * (R + r - D) / R
            a_R   = a / R
            da_dR = tE * (r*r / (R2 * D) - r / R2)
            cos_phi = dx / R
            sin_phi = dy / R
            s += da_dR * cos_phi*cos_phi + a_R * sin_phi*sin_phi
        end
        return s
    end

    # ── Hessian xy-component ────────────────────────────────────
    #  f_xy = Σ (dα/dR − α/R) · sinφ · cosφ

    @inline function _sub_hxy(xv, yv, theta_Es, r_ts, xs, ys, n::Int)
        T = eltype(xv)
        s = zero(T)
        for i in 1:n
            dx = xv - T(xs[i])
            dy = yv - T(ys[i])
            R2 = dx*dx + dy*dy
            R  = max(sqrt(R2), eps(T))
            r  = T(r_ts[i])
            tE = T(theta_Es[i])
            D  = sqrt(R2 + r*r)
            a     = tE * (R + r - D) / R
            a_R   = a / R
            da_dR = tE * (r*r / (R2 * D) - r / R2)
            cos_phi = dx / R
            sin_phi = dy / R
            s += (da_dR - a_R) * sin_phi * cos_phi
        end
        return s
    end

    # ── Hessian yy-component ────────────────────────────────────
    #  f_yy = Σ (dα/dR · sin²φ + α/R · cos²φ)

    @inline function _sub_hyy(xv, yv, theta_Es, r_ts, xs, ys, n::Int)
        T = eltype(xv)
        s = zero(T)
        for i in 1:n
            dx = xv - T(xs[i])
            dy = yv - T(ys[i])
            R2 = dx*dx + dy*dy
            R  = max(sqrt(R2), eps(T))
            r  = T(r_ts[i])
            tE = T(theta_Es[i])
            D  = sqrt(R2 + r*r)
            a     = tE * (R + r - D) / R
            a_R   = a / R
            da_dR = tE * (r*r / (R2 * D) - r / R2)
            cos_phi = dx / R
            sin_phi = dy / R
            s += da_dR * sin_phi*sin_phi + a_R * cos_phi*cos_phi
        end
        return s
    end

    # ── Lensing potential ───────────────────────────────────────
    #  ψ(R) = θ_E · [R − D + r_t · log((D + r_t) / (2 r_t))]

    @inline function _sub_psi(xv, yv, theta_Es, r_ts, xs, ys, n::Int)
        T = eltype(xv)
        s = zero(T)
        for i in 1:n
            dx = xv - T(xs[i])
            dy = yv - T(ys[i])
            R2 = dx*dx + dy*dy
            R  = max(sqrt(R2), eps(T))
            r  = T(r_ts[i])
            D  = sqrt(R2 + r*r)
            s += T(theta_Es[i]) * (R - D + r * log((D + r) / (T(2) * r)))
        end
        return s
    end

    # ═══════════════════════════════════════════════════════════
    #  LensBase interface — broadcast-fused single-kernel dispatch
    # ═══════════════════════════════════════════════════════════

    function lens_derivative(pop::SubhaloPop, x, y; kwargs...)
        # ① Host lens
        ax, ay = lens_derivative(pop.host, x, y; kwargs...)

        # ② All subhalos — 1 kernel per direction (fused inner loop)
        if pop.n > 0
            ax_sub = _sub_ax.(x, y, Ref(pop.theta_E), Ref(pop.r_t),
                               Ref(pop.xs), Ref(pop.ys), pop.n)
            ay_sub = _sub_ay.(x, y, Ref(pop.theta_E), Ref(pop.r_t),
                               Ref(pop.xs), Ref(pop.ys), pop.n)
            ax .+= ax_sub
            ay .+= ay_sub
        end
        return ax, ay
    end

    function lens_hessian(pop::SubhaloPop, x, y; kwargs...)
        # ① Host lens
        fxx, fxy, fyy = lens_hessian(pop.host, x, y; kwargs...)

        # ② All subhalos — 1 kernel per Hessian component
        if pop.n > 0
            hxx = _sub_hxx.(x, y, Ref(pop.theta_E), Ref(pop.r_t),
                             Ref(pop.xs), Ref(pop.ys), pop.n)
            hxy = _sub_hxy.(x, y, Ref(pop.theta_E), Ref(pop.r_t),
                             Ref(pop.xs), Ref(pop.ys), pop.n)
            hyy = _sub_hyy.(x, y, Ref(pop.theta_E), Ref(pop.r_t),
                             Ref(pop.xs), Ref(pop.ys), pop.n)
            fxx .+= hxx
            fxy .+= hxy
            fyy .+= hyy
        end
        return fxx, fxy, fyy
    end

    function lens_potential(pop::SubhaloPop, x, y; kwargs...)
        # ① Host lens
        psi = lens_potential(pop.host, x, y; kwargs...)

        # ② All subhalos — 1 kernel
        if pop.n > 0
            psi_sub = _sub_psi.(x, y, Ref(pop.theta_E), Ref(pop.r_t),
                                 Ref(pop.xs), Ref(pop.ys), pop.n)
            psi .+= psi_sub
        end
        return psi
    end

end # module SubhaloPopulation