module LensPSF

    using PSFModels, ImageFiltering, Optim, SpecialFunctions

    export GaussianPSF, MoffatPSF, AirydiskPSF

    _fwhm_half(fwhm, scale=4.0) = ceil(Int, scale * fwhm / 2)

    function _make_grid(half::Int)
        xs = -half:half
        return xs, xs
    end

    function GaussianPSF(;
            fwhm::Real = 5.0,
            x::Real    = 0.0,
            y::Real    = 0.0,
            half::Int  = 0,
        )
        h  = half == 0 ? _fwhm_half(fwhm, 3.0) : half
        xs, ys = _make_grid(h)
        k = [PSFModels.gaussian(px, py; fwhm, x, y) for px in xs, py in ys]
        return k ./ sum(k)
    end

    function MoffatPSF(;
            fwhm::Real  = 5.0,
            alpha::Real = 3.0,
            x::Real     = 0.0,
            y::Real     = 0.0,
            half::Int   = 0,
        )
        h  = half == 0 ? _fwhm_half(fwhm, 5.0) : half
        xs, ys = _make_grid(h)
        k = [PSFModels.moffat(px, py; fwhm, alpha, x, y) for px in xs, py in ys]
        return k ./ sum(k)
    end

    function AirydiskPSF(;
            fwhm::Real = 5.0,
            x::Real    = 0.0,
            y::Real    = 0.0,
            half::Int  = 0,
        )
        h  = half == 0 ? _fwhm_half(fwhm, 5.0) : half
        xs, ys = _make_grid(h)
        k = [PSFModels.airydisk(px, py; fwhm, x, y) for px in xs, py in ys]
        return k ./ sum(k)
    end


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

        # ---- initial parameters per model ----
        p0 = (model == :moffat ?
             [fwhm0, alpha0, amp0, x0, y0, bkg0] :
             [fwhm0, amp0, x0, y0, bkg0])

        # ---- loss function ----
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
            else  # :airy
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

        # ---- reconstruct kernel & residual ----
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

end
