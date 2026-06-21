module LensUtils

    using LinearAlgebra

    export  LensGrid, LensPolGrid, LensInverse, LensRotation
    export  Pol2Car, Car2Pol
    export  e2phiq, phiq2e
    export  ShearPol2Car, ShearCar2Pol
    export  EllipticalDistortion
    export  PoissonNoise, GaussianNoise, BackgroundNoise
    export  include_folder
    export  name2str
    export ndgrid

    # ═══════════════════════════════════════════════════════════════
    #  1. MATRIX INVERSE (using rcond check instead of try/catch)
    # ═══════════════════════════════════════════════════════════════

    """
        MatrixInverse(M, e=1e-10)

    Safe inverse: if M is singular (rcond ≤ e), regularise with M + e·I.
    """
    function MatrixInverse(M::AbstractMatrix{T}, e::Float64=1e-10) where T
        if rcond(M) > eps(real(T))^(2/3)
            return inv(M)
        else
            return inv(M + e * I)
        end
    end

    # ═══════════════════════════════════════════════════════════════
    #  2. COORDINATE GRIDS (lazy 1D vectors → GPU/threading ready)
    # ═══════════════════════════════════════════════════════════════

    """
        LensGrid(; xl, nx=100, yl=nothing, ny=nothing, CarOut=true)

    Return 2D Cartesian or polar grid on [-xl,xl]×[-yl,yl] as Matrices.
    """
    function LensGrid(; xl, nx=100, yl=nothing, ny=nothing, CarOut=true)
        if yl === nothing
            yl = xl
        end
        if ny === nothing
            ny = nx
        end
        xg = _build_range(xl, nx)
        yg = _build_range(yl, ny)
        if CarOut
            return ndgrid(collect(xg), collect(yg))
        else
            r, phi = Car2Pol(ndgrid(collect(xg), collect(yg))...)
            return r, phi
        end
    end

    """
        LensPolGrid(; rl, thetal=2pi, nr=100, ntheta=100, Polout=true)

    Return a polar grid in either polar or Cartesian output.
    """
    function LensPolGrid(; rl, thetal=2pi, nr=100, ntheta=100, Polout=true)
        rg = _build_range(rl, nr)
        thetag = range(0, thetal; length=ntheta)
        if Polout
            return ndgrid(collect(rg), collect(thetag))
        else
            Rgrid, Theta = ndgrid(collect(rg), collect(thetag))
            return Pol2Car(Rgrid, Theta)
        end
    end

    # Internal: build a 1D range from scalar (asymmetric) or 2-element array
    _build_range(x::Real, n::Int) = range(-x, x; length=n)
    _build_range(x::AbstractVector, n::Int) = range(x[1], x[2]; length=n)

    """
        ndgrid(x, y)

    Build 2D grid matrices from 1D vectors (like MATLAB's ndgrid).
    Uses broadcasting — works on both CPU and GPU arrays.
    """
    function ndgrid(x::AbstractVector{T}, y::AbstractVector{T}) where T
        nx, ny = length(x), length(y)
        X = similar(x, T, nx, ny)
        Y = similar(y, T, nx, ny)
        X .= reshape(x, :, 1)
        Y .= reshape(y, 1, :)
        return X, Y
            end

    # ═══════════════════════════════════════════════════════════════
    #  3. ROTATION
    # ═══════════════════════════════════════════════════════════════

    """
        LensRotation(xg, yg, varphi)

    Rotate (xg, yg) by angle varphi (radians, counter-clockwise).
    """
    function LensRotation(xg, yg, varphi)
        xr = @. cos(varphi) * xg - sin(varphi) * yg
        yr = @. sin(varphi) * xg + cos(varphi) * yg
        return xr, yr
    end

    # ═══════════════════════════════════════════════════════════════
 # 4. COORDINATE TRANSFORMATIONS
    # ═══════════════════════════════════════════════════════════════

    """
        Car2Pol(x, y; xc=0, yc=0)

    Cartesian → Polar. Uses 2-argument `atan(y, x)` to avoid Inf/NaN quadrant issues.
    """
    function Car2Pol(x, y; xc=0.0, yc=0.0)
        xsh = x .- xc
        ysh = y .- yc
        r = @. sqrt(xsh^2 + ysh^2)
        phi = @. atan(ysh, xsh)
        return r, phi
    end

    """
        Pol2Car(r, phi; xc=0, yc=0)

    Polar → Cartesian. Inverse of Car2Pol (adds center back).
    """
    function Pol2Car(r, phi; xc=0.0, yc=0.0)
        x = @. r * cos(phi) + xc
        y = @. r * sin(phi) + yc
        return x, y
    end

    # ═══════════════════════════════════════════════════════════════
 # 5. SHEAR CONVENTIONS (spin-2 field)
    # ═══════════════════════════════════════════════════════════════

    """
        ShearPol2Car(phi, gamma)

    Shear from polar (phi_gamma, |gamma|) → (gamma1, gamma2).
    gamma1 = gamma·cos(2*phi), gamma2 = gamma·sin(2*phi).
    """
    function ShearPol2Car(phi, gamma)
        gamma1 = @. gamma * cos(2 * phi)
        gamma2 = @. gamma * sin(2 * phi)
        return gamma1, gamma2
    end

    """
        ShearCar2Pol(gamma1, gamma2)

    Shear from Cartesian (gamma1, gamma2) → (phi_gamma, |gamma|).
    """
    function ShearCar2Pol(gamma1, gamma2)
        phi = @. atan(gamma2, gamma1) / 2
        gamma_mag = @. sqrt(gamma1^2 + gamma2^2)
        return phi, gamma_mag
    end

    # ═══════════════════════════════════════════════════════════════
    #  6. ELLIPTICITY CONVENTIONS (matches lenstronomy)
    # ═══════════════════════════════════════════════════════════════

    """
        e2phiq(e1, e2)

    Ellipticity moduli → (axis ratio q, position angle phi in rad).
    """
    function e2phiq(e1, e2)
        phi = @. atan(e2, e1) / 2
        e = @. sqrt(e1^2 + e2^2)
        e = min.(e, 0.9999)
        q = @. (1 - e) / (1 + e)
        return q, phi
    end

    """
        phiq2e(q, varphi)

    Axis ratio + angle → ellipticity moduli (e₁, e₂).
    """
    function phiq2e(q, varphi)
        e1 = @. (1 - q) / (1 + q) * cos(2 * varphi)
        e2 = @. (1 - q) / (1 + q) * sin(2 * varphi)
        return e1, e2
    end

    # ═══════════════════════════════════════════════════════════════
    #  7. ELLIPTICAL DISTORTION
    # ═══════════════════════════════════════════════════════════════

    """
        EllipticalDistortion(xg, yg; e1, e2, xcentre, ycentre)

    Map (x, y) to elliptically-distorted coordinates where the
    profile becomes circular. Uses lenstronomy's convention.
    """
    function EllipticalDistortion(xg, yg; e1, e2, xcentre, ycentre)
        xsh = xg .- xcentre
        ysh = yg .- ycentre
        norm = sqrt(max(abs(1 - e1^2 - e2^2), 1e-10))
        xed = @. ((1 - e1) * xsh - e2 * ysh) / norm
        yed = @. (-e2 * xsh + (1 + e1) * ysh) / norm
        return xed, yed
    end

    # ═══════════════════════════════════════════════════════════════
    #  8. NOISE MODELS
    # ═══════════════════════════════════════════════════════════════

    """
        PoissonNoise(image, exp_time)

    Poisson (shot) noise: σ = √(|image| / exp_time).
    """
    function PoissonNoise(image::AbstractArray, exp_time::Real)
        sigma = @. sqrt(abs(image) / exp_time)
        return randn!(similar(image)) .* sigma
    end

    """
        GaussianNoise(image, sigma_bkd)

    Gaussian readout noise with std σ_bkg.
    """
    function GaussianNoise(image::AbstractArray, sigma_bkd::Real)
        return randn!(similar(image)) .* sigma_bkd
    end

    """
        BackgroundNoise(img; ...)

    Placeholder — requires `sigma_clip` and `estimate_background`
    from an external package (e.g. AstroImages.jl).
    Returns (0.0, 0.0) as stub.
    """
    function BackgroundNoise(img; kwargs...)
        @warn "BackgroundNoise is a stub — add AstroImages.jl or similar for full functionality."
        return 0.0, 0.0
    end

    # ═══════════════════════════════════════════════════════════════
    #  9. UTILITY
    # ═══════════════════════════════════════════════════════════════

    """
        include_folder(path, m=@__MODULE__)

    Include all .jl files in a directory into module `m`.
    """
    function include_folder(path::AbstractString, m::Module=@__MODULE__)
        paths = joinpath.(Ref(path), readdir(path))
        jl_files = sort(filter(f -> endswith(f, ".jl"), paths))
        Base.include.(Ref(m), jl_files)
        return jl_files
    end

    """
        name2str(arg)

    Convert an identifier to its string representation at compile time.
    """
    macro name2str(arg)
        return string(arg)
    end

end # module LensUtils
