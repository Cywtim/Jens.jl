module LensUtils

    using LinearAlgebra, LazyGrids

    export  LensGrid, LensPolGrid, LensInverse, LensRotation
    export Pol2Car, Car2Pol
    export e2phiq, phiq2e
    export PoissonNoise, GaussianNoise, BackgroundNoise
    export include_folder
    export  name2str
    
    function MatrixInverse(M::AbstractArray, e::Float64=1e-10)
    #=
    Inverse function: Calculate the inverse of a matrix or a number

        M:: Matrix or Number
        e:: err value, default 1e-10
    =#
        try
            return inv(M)
        catch error
            return inv(M + e * I)
        end
    end

    function LensGrid(;xl::Union{Real, Array{Real}}, 
                nx::Union{Int, Array{Int}}=100,
            yl::Union{Real, Array{Real}, Nothing}=nothing,
            ny::Union{Int, Array{Int}, Nothing}=nothing, 
            CarOut::Bool=true)
    #=
    Grid function in the region of (xl[1], xl[2]) and (yl[1], yl[2]), with interval of nx-1 and ny-1.

        xl:: Int or 1D-Array
        yl:: Int or 1D-Array
        nx:: Int
        ny:: Int
    =#
        if yl === nothing
            yl = xl
        end

        if ny === nothing
            ny = nx
        end

        if length(xl) == 1
            xg = range(-xl, xl, nx)
            yg = range(-yl, yl, ny)

        elseif  length(xl) == 2
            xg = range(xl[1], xl[2], nx)
            yg = range(yl[1], yl[2], ny)
        else
            println("Wrong size of x,y ranges.")
            return
        end
        if CarOut
            return ndgrid(xg, yg)
        else
            return Car2Pol(xg, yg)
        end
        
    end

    function LensPolGrid(;rl::Union{Float64, Array{Float64}}, thetal::Union{Float64, Array{Float64}, Nothing}=2*pi,
        nr::Union{Int, Array{Int}}=100, ntheta::Union{Int, Array{Int}}=100, Polout::Bool=true)
        #=
        Grid function in the region of (xl[1], xl[2]) and (yl[1], yl[2]), with interval of nx-1 and ny-1.
    
            xl:: Int or 1D-Array
            yl:: Int or 1D-Array
            nx:: Int
            ny:: Int
        =#
    
        if length(rl) == 1
            rg = range(0, xl, nx)

        elseif  length(rl) == 2
            rg = range(rl[1], rl[2], nr)
        else
            println("Wrong size of r ranges.")
            return
        end

        if length(thetal) == 1
            rg = range(0, thetal, ntheta)

        elseif  length(rl) == 2
            thetag = range(thetal[1], thetal[2], ntheta)
        else
            println("Wrong size of theta ranges.")
            return
        end
        
        if Polout
            return ndgrid(rg, thteag)
        else
            return Pol2Car(rg, thetag)
        end
    end


    function LensRotation(xg, yg, varphi)
        print(typeof(xg),size(xg))
        print(typeof(varphi),size(varphi))

        xr = @. cos(varphi) .* xg .- sin(varphi) .* yg

        yr = @. sin(varphi) .* xg .+ cos(varphi) .* yg
        
        return xr, yr
        
    end


    function Mesh2Array()
        
    end

    function Array2Mesh()

    end

    function Car2Pol(x, y; xc::Real=0., yc::Real=0.)

        xsh = x .- xc
        ysh = y .- yc

        r = @. sqrt.(xsh.^2 .+ ysh.^2)
        phi = @. atan.(ysh ./ xsh)

        return r, phi

    end

    function Pol2Car(r, phi; xc::Real=0., yc::Real=0.)

        x = @. r .* cos.(phi)
        y = @. r .* sin.(phi)

        return x .- xc, y .- yc

    end

    function ShearPol2Car(phi, gamma)

        gamma1 = @. gamma * cos(2 * phi)
        gamma2 = @. gamma * sin(2 * phi)
        return gamma1, gamma2

    end

    function ShearCar2Pol(gamma1, gamma2)

        phi = @. atan(gamma2, gamma1) / 2
        gamma = @. sqrt(gamma1^2 + gamma2^2)
        return phi, gamma

    end

    function e2phiq(e1::Real, e2::Real)
        #=

        Transformation from ellipticities to orientation angle and axis ratio.

        e1:: eccentricity in x-direction
        e2:: eccentricity in y-direction
        return::  axis ratio (minor/major), angle in radian

        =#
        varphi = @. atan.(e2, e1) ./ 2.0
        e = @. sqrt.(e1.^2 .+ e2.^2)
        e = @. min.(e, 0.9999)
        q = @. (1.0 .- e) ./ (1.0 .+ e)
        return q, varphi

    end

    function phiq2e(q::Real, varphi::Real)
        #=

        Transformation from orientation angle and axis ratio to ellipticities.

        q:: axis ratio (minor/major)
        varphi:: angle in radian
        return::  eccentricity in x-direction and y-direction

        =#
        e1 = @. (1.0 .- q) ./ (1.0 .+ q) .* cos.(2.0 .* varphi)
        e2 = @. (1.0 .- q) ./ (1.0 .+ q) .* sin.(2.0 .* varphi)

        return e1, e2

    end

    function EllipticalDistortion(xg::AbstractArray, yg::AbstractArray;
                     e1::Real, e2::Real, xcentre::Real, ycentre::Real)

        xsh = xg .- xcentre
        ysh = yg .- ycentre
    
        norm = @. sqrt(max(abs(1 - e1^2 - e2^2), 1e-10))
        xed = @. ((1 - e1) * xsh - e2 * ysh) / norm
        yed = @. (-e2 * xsh + (1 + e1) * ysh) / norm
        return xed, yed
    
    end

    function PoissonNoise(image::AbstractArray,
        exp_time::Real)

       sigma =@.  sqrt(abs(image) / exp_time)
       
       poisson = rand(Normal(0., 1.), size(image)) .* sigma
   
       return poisson
   
   end

   function GaussianNoise(image::AbstractArray,
                       sigma_bkd::Real)

       Gauss = rand(Normal(0., 1.), size(image))  .* sigma_bkd
   
       return Gauss
   
   end

   function BackgroundNoise(img::AbstractArray;
       clipara::Dict=Dict(:fill=>NaN, :center=>median(img), :std=>std(img, corrected=false)),
       sigma::Real=1, boxsize::Real=50, filtersize::Real=5)

       clipped = sigma_clip(img, sigma; clipara...)

       bkg, bkg_rms = estimate_background(clipped, boxsize, filter_size=filtersize)

       return bkg, bkg_rms
       
   end

   function include_folder(path::AbstractString, m::Module=@__MODULE__)

        paths = joinpath.(Ref(path), readdir(path))
        julia_files = filter(f -> endswith(f, ".jl"), paths)
        Base.include.(Ref(m), julia_files)

        return julia_files

    end

    macro name2str(arg)
        x = string(arg)
        quote
            $x
        end
    end


end 