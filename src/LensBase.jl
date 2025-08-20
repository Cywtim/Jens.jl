module LensBase

    using AstroLib, NLsolve, Optim, LazyGrids

    include("LensUtils.jl")

    export LensCheck, LensFermat, LensDeflection
    export LensRayShootingPosition, LensRayShooting
    export LensRayShootingPosition, MultiLensRayShooting
    export LensCriticalCurve, LensCaustic

    function LensCheck(LensModel; LensKwargs)
        
        LensModel.LensCheck(; LensKwargs...)

    end


    function LensFermat(xg::Union{LazyGrids.GridSL, Matrix{Float64}},
         yg::Union{LazyGrids.GridSL, Matrix{Float64}},
          beta=[0., 0.]; LensModel::Module, LensKwargs::Dict)
        
        phi = LensModel(xg, yg, LensKwargs)

        Fermat = @. ((beta[1] - xg)^2 + (beta[2] - yg)^2) / 2 - phi

        return Fermat

    end

    function LensDeflection(xg::AbstractArray, yg::AbstractArray;
         LensModel::Module, LensKwargs::Dict )
        #=?=#
        alpha_x, alpha_y = LensModel.LensDerivative(xg, yg; LensKwargs...)
        
        return alpha_x, alpha_y

    end

    function LensPlane(xg::Union{Array, LazyGrids.GridSL, Matrix{Float64}},
      yg::Union{Array, LazyGrids.GridSL, Matrix{Float64}};
       LensModel::Module, LensKwargs::Dict )
      #=?=#
      alpha_x, alpha_y = LensModel.LensDerivative(xg, yg; LensKwargs...)
      
      beta_x = @. xg - alpha_x
      beta_y = @. yg - alpha_y

      return beta_x, beta_y

    end

    function LensMagnificationR(thetax::Union{LazyGrids.GridSL, Array{Any, 2}},
      thetay::Union{LazyGrids.GridSL, Array{Any, 2}};
       LensModel::Module, LensKwargs::Dict)

          h_xx, h_xy, h_yy = LensModel.LensHessian(thetax, thetay; LensKwargs...)
          
          magr = @. (1. .- h_xx) .* (1. .- h_yy) .- h_xy.^2.;

          return magr

    end

    function LensRayShootingPosition(
      thetax::Union{LazyGrids.GridSL, Matrix{Float64}},
         thetay::Union{LazyGrids.GridSL, Matrix{Float64}};
          LensModel::Module, LensKwargs::Dict)

        alphax, alphay = LensModel.LensDerivative(thetax, thetay; LensKwargs...)
        betax = thetax .- alphax
        betay = thetay .- alphay

        return betax, betay
    end

    function MultiLensRayShootingPosition(
      thetax::Union{LazyGrids.GridSL, Matrix{Float64}},
        thetay::Union{LazyGrids.GridSL, Matrix{Float64}};
          LensModel::Union{Module, Vector{Module}},
            LensKwargs::Vector{Dict{Symbol, Float64}})

        xl = Union{LazyGrids.GridSL, Matrix{Float64}}[thetax]
        yl = Union{LazyGrids.GridSL, Matrix{Float64}}[thetay]
          
        len = length(LensKwargs)

        if typeof(LensModel) == Module

          for i in 1:len
            ax, ay = LensModel.LensDerivative(xl[i], yl[i]; LensKwargs[i]...)
            push!(xl, xl[i] .- ax)
            push!(yl, yl[i] .- ay)
          end
    
          # return Vector{Union{GridSL, Matrix{Float64}}}
          return xl, yl

        elseif typeof(LensModel) == Vector{Module}

          for i in 1:len
            ax, ay = LensModel[i].LensDerivative(xl[i], yl[i]; LensKwargs[i]...)
            push!(xl, xl[i] .- ax)
            push!(yl, yl[i] .- ay)
          end

          # return Vector{Union{GridSL, Matrix{Float64}}}
          return xl, yl
        
        else

          println("The type of LensModel is not Module or Moduel Vector.")

        end

     end


    function LensRayShooting(thetax::Union{LazyGrids.GridSL, Matrix{Float64}},
         thetay::Union{LazyGrids.GridSL, Matrix{Float64}};
          LensModel::Module, LensKwargs::Dict,
             SourceProfile::Function, SourceKwargs::Dict)

        alphax, alphay = LensModel.LensDerivative(thetax, thetay; LensKwargs...)
        betax = thetax .- alphax
        betay = thetay .- alphay
        
        #print(typeof(betax))
        
        #h_xx, h_xy, h_yy = LensModel.LensHessian(betax, betay; LensKwargs...)
        #magr = @. abs.((1. - h_xx)*(1. - h_yy) - h_xy^2.);

        light = SourceProfile(betax, betay; SourceKwargs...)

        return light

    end

    function MultiLensRayShooting(
      thetax::Union{LazyGrids.GridSL, Matrix{Float64}},
        thetay::Union{LazyGrids.GridSL, Matrix{Float64}};
          LensModel::Union{Module, Vector{Module}},
            LensKwargs::Vector{Dict{Symbol, Float64}},
              SourceProfile::Function, SourceKwargs::Dict)

        xl = Union{LazyGrids.GridSL, Matrix{Float64}}[thetax]
        yl = Union{LazyGrids.GridSL, Matrix{Float64}}[thetay]
          
        len = length(LensKwargs)

        if typeof(LensModel) == Module

          for i in 1:len
            ax, ay = LensModel.LensDerivative(xl[i], yl[i]; LensKwargs[i]...)
            push!(xl, xl[i] .- ax)
            push!(yl, yl[i] .- ay)
          end
    
          # return Vector{Union{GridSL, Matrix{Float64}}}
          return xl, yl

        elseif typeof(LensModel) == Vector{Module}

          for i in 1:len
            ax, ay = LensModel[i].LensDerivative(xl[i], yl[i]; LensKwargs[i]...)
            push!(xl, xl[i] .- ax)
            push!(yl, yl[i] .- ay)
          end

          # return Vector{Union{GridSL, Matrix{Float64}}}
          return xl, yl
        
        else

          println("The type of LensModel is not Module or Moduel Vector.")

        end

     end


    function LensCriticalCurve(; LensModel::Module, LensKwargs::Dict, hperr::Float64=0.01, 
      r_max::Float64=2., r_bins::Int=4000, theta_bins::Int=4000)

        r = range(0, r_max, r_bins)
        theta = range(0, 2 * pi, theta_bins)
        (rg, thetag) = ndgrid(r, theta);
        (xpg, ypg) = LensUtils.Pol2Car(rg, thetag, xc=-LensKwargs[:xcentre], yc=-LensKwargs[:ycentre])
      
        hp_xx, hp_xy, hp_yy = LensModel.LensHessian(xpg, ypg; LensKwargs...)
        hp = @. (1-hp_xx)*(1-hp_yy) - hp_xy^2

        ccindex = findall( 0.0 .< hp .< hperr)

        ccx = [xpg[i] for i in ccindex]
        ccy = [ypg[i] for i in ccindex]

        return ccx, ccy
    end

    function LensCaustic(; LensModel::Module, LensKwargs::Dict, hperr::Float64=0.01, 
            r_max::Float64=2., r_bins::Int=1000, theta_bins::Int=4000)

        r = range(0, r_max, r_bins)
        theta = range(0, 2 * pi, theta_bins)
        (rg, thetag) = ndgrid(r, theta);
        (xpg, ypg) = LensUtils.Pol2Car(rg, thetag, xc=-LensKwargs[:xcentre], yc=-LensKwargs[:ycentre])

        fp_x, fp_y = LensModel.LensDerivative(xpg, ypg; LensKwargs...) # deflection angles f_i
        xpb, ypb = xpg .- fp_x, ypg .- fp_y  # lensed image plane
      
        hp_xx, hp_xy, hp_yy = LensModel.LensHessian(xpg, ypg; LensKwargs...)
        hp = @. (1-hp_xx)*(1-hp_yy) - hp_xy^2

        ccindex = findall( 0.0 .< hp .< hperr)

        csx = [xpb[i] for i in ccindex]
        csy = [ypb[i] for i in ccindex]

        return csx, csy

    end

end