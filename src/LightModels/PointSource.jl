module PointSource

    mutable struct  

    
    ErrorMap::AbstractArray
    PsfMap::AbstractArray
    ImagePairs::AbstractArray # if the pairs of images determined
    
    end
    
    function PS(amp::Real=1., xcentre::Real=0., ycentre::Real=0.,)


        b = max(1.999  - 0.327, bmin)

        R = @. sqrt((x - xcentre)^2 + (y - ycentre)^2)

        I = @. amp * exp( - b * ( R / Rsersic ))

        return I

    end

    function PI(x::AbstractArray, y::AbstractArray;
        amp::Real=1., xcentre::Real=0., ycentre::Real=0., bmin::Real=1e-4)
        
        A = @. (x-xcentre) * cos(varphi) + (y-ycentre) * sin(varphi)
        B = @. -(x-xcentre) * sin(varphi) + (y-ycentre) * cos(varphi)
        R = @. sqrt(A^2 + (B / ( 1 - q ))^2)

        I = @. amp * exp( -b * ((R/Rsersic) - 1))

        return I 
    end


    function PSolver(LensModel, image_position)
    
    end
end
