module LensPyPlot

    using PyPlot

    export PlotPlane
    
    function PlotPlane(xg::AbstractArray, yg::AbstractArray,
         image::AbstractArray; figsize::Vector=[6, 6])

        LensFig, LensAx = PyPlot.subplots(figsize=figsize)
        LensAx.imshow(image)

        xmax = maximum(xg)
        xmin = minimum(xg)
        xticks = LensAx.get_xticks()[2:end-1]
        xmun = size(LensAx.get_xticks())[1]-2
        xlabels = round.(range(start=xmin, stop=xmax, length=xmun), digits=2)

        ymax = maximum(yg)
        ymin = minimum(yg)
        yticks = LensAx.get_yticks()[2:end-1]
        ymun = size(LensAx.get_yticks())[1]-2
        ylabels = round.(range(start=ymin, stop=ymax, length=ymun), digits=2)

        LensAx.set_xticks(xticks, labels=xlabels)
        LensAx.set_yticks(yticks, labels=ylabels)

        return LensFig, LensAx
    end


end