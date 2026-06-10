
module LensFITS

    using FITSIO, CFITSIO, AstroLib

    function FitsRead(Path, HDUs=false)

        if split(Path, ".")[end] != "fits"
            Path = Path * ".fits"
        end

        try
            if HDUs
                hdus = FITS(Path)
            else
                hdus = FITS(Path)
            end
        catch e
            println("There is no such file " * Path)
            rethrow(e)
        end
        return hdus
    end

    function FitsWrite(NewFile, data)

        FITS(NewFile, "w") do hdus
            write(hdus, data)
        end

    end


end