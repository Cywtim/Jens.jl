
module LensFITS

    using FITSIO, CFITSIO, AstroLib, Distributions
    using Statistics, Random

    function FitsRead(Path, HDUs=falses)

        if split(Path, ".")[end] != "fits"
            Path = Path * ".fits"
        end

        try
            if HDUs
                hdus = FITS(Path);
            else
                hdus = FITS(Path)
            end
        catch err
            println("There is no such file " * Path)
            rethrow(e)

        end
    end

    function FitsWrite(NewFile, data)

        FITS(NewFile, "w") do hdus
            write(hdus, data)
        end

    end


end