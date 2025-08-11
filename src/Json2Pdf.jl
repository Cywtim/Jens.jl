module Json2Pdf

    using JSON, PDFIO, CairoMakie

    export JsonLoad, JsonToPdf

    function JsonLoad(file_path::String)

        json_data = JSON.parsefile(file_path)

        return json_data
    end

    function JsonToPdf(json_data::Dict{String, Any}, pdf_filename::String="JSON2PDF.pdf")
        pdDoc = pdDocOpen(pdf_filename)
        pdPage = pdPageCreate(pdDoc)
        
        # Start writing content to the PDF
        pdPageBeginText(pdPage)
        pdSetFont(pdPage, "Helvetica", 12)
        
        # Simple recursive function to print JSON data
        function print_json(data, indent=0)
            for (key, value) in data
                line = " " ^ indent * "$key: $value\n"
                pdShowText(pdPage, line)
                if typeof(value) == Dict{String,Any}
                    print_json(value, indent + 2)
                end
            end
        end
        
        print_json(json_data)
        
        pdPageEndText(pdPage)
        pdPageAdd(pdDoc, pdPage)
        pdDocClose(pdDoc)
    end

end 


