using Jens
using Documenter

DocMeta.setdocmeta!(Jens, :DocTestSetup, :(using Jens); recursive=true)

makedocs(;
    modules=[Jens],
    authors="Cywtim",
    sitename="Jens.jl",
    format=Documenter.HTML(;
        canonical="https://Cywtim.github.io/Jens.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/Cywtim/Jens.jl",
    devbranch="main",
)
