using JuliaML
using Documenter

DocMeta.setdocmeta!(JuliaML, :DocTestSetup, :(using JuliaML); recursive=true)

makedocs(;
    modules=[JuliaML],
    authors="Norel <norel.evoagile@gmail.com> and contributors",
    repo="https://github.com/erdosxx/JuliaML.jl/blob/{commit}{path}#{line}",
    sitename="JuliaML.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://erdosxx.github.io/JuliaML.jl",
        edit_link="master",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/erdosxx/JuliaML.jl",
    devbranch="master",
)
