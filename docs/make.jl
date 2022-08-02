using DelayedKalmanFilter
using Documenter
using DifferentialEquations, StochasticDelayDiffEq, Plots, Statistics

DocMeta.setdocmeta!(DelayedKalmanFilter, :DocTestSetup, :(using DelayedKalmanFilter); recursive=true)

makedocs(;
    modules = [DelayedKalmanFilter],
    authors = "Joshua Burton",
    repo = "https://github.com/burtonjosh/DelayedKalmanFilter.jl/blob/{commit}{path}#L{line}",
    sitename = "DelayedKalmanFilter.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        canonical = "https://burtonjosh.github.io/DelayedKalmanFilter.jl",
        assets = String[],
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => ["Tutorial" => "examples/tutorial.md"],
        "Library" => ["Public" => "lib/public.md", "Internals" => "lib/internals.md"]
        ],
)

deploydocs(; repo = "github.com/burtonjosh/DelayedKalmanFilter.jl", devbranch="diffeq_new")