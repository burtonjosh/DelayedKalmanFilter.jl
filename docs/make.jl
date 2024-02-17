using DelayedKalmanFilter
using Documenter

DocMeta.setdocmeta!(DelayedKalmanFilter, :DocTestSetup, :(using DelayedKalmanFilter); recursive = true)

makedocs(;
  modules = [DelayedKalmanFilter],
  authors = "Joshua Burton",
  repo = Remotes.GitHub("burtonjosh", "DelayedKalmanFilter.jl"),
  sitename = "DelayedKalmanFilter.jl",
  format = Documenter.HTML(;
    prettyurls = get(ENV, "CI", "false") == "true",
    canonical = "https://burtonjosh.github.io/DelayedKalmanFilter.jl",
    assets = ["assets/favicon.ico"],
  ),
  pages = [
    "Home" => "index.md",
    "Examples" => ["Tutorial" => "examples/tutorial.md", "Parameter Estimation" => "examples/parameter-estimation.md"],
    "Library" => ["Public" => "lib/public.md", "Internals" => "lib/internals.md"],
  ],
)

deploydocs(repo = "github.com/burtonjosh/DelayedKalmanFilter.jl")
