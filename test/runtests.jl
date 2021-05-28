using SafeTestsets

@safetestset "Kalman Filter tests" begin include("kalman_filter_tests.jl") end
@safetestset "Likelihood tests" begin include("likelihood_tests.jl") end
