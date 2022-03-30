using DelayedKalmanFilter
using DelimitedFiles
using Test

@testset "Kalman Filter tests" begin
    loading_path = string(pwd(), "/data/")
    protein_at_observations =
        readdlm(string(loading_path, "kalman_filter_test_trace_observations.csv"), ',')
    model_parameters = ModelParameters()
    measurement_variance = 10000.0
    state_space_and_distributions =
        kalman_filter(protein_at_observations, model_parameters, measurement_variance)

    state_space = state_space_and_distributions.state_space
    state_space_mean = state_space.mean
    state_space_variance = state_space.variance

    distributions = state_space_and_distributions.distributions

    # check arrays are correct shape
    @test size(state_space_mean) ==
          (protein_at_observations[end, 1] + 1 + model_parameters.τ, 3)
    @test size(state_space_variance) == (
        (protein_at_observations[end, 1] + 1 + model_parameters.τ) * 2,
        (protein_at_observations[end, 1] + 1 + model_parameters.τ) * 2,
    )
    @test size(distributions, 1) == size(protein_at_observations, 1)
end
