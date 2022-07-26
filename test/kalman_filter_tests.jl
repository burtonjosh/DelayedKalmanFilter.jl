using DelayedKalmanFilter
using DelimitedFiles
using Test

@testset "Kalman Filter tests" begin
    loading_path = string(pwd(), "/data/")
    protein_at_observations =
        readdlm(string(loading_path, "kalman_filter_test_trace_observations.csv"), ',')
    model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0]
    measurement_variance = 10000.0
    mean, variance, distributions =
        kalman_filter(protein_at_observations, model_parameters, measurement_variance)

    discrete_delay = ceil(Int,model_parameters[end])
    number_of_hidden_states = protein_at_observations[2,1] - protein_at_observations[1,1]
    length_of_mean = ceil(Int,number_of_hidden_states/discrete_delay) + ceil(Int,discrete_delay/number_of_hidden_states)

    # check arrays are correct shape
    @test length(mean) ==
          length_of_mean#(protein_at_observations[end, 1] + 1 + model_parameters[7], 3)
    @test size(variance) == (
        (protein_at_observations[end, 1] + 1 + model_parameters[7]) * 2,
        (protein_at_observations[end, 1] + 1 + model_parameters[7]) * 2,
    )
    @test size(distributions, 1) == size(protein_at_observations, 1)
end
