using DelayedKalmanFilter
using DelimitedFiles
using Test

@testset "Likelihood tests" begin
    loading_path = string(pwd(), "/data/")
    protein_at_observations =
        readdlm(string(loading_path, "kalman_filter_test_trace_observations.csv"), ',')
    model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0] #ModelParameters()
    measurement_variance = 10000.0

    # test negative parameter check works
    negative_parameter_1 =
        [10000.0,5.0,log(2)/30,log(2)/90,-1.0,1.0,29.0]#ModelParameters(10000.0, 5.0, log(2) / 30, log(2) / 90, -1.0, 1.0, 29.0)
    negative_parameter_2 =
        [-10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0]#ModelParameters(-10000.0, 5.0, log(2) / 30, log(2) / 90, 1.0, 1.0, 29.0)

    ll = calculate_log_likelihood_at_parameter_point(
        protein_at_observations,
        negative_parameter_1,
        measurement_variance,
    )
    @test ll == -Inf
    ll = calculate_log_likelihood_at_parameter_point(
        protein_at_observations,
        negative_parameter_2,
        measurement_variance,
    )
    @test ll == -Inf
    # test log likelihood and derivative are correct
    ll = calculate_log_likelihood_at_parameter_point(
        protein_at_observations,
        model_parameters,
        measurement_variance,
    )
    @test ll ≈ readdlm(string(loading_path, "log_likelihood_value.csv"), ',')[1]

    # test time delays which are less than the obervation time step
    small_delay_1 =
        [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,2.0]

    small_delay_2 =
        [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,7.0]

    ll = calculate_log_likelihood_at_parameter_point(
        protein_at_observations,
        small_delay_1,
        measurement_variance,
    )
    @test ll ≈ readdlm(string(loading_path, "log_likelihood_value.csv"), ',')[2]

    ll = calculate_log_likelihood_at_parameter_point(
        protein_at_observations,
        small_delay_2,
        measurement_variance,
    )
    @test ll ≈ readdlm(string(loading_path, "log_likelihood_value.csv"), ',')[3]
end
