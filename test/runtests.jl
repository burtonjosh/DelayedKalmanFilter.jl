using DelayedKalmanFilter
using DelimitedFiles
using Test

@testset "DelayedKalmanFilter.jl" begin
    loading_path = string(pwd(),"/data/");
    protein_at_observations = readdlm(string(loading_path,"kalman_filter_test_trace_observations.csv"),',');
    model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];
    measurement_variance = 10000;
    a,b,c,d,e,f,g = kalman_filter(protein_at_observations,model_parameters,measurement_variance);

    # check arrays are correct shape
    @test size(a) == (protein_at_observations[end,1] + 1 + model_parameters[end],3)
    @test size(b) == ((protein_at_observations[end,1] + 1 + model_parameters[end])*2,(protein_at_observations[end,1] + 1 + model_parameters[end])*2)
    @test size(c) == (protein_at_observations[end,1] + 1 + model_parameters[end],length(model_parameters),2)
    @test size(d) == (length(model_parameters),(protein_at_observations[end,1] + 1 + model_parameters[end])*2,(protein_at_observations[end,1] + 1 + model_parameters[end])*2)
    @test size(e) == (size(protein_at_observations,1),3)
    @test size(f) == (size(protein_at_observations,1),length(model_parameters),2)
    @test size(g) == (size(protein_at_observations,1),length(model_parameters),2,2)

    # test log likelihood and derivative are correct
    ll = calculate_log_likelihood_at_parameter_point(protein_at_observations,model_parameters,measurement_variance);
    @test ll ≈ readdlm(string(loading_path,"log_likelihood_value.csv"),',')[1]

    ll, lld = calculate_log_likelihood_and_derivative_at_parameter_point(protein_at_observations,model_parameters,10000.,measurement_variance);
    @test ll ≈ readdlm(string(loading_path,"log_likelihood_value.csv"),',')[1]
    @test lld ≈ reshape(readdlm(string(loading_path,"log_likelihood_derivative_value.csv"),','),7)
end

# using ForwardDiff
# log_likelihood_gradient(model_parameters,protein_at_observations,measurement_variance) = ForwardDiff.gradient(model_parameters -> calculate_log_likelihood_at_parameter_point(model_parameters,protein_at_observations,measurement_variance),model_parameters)
# log_likelihood_gradient(model_parameters,protein_at_observations,measurement_variance)
