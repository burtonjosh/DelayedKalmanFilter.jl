using DelayedKalmanFilter
using Test

@testset "DelayedKalmanFilter.jl" begin
    protein_at_observations = [0 100; 10 200; 20 500; 30 700; 40 400];
    model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];
    measurement_variance = 10;
    a,b,c,d,e,f,g = kalman_filter(protein_at_observations,model_parameters,measurement_variance);

    @test size(a) == (protein_at_observations[end,1] + 1 + model_parameters[end],3)
    @test size(b) == ((protein_at_observations[end,1] + 1 + model_parameters[end])*2,(protein_at_observations[end,1] + 1 + model_parameters[end])*2)
    @test size(c) == (protein_at_observations[end,1] + 1 + model_parameters[end],length(model_parameters),2)
    @test size(d) == (length(model_parameters),(protein_at_observations[end,1] + 1 + model_parameters[end])*2,(protein_at_observations[end,1] + 1 + model_parameters[end])*2)
    @test size(e) == (size(protein_at_observations,1),3)
    @test size(f) == (size(protein_at_observations,1),length(model_parameters),2)
    @test size(g) == (size(protein_at_observations,1),length(model_parameters),2,2)
end

protein_at_observations = [0 100; 10 200; 20 500; 30 700; 40 400];
model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];
measurement_variance = 10;
a,b,c,d,e,f,g = kalman_filter(protein_at_observations,model_parameters,measurement_variance);
calculate_log_likelihood_at_parameter_point(model_parameters,protein_at_observations,measurement_variance);
ll, lld = calculate_log_likelihood_and_derivative_at_parameter_point(protein_at_observations,model_parameters,150.,measurement_variance);

# using ForwardDiff
# log_likelihood_gradient(model_parameters,protein_at_observations,measurement_variance) = ForwardDiff.gradient(model_parameters -> calculate_log_likelihood_at_parameter_point(model_parameters,protein_at_observations,measurement_variance),model_parameters)
# log_likelihood_gradient(model_parameters,protein_at_observations,measurement_variance)
