using DelayedKalmanFilter
using Test

@testset "DelayedKalmanFilter.jl" begin
    protein_at_observations = [0 100; 10 200; 20 500; 30 700; 40 400];
    model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];
    measurement_variance = 10;
    kalman_filter(protein_at_observations,model_parameters,measurement_variance);
end

protein_at_observations = [0 100; 10 200; 20 500; 30 700; 40 400];
model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0];
measurement_variance = 10
a,b,_,_,c,_,_ = kalman_filter(protein_at_observations,model_parameters,measurement_variance);
calculate_log_likelihood_at_parameter_point(model_parameters,[protein_at_observations],measurement_variance)
