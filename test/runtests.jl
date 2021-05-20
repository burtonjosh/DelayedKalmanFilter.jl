using DelayedKalmanFilter
# using Test

protein_at_observations = [0 100; 10 200; 20 500; 30 700; 40 400]
model_parameters = [10000.0,5.0,log(2)/30,log(2)/90,1.0,1.0,29.0]
measurement_variance = 10

kalman_filter(protein_at_observations,model_parameters,measurement_variance)

@testset "DelayedKalmanFilter.jl" begin
    # tests here
end
