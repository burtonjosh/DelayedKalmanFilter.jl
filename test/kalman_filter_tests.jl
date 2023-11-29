using DelayedKalmanFilter
using DelimitedFiles
using Test

@testset "Kalman Filter tests" begin
  loading_path = string(dirname(@__FILE__), "/data/")
  protein_at_observations = readdlm(string(loading_path, "kalman_filter_test_trace_observations.csv"), ',')
  model_parameters = [10000.0, 5.0, log(2) / 30, log(2) / 90, 1.0, 1.0, 29.0]
  measurement_variance = 10000.0
  system_state, distributions = kalman_filter(protein_at_observations, model_parameters, measurement_variance)

  pd = readdlm(string(loading_path, "python_distributions.txt"))
  length_of_mean = first(size(protein_at_observations))
  last_time, last_protein = protein_at_observations[end, :]

  # check arrays are correct shape
  @test length(system_state.means) == length_of_mean
  @test length(system_state.variances) == length_of_mean
  @test size(distributions, 1) == size(protein_at_observations, 1)

  # check system_state finished on final observation
  @test system_state.current_time == last_time
  @test system_state.current_observation == last_protein

  # check output is same as python code
  @test distributions ≈ pd
end
