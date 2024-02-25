using DelayedKalmanFilter: SystemState, update_mean!, update_variance!, update_step!, calculate_helper_inverse
using LinearAlgebra
using Test

initial_mean = [1.0; 2.0]
initial_variance = [1.0 0.2; 0.2 1.0]
observation = 10.0
observation_transform = [0.0 1.0]
measurement_variance = 4

@testset "Test single update step, mean and variance separately" begin
  system_state = SystemState(initial_mean, initial_variance, 0.0, 10.0)

  helper_inverse = calculate_helper_inverse(initial_variance, observation_transform, measurement_variance)
  @test helper_inverse == 1 / 5

  @testset "Test mean update" begin
    update_mean!(system_state, observation, observation_transform, helper_inverse)

    adaptation_coefficient = initial_variance * observation_transform' * helper_inverse
    expected = initial_mean + adaptation_coefficient * (observation - dot(observation_transform, initial_mean))
    actual = system_state.mean
    @test expected == actual
  end

  @testset "Test variance update" begin
    update_variance!(system_state, observation_transform, helper_inverse)

    expected =
      initial_variance - initial_variance * observation_transform' * observation_transform * initial_variance * helper_inverse
    actual = system_state.variance
    @test actual == expected
  end
end

@testset "Test single update step, mean and variance together" begin
  system_state = SystemState(initial_mean, initial_variance, 0.0, 10.0)
  update_step!(system_state, observation, measurement_variance, observation_transform)
  initial_variance

  helper_inverse = calculate_helper_inverse(initial_variance, observation_transform, measurement_variance)
  @test helper_inverse == 1.0 / (dot(observation_transform, initial_variance * observation_transform') + measurement_variance)

  adaptation_coefficient = initial_variance * observation_transform' * helper_inverse
  expected_mean = initial_mean + adaptation_coefficient * (observation - dot(observation_transform, initial_mean))
  expected_variance =
    initial_variance - initial_variance * observation_transform' * observation_transform * initial_variance * helper_inverse

  actual_mean = system_state.mean
  actual_variance = system_state.variance
  @test actual_mean == expected_mean
  @test actual_variance == expected_variance
end
