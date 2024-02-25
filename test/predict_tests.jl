using DelayedKalmanFilter:
  prediction_step!, predict_mean!, predict_variance!, state_space_mean_RHS, calculate_noise_variance, hill_function
using OrdinaryDiffEq

initial_mean = [1.0; 2.0]
initial_variance = [1.0 0.2; 0.2 1.0]
observation = 10.0
observation_transform = [0.0 1.0]
measurement_variance = 4

model_params = [300.0, 4.0, 0.1, 0.1, 1.0, 1.0]
instant_jac = [-model_params[3] 0.0; model_params[6] -model_params[4]]
ode_solver = Tsit5()

@testset "Test single prediction step, mean and variance separately" begin
  system_state = SystemState(initial_mean, initial_variance, 0.0, 10.0)
  # construct solution directly
  tspan = (0.0, 10.0)

  mean_prob = ODEProblem(state_space_mean_RHS, initial_mean, tspan, model_params)
  mean_solution = solve(mean_prob, ode_solver; save_everystep = false)
  expected = mean_solution.u[end]

  predict_mean!(system_state, model_params; ode_solver)
  actual = system_state.mean
  @test expected == actual

  function variance_RHS(dvariance, current_variance, params, t)
    variance_of_noise = calculate_noise_variance(params, mean_solution, t)
    dvariance .= instant_jac * current_variance + current_variance * instant_jac' + variance_of_noise
  end

  var_prob = ODEProblem(variance_RHS, initial_variance, tspan, model_params)
  var_solution = solve(var_prob, ode_solver; save_everystep = false)
  expected = var_solution.u[end]

  predict_variance!(system_state, mean_solution, model_params, instant_jac; ode_solver)

  actual = system_state.variance
  @test actual == expected
end
