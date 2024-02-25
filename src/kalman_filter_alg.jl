"The state space `mean` and `variance` of the system at the `current_time`"
mutable struct SystemState
  mean
  variance
  current_time
  next_time
end

# TODO add Model constructor
# struct Model{T}
#   params::Vector{T} # model parameters
#   jac_f::Matrix{T} # instant jacobian
#   Σ::T # measurement variance
#   F::Union{T, Matrix{T}} # observation transform
# end

# function Model(params, jac_f, Σ)
#   dim = last(size(jac_f))
#   F = dim == 1 ? 1.0 : fill(1.0, (last(size(jac_f)), 1))
#   Model(params, jac_f, Σ, F)
# end

"Helper function for saving the current mean and variance prediction"
function distribution_prediction(system_state::SystemState, observation_transform, measurement_variance)
  mean_prediction = dot(system_state.mean, observation_transform)
  variance_prediction = dot(observation_transform, system_state.variance * observation_transform') + measurement_variance
  return mean_prediction, variance_prediction
end

"""
A continuous-discrete extended Kalman-Bucy filter for a general stochastic process, based on noisy, potentially only partially observed,
observation data.

# Arguments

- `data`:

- `model_params`:

- `measurement_variance`:

# Returns

- `predicted_observation_distributions:

# Example
```jldoctest
julia> using DelayedKalmanFilter

julia> data = [0. 105.; 10. 100.; 20. 98.];

julia> params = [300.0, 4.0, 0.1, 0.1, 1.0, 1.0];

julia> measurement_variance = 1000.0;

julia> system_state, distributions = kalman_filter(
         data,
         params,
         measurement_variance
       );

julia> distributions[1, :]
2-element Vector{Float64}:
    98.83526916576103
 10883.526916576104
```
"""
function kalman_filter(data, model_params, measurement_variance; ode_solver = Tsit5())
  # F in the paper
  observation_transform = [0.0 1.0]
  # Jacobian of the system
  instant_jac = [
    -model_params[3] 0.0
    model_params[6] -model_params[4]
  ]

  # initialise state space and distribution predictions
  system_state, predicted_observation_distributions =
    state_space_initialisation(data, model_params, observation_transform, measurement_variance)

  # loop through observations and at each observation apply the Kalman prediction step and then the update step
  for (observation_index, (time, observation)) in enumerate(eachrow(data)[2:end])
    # update next_time
    system_state.next_time = time
    system_state = prediction_step!(system_state, model_params, instant_jac; ode_solver)

    # Record the predicted mean and variance for our likelihood
    predicted_observation_distributions[observation_index + 1, :] .=
      distribution_prediction(system_state, observation_transform, measurement_variance)

    system_state = update_step!(system_state, observation, measurement_variance, observation_transform)
  end
  return system_state, predicted_observation_distributions
end

"""
A function for initialisation of the state space mean and variance, and update for the "negative" times that
are a result of the time delay. Initialises the negative times using the steady state of the deterministic system,
and then updates them with `update_step!`.

# Arguments

- `data`:

- `params`:

- `observation_transform`:

- `measurement_variance`:

# Returns

- `system_state::SystemState`:

- `predicted_observation_distributions`:

"""
function state_space_initialisation(data, params, observation_transform, measurement_variance)
  steady_state = calculate_steady_state_of_ode(params)

  # construct system state space
  mean = steady_state
  variance = diagm(mean .* [20.0, 100.0])
  system_state = SystemState(mean, variance, data[1, 1], data[2, 1])

  # initialise distributions
  predicted_observation_distributions = zeros(eltype(mean), first(size(data)), 2)
  predicted_observation_distributions[1, :] .= distribution_prediction(system_state, observation_transform, measurement_variance)

  # inital update step
  update_step!(system_state, data[1, 2], measurement_variance, observation_transform)
  return system_state, predicted_observation_distributions
end

function state_space_mean_RHS(du, u, p, t)
  du[1] = -p[3] * u[1] + p[5] * hill_function(u[2], p[1], p[2])
  du[2] = p[6] * u[1] - p[4] * u[2]
  nothing
end

"""
Predict state space mean to the next observation time index
"""
function predict_mean!(system_state, model_params; ode_solver)
  tspan = (system_state.current_time, system_state.next_time)

  mean_prob = ODEProblem(state_space_mean_RHS, system_state.mean, tspan, model_params)
  mean_solution = solve(mean_prob, ode_solver; save_everystep = false)
  system_state.mean .= last(mean_solution.u)
  system_state, mean_solution
end

function calculate_noise_variance(params, mean_solution, t)
  [
    (params[3] * first(mean_solution(t)))+(params[5] * hill_function(first(mean_solution(t)), params[1], params[2])) 0.0
    0.0 (params[6] * first(mean_solution(t)))+(params[4] * last(mean_solution(t)))
  ]
end

"""
Predict state space variance to the next observation time index.
"""
function predict_variance!(system_state, mean_solution, model_params, instant_jac; ode_solver)
  # TODO this currently has to be nested since we don't pass current_mean as a parameter, is this possible / faster?
  function variance_RHS(dvariance, current_variance, params, t)
    variance_of_noise = calculate_noise_variance(params, mean_solution, t)
    dvariance .= instant_jac * current_variance + current_variance * instant_jac' + variance_of_noise
  end

  tspan = (system_state.current_time, system_state.next_time)
  var_prob = ODEProblem(variance_RHS, system_state.variance, tspan, model_params)
  var_solution = solve(var_prob, ode_solver; save_everystep = false)

  system_state.variance .= last(var_solution.u)
  system_state
end

"""
Obtain the Kalman filter prediction about a future observation, `rho_{t+Δt}` and `P_{t+Δt}`

# Arguments

- `system_state::SystemState`:

- `params`:

# Returns

- `system_state::SystemState`

"""
function prediction_step!(system_state, params, instant_jac; ode_solver)
  system_state, mean_solution = predict_mean!(system_state, params; ode_solver)
  system_state = predict_variance!(system_state, mean_solution, params, instant_jac; ode_solver)
  # update current_time
  system_state.current_time = system_state.next_time
  system_state
end

#### UPDATE

function update_mean!(system_state, observation, observation_transform, helper_inverse)
  adaptation_coefficient = system_state.variance * observation_transform' * helper_inverse

  system_state.mean += adaptation_coefficient * (observation - dot(observation_transform, system_state.mean))
  system_state
end

function update_variance!(system_state, observation_transform, helper_inverse)
  system_state.variance -=
    system_state.variance * observation_transform' * observation_transform * system_state.variance * helper_inverse
  system_state
end

function calculate_helper_inverse(cov, F, Σ)
  1.0 / (dot(F, cov * F') + Σ)
end

"""
Perform the Kalman filter update step on the predicted mean and variance, given a new observation.
This implements the equations at the beginning of page 4 in Calderazzo et al., Bioinformatics (2018).
This assumes that the observations are collected at fixed time intervals.

# Arguments

- `system_state::SystemState`

- `measurement_variance`: The variance which defines the measurement error, it is ``Σ_ϵ`` in the equation ``y = Fx + Σ_ϵ``.

- `observation_transform`: A 1 x 2 matrix corresponding to the transformation from observed data to molecule number, for mRNA and protein
respectively.

# Returns

- `system_state::SystemState`
"""
function update_step!(system_state, observation, measurement_variance, observation_transform)
  # This is (FP_{t+Δt}F^T + Σ_ϵ)^{-1}
  helper_inverse = calculate_helper_inverse(system_state.variance, observation_transform, measurement_variance)

  update_mean!(system_state, observation, observation_transform, helper_inverse)
  system_state = update_variance!(system_state, observation_transform, helper_inverse)

  system_state
end
