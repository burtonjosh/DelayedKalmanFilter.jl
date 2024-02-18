mutable struct SystemState
  mean
  variance
  current_time
end

const OBS_TIME_STEP = 10

"""
Calculate the mean and standard deviation of the Normal approximation of the state space for a given
time point in the Kalman filtering algorithm and return it as an array.
"""
function distribution_prediction(system_state::SystemState, observation_transform, measurement_variance)
  mean_prediction = dot(system_state.mean, observation_transform)
  variance_prediction = dot(observation_transform, system_state.variance * observation_transform') + measurement_variance
  return mean_prediction, variance_prediction
end

"""
    kalman_filter(
        protein_at_observations,
        model_parameters,
        measurement_variance;
    )

A Kalman filter for a delay-adjusted non-linear stochastic process, based on observation of protein
copy numbers. This implements the filter described by Calderazzo et al., Bioinformatics (2018).

# Arguments

- `protein_at_observations`: Observed protein. The dimension is N x 2, where N is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number. The filter assumes that observations are generated with a fixed, regular time interval.

- `model_parameters`: A vector containing the model parameters in the following order:
    repression threshold (`P₀`), hill coefficient (`h`), mRNA degradation rate (`μₘ`), protein degradation rate (`μₚ`), basal transcription rate (`αₘ`),
    translation rate (`αₚ`), and time delay (`τ`).

- `measurement_variance`: The variance in our measurement. This is given by ``Σ_ε``  in Calderazzo et. al. (2018).

# Returns
- `predicted_observation_distributions::Matrix{<:AbstractFloat}`: A matrix of size N x 2. The entries in the first column are the predicted state
 space mean and the entries in the second column are the predicted state space variance for each observation.

# Example
```jldoctest
julia> using DelayedKalmanFilter

julia> protein = [0. 105.; 10. 100.; 20. 98.]; # times are 0., 10., 20., and protein levels are 105., 100., and 98. respectively

julia> model_parameters = [300.0, 4.0, 0.1, 0.1, 1.0, 1.0];

julia> measurement_variance = 1000.0;

julia> system_state, distributions = kalman_filter(
         protein,
         model_parameters,
         measurement_variance
       );

julia> distributions[1, :]
2-element Vector{Float64}:
    98.83526916576103
 10883.526916576104
```
"""
function kalman_filter(
  protein_at_observations,
  model_parameters,
  measurement_variance;
  adaptive = false,
  alg = Euler(),
  euler_dt = 1.0,
  relative_tolerance = 1e-6,
  absolute_tolerance = 1e-6,
)
  # F in the paper
  observation_transform = [0.0 1.0]
  # Jacobian of the system
  instant_jac = [
    -model_parameters[3] 0.0
    model_parameters[6] -model_parameters[4]
  ]

  # initialise state space and distribution predictions
  system_state, predicted_observation_distributions = kalman_filter_state_space_initialisation(
    protein_at_observations,
    model_parameters,
    observation_transform,
    measurement_variance,
  )

  # loop through observations and at each observation apply the Kalman prediction step and then the update step
  for (observation_index, (time, protein)) in enumerate(eachrow(protein_at_observations)[2:end])
    system_state = kalman_prediction_step!(
      system_state,
      model_parameters,
      instant_jac;
      adaptive,
      alg,
      euler_dt,
      relative_tolerance,
      absolute_tolerance,
    )

    # between the prediction and update steps we record the predicted mean and variance our likelihood
    predicted_observation_distributions[observation_index + 1, :] .=
      distribution_prediction(system_state, observation_transform, measurement_variance)

    system_state = kalman_update_step!(system_state, protein, measurement_variance, observation_transform)
  end
  return system_state, predicted_observation_distributions
end

"""
A function for initialisation of the state space mean and variance, and update for the "negative" times that
are a result of the time delay. Initialises the negative times using the steady state of the deterministic system,
and then updates them with kalman_update_step.

# Arguments

- `protein_at_observations`: Observed protein. The dimension is n x 2, where n is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number at
    that time. The filter assumes that observations are generated with a fixed, regular time interval.

- `model_parameters`: An array containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

- `observation_transform`: A 1 x 2 matrix corresponding to the transformation from observed data to molecule number, for mRNA and protein
    respectively.

- `measurement_variance`: The variance in our measurement. This is given by Sigma epsilon in Calderazzo et. al. (2018).

# Returns
- `system_state::SystemState`:

- `predicted_observation_distributions::Matrix{Float64}`: A matrix of size N x 2. The entries in the first column are the predicted state
 space mean and the entries in the second column are the predicted state space variance for each observation.
"""
function kalman_filter_state_space_initialisation(
  protein_at_observations,
  model_parameters,
  observation_transform,
  measurement_variance,
)
  steady_state = calculate_steady_state_of_ode(model_parameters)

  # construct system state space
  mean = steady_state
  variance = diagm(mean .* [20.0, 100.0])
  system_state = SystemState(mean, variance, protein_at_observations[1, 1])

  # initialise distributions
  predicted_observation_distributions = zeros(eltype(mean), first(size(protein_at_observations)), 2)
  predicted_observation_distributions[1, :] .= distribution_prediction(system_state, observation_transform, measurement_variance)

  # inital update step
  # kalman_update_step!(system_state, protein_at_observations[1, 2], measurement_variance, observation_transform)
  return system_state, predicted_observation_distributions
end

function state_space_mean_RHS(du, u, p, t)
  du .= [-p[3] 0; p[6] -p[4]] * u + [p[5] * hill_function(u[2], p[1], p[2]); 0]
end

"""
Predict state space mean to the next observation time index
"""
function predict_mean!(system_state, model_parameters; adaptive, alg, euler_dt, relative_tolerance, absolute_tolerance)
  tspan = (system_state.current_time, system_state.current_time + OBS_TIME_STEP)

  mean_prob = ODEProblem(state_space_mean_RHS, system_state.mean, tspan, model_parameters)
  if adaptive
    mean_solution = solve(mean_prob, alg, reltol = relative_tolerance, abstol = absolute_tolerance)
  else
    mean_solution = solve(mean_prob, alg, dt = euler_dt, adaptive = false, saveat = euler_dt, dtmin = euler_dt, dtmax = euler_dt)
  end
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
function predict_variance!(
  system_state,
  mean_solution,
  model_parameters,
  instant_jac;
  adaptive,
  alg,
  euler_dt,
  relative_tolerance,
  absolute_tolerance,
)
  # TODO this currently has to be nested since we don't pass current_mean as a parameter, is this possible / faster?
  function variance_RHS(dvariance, current_variance, params, t)
    variance_of_noise = calculate_noise_variance(params, mean_solution, t)

    dvariance .= instant_jac * current_variance + current_variance * instant_jac' + variance_of_noise
  end

  tspan = (system_state.current_time, system_state.current_time + OBS_TIME_STEP)

  var_prob = ODEProblem(variance_RHS, system_state.variance, tspan, model_parameters)

  if adaptive
    var_solution = solve(var_prob, alg, reltol = relative_tolerance, abstol = absolute_tolerance)
  else
    var_solution = solve(var_prob, alg, dt = euler_dt, adaptive = false, saveat = euler_dt, dtmin = euler_dt, dtmax = euler_dt)
  end

  system_state.variance .= last(var_solution.u)
  system_state
end

"""
Perform the Kalman filter prediction about future observation, based on current knowledge i.e. current
state space mean and variance. This gives rho_{t+Δt-τ:t+Δt} and P_{t+Δt-τ:t+Δt},
using the differential equations in supplementary section 4 of Calderazzo et al., Bioinformatics (2018),
approximated using a forward Euler scheme.

# Arguments

- `system_state::SystemState`:

- `model_parameters`: An array containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

# Returns

- `system_state::SystemState`
"""
function kalman_prediction_step!(
  system_state,
  model_parameters,
  instant_jac;
  adaptive,
  alg,
  euler_dt,
  relative_tolerance,
  absolute_tolerance,
)
  system_state, mean_solution =
    predict_mean!(system_state, model_parameters; adaptive, alg, euler_dt, relative_tolerance, absolute_tolerance)
  system_state = predict_variance!(
    system_state,
    mean_solution,
    model_parameters,
    instant_jac;
    adaptive,
    alg,
    euler_dt,
    relative_tolerance,
    absolute_tolerance,
  )
  system_state
end

#### UPDATE

function update_mean!(system_state, observation, observation_transform, helper_inverse)
  adaptation_coefficient = system_state.variance * observation_transform' * helper_inverse
  # need to copy current_observation since it get's updated later
  system_state.mean .+= adaptation_coefficient * (observation - dot(observation_transform, system_state.mean))
  system_state
end

function update_variance!(system_state, observation_transform, helper_inverse)
  system_state.variance .-=
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
function kalman_update_step!(system_state, observation, measurement_variance, observation_transform)
  # This is (FP_{t+Δt}F^T + Σ_ϵ)^{-1}
  helper_inverse = calculate_helper_inverse(system_state.variance, observation_transform, measurement_variance)

  update_mean!(system_state, observation, observation_transform, helper_inverse)
  update_variance!(system_state, observation_transform, helper_inverse)
  # update time
  system_state.current_time += OBS_TIME_STEP

  system_state
end
