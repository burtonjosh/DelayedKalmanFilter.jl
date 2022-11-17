mutable struct SystemState
    means # array of sol objects
    variances # array of sol objects
    off_diagonals # circular buffer with entries which are arrays
    off_diagonal_timepoints # the s values corresponding to the entries in off_diagonals
    delay # Float64
    off_diagonal_timestep # Float64
    observations # array
    observation_time_points # array
    observation_time_step # Integer
    current_time # Integer
    current_observation # Float64
end

mutable struct SolutionObject
    at_time # 1-d function
    tspan # tuple (t_start, t_end)
end

"""
Calculate the mean and standard deviation of the Normal approximation of the state space for a given
time point in the Kalman filtering algorithm and return it as an array.
"""
function distribution_prediction(
    system_state,
    observation_transform,
    measurement_variance,
)
    mean_prediction = system_state.means[end].at_time(system_state.current_time)[2]
    last_predicted_covariance_matrix = system_state.variances[end].at_time(system_state.current_time)

    variance_prediction =
        dot(
            observation_transform,
            last_predicted_covariance_matrix * observation_transform',
        ) + measurement_variance

    return mean_prediction, variance_prediction
end

"""
Update the current time and observation of the system state
"""
function update_current_time_and_observation!(
    system_state::SystemState
)
    system_state.current_time += system_state.observation_time_step
    observation_index = (system_state.current_time ÷ system_state.observation_time_step) + 1
    system_state.current_observation = system_state.observations[observation_index]
    return system_state
end

"""
    kalman_filter(
        protein_at_observations::Matrix{<:AbstractFloat},
        model_parameters::Vector{<:AbstractFloat},
        measurement_variance::AbstractFloat,
    )

A Kalman filter for a delay-adjusted non-linear stochastic process, based on observation of protein
copy numbers. This implements the filter described by Calderazzo et al., Bioinformatics (2018).

# Arguments

- `protein_at_observations::Matrix{<:AbstractFloat}`: Observed protein. The dimension is N x 2, where N is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number. The filter assumes that observations are generated with a fixed, regular time interval.

- `model_parameters::Vector{<:AbstractFloat}`: A vector containing the model parameters in the following order:
    repression threshold (`P₀`), hill coefficient (`h`), mRNA degradation rate (`μₘ`), protein degradation rate (`μₚ`), basal transcription rate (`αₘ`),
    translation rate (`αₚ`), and time delay (`τ`).

- `measurement_variance::AbstractFloat`: The variance in our measurement. This is given by ``Σ_ϵ`` in Calderazzo et. al. (2018).

# Returns
- `system_state::SystemState`: 

- `predicted_observation_distributions::Matrix{Float64}`: A matrix of size N x 2. The entries in the first column are the predicted state
 space mean and the entries in the second column are the predicted state space variance for each observation.

# Example
```jldoctest
julia> using DelayedKalmanFilter

julia> protein = [0. 105.; 10. 100.; 20. 98.]; # times are 0., 10., 20., and protein levels are 105., 100., and 98. respectively

julia> model_parameters = [100.0, 5.0, 0.1, 0.1, 1.0, 1.0, 15.0];

julia> measurement_variance = 1000.0;

julia> system_state, distributions = kalman_filter(
           protein,
           model_parameters,
           measurement_variance
       );

julia> distributions[1,:]
2-element Vector{Float64}:
  77.80895986786031
  8780.895986786032
```
"""
function kalman_filter(
    protein_at_observations::Matrix{<:AbstractFloat},
    model_parameters::Vector{<:AbstractFloat},
    measurement_variance::AbstractFloat;
    off_diagonal_timestep::AbstractFloat=1.0,
)
    # F in the paper
    observation_transform = [0.0 1.0]
    
    # initialise state space and distribution predictions
    system_state, predicted_observation_distributions = kalman_filter_state_space_initialisation(
        protein_at_observations,
        model_parameters,
        observation_transform,
        measurement_variance,
        off_diagonal_timestep,
    )

    # loop through observations and at each observation apply the Kalman prediction step and then the update step
    for observation_index = 2:size(protein_at_observations, 1)
        system_state = kalman_prediction_step!(
            system_state,
            model_parameters,
        )

        # between the prediction and update steps we record the predicted mean and variance our likelihood
        predicted_observation_distributions[observation_index,:] .= distribution_prediction(
                system_state,
                observation_transform,
                measurement_variance,
            )

        # TODO the update step on the last loop iteration is redundant for likelihood calculation
        system_state = kalman_update_step!(
            system_state,
            measurement_variance,
            observation_transform,
        )

    end # for
    return system_state, predicted_observation_distributions
end # function

"""
Initialise the state space mean for a given set of time states and the ODE system's steady state
"""
function initialise_state_space_mean(steady_state,τ)

    function initial_mean!(du,u,p,t)
        du[1] = 0
        du[2] = 0
    end
    u0 = steady_state
    tspan = (-τ, 0.)
    prob = ODEProblem(initial_mean!,u0,tspan)
    sol = solve(prob)
    return [SolutionObject(sol, tspan)] # mean is an array of solution objects
end

"""
Initialse the state space variance for a given set of time states and the ODE system's steady state
"""
function initialise_state_space_variance(
    steady_state,
    τ;
    mRNA_scaling = 20.0,
    protein_scaling = 100.0,
)
    function initial_variance!(du,u,p,t)
        du = 0
    end
    u0 = [steady_state[1]*mRNA_scaling 0.0; 0.0 steady_state[2]*protein_scaling]
    tspan = (-τ,0.0)
    prob = ODEProblem(initial_variance!,u0,tspan)
    sol = solve(prob)

    return [SolutionObject(sol, tspan)] # variance is an array of solution objects
end

"""
Initialse the off diagonals for a given set of time states and the ODE system's steady state
"""
function initialise_off_diagonals(
    steady_state,
    τ,
    off_diagonal_timestep;
    mRNA_scaling = 20.0,
    protein_scaling = 100.0,
)
    number_of_offdiagonal_timepoints = round(Int,τ/off_diagonal_timestep) + 1
    off_diagonals = CircularBuffer{Array}(number_of_offdiagonal_timepoints)
    off_diagonal_timepoints = CircularBuffer{Float64}(number_of_offdiagonal_timepoints)

    for timepoint in LinRange(-τ, 0, number_of_offdiagonal_timepoints)
        # TODO round might not be a good solution
        push!(off_diagonal_timepoints, round(timepoint, digits=2))
    end

    function initial_off_diagonal!(du,u,p,t)
        du = 0
    end
    u0 = [0.0 0.0; 0.0 0.0]
    tspan = (-τ,0.0)
    prob = ODEProblem(initial_off_diagonal!,u0,tspan)
    sol = solve(prob)

    # fill!(off_diagonals,[SolutionObject(sol, tspan)])
    for _ in 1:number_of_offdiagonal_timepoints
        push!(off_diagonals, [SolutionObject(sol,tspan)])
    end

    # final off diagonal -- negative times don't matter because we don't use them
    u0 = [steady_state[1]*mRNA_scaling 0.0; 0.0 steady_state[2]*protein_scaling]
    prob = remake(prob, u0=u0)
    sol = solve(prob)

    off_diagonals[end] = [SolutionObject(sol, tspan)]

    return off_diagonals, off_diagonal_timepoints
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
    off_diagonal_timestep,
)
    steady_state = calculate_steady_state_of_ode(model_parameters)

    # construct system state space
    τ = model_parameters[7]
    means = initialise_state_space_mean(steady_state, τ)
    variances = initialise_state_space_variance(steady_state, τ)
    off_diagonals, off_diagonal_timepoints = initialise_off_diagonals(steady_state, τ, off_diagonal_timestep)

    observation_time_step = Int(protein_at_observations[2,1] - protein_at_observations[1,1])
    current_time = Int(protein_at_observations[1,1])
    current_observation = protein_at_observations[1,2]

    system_state = SystemState(
        means,
        variances,
        off_diagonals,
        off_diagonal_timepoints,
        τ,
        off_diagonal_timestep,
        protein_at_observations[:,2],
        protein_at_observations[:,1],
        observation_time_step,
        current_time,
        current_observation
    )

    # initialise distributions
    predicted_observation_distributions =
        zeros(length(system_state.observations), 2)
    predicted_observation_distributions[1,:] .= distribution_prediction(
        system_state,
        observation_transform,
        measurement_variance,
    )

    # update the past ("negative time")
    system_state = kalman_update_step!(
        system_state,
        measurement_variance,
        observation_transform,
    )

    return system_state, predicted_observation_distributions
end # function

function construct_instant_jacobian(model_parameters)
    [
        -model_parameters[3] 0.0
        model_parameters[6] -model_parameters[4]
    ]
end

function construct_delayed_jacobian(model_parameters, past_protein)
    [
        0.0 model_parameters[5]*∂hill∂p(past_protein, model_parameters[1], model_parameters[2])
        0.0 0.0
    ]
end

function get_mean_at_time(t, system_state)
    for solution_object in system_state.means
        # start_time = solution_object.tspan[1]
        # end_time = solution_object.tspan[end]

        if solution_object.tspan[1] <= t <= solution_object.tspan[end]
            return solution_object.at_time(t)
        end 
    end
end

function get_variance_at_time(t, system_state)
    for solution_object in system_state.variances
        # start_time = solution_object.tspan[1]
        # end_time = solution_object.tspan[end]

        if solution_object.tspan[1] <= t <= solution_object.tspan[end]
            # if start_time <= t <= end_time
            return solution_object.at_time(t)
        end 
    end
end

function get_specific_off_diagonal_value_at_time(t, off_diagonal_entry)
    for solution_object in off_diagonal_entry
        start_time = solution_object.tspan[1]
        end_time = solution_object.tspan[end]
        if start_time <= t <= end_time
            return solution_object.at_time(t)
        end
    end
end

function get_off_diagonal_at_time_combination(time_1, time_2, system_state)

    if time_1 == time_2
        return get_variance_at_time(time_1, system_state)
    end

    transpose_result = false
    if time_1 > time_2
        transpose_result = true
    end

    min_time = min(time_1, time_2)
    max_time = max(time_1, time_2)
    past_values = fill(zeros(2,2), length(system_state.off_diagonal_timepoints))

    for (off_diagonal_index, off_diagonal_entry) in enumerate(system_state.off_diagonals)
        this_value = get_specific_off_diagonal_value_at_time(max_time, off_diagonal_entry)

        if transpose_result
            past_values[off_diagonal_index] = this_value'
        else
            past_values[off_diagonal_index] = this_value
        end
    end

    interpolated_values = linear_interpolation(
        system_state.off_diagonal_timepoints,
        past_values
    )
    
    return interpolated_values(min_time)

end

"""
Predict state space mean to the next observation time index
"""
function predict_state_space_mean!(
    system_state,
    model_parameters,
    )

    function state_space_mean_RHS(du,u,h,p,t)
        past_time = t - system_state.delay # p[7]
        # if past_time >= tspan[1] # history function in case τ < number_of_hidden_states
        past_protein = h(p,past_time)
        # else # otherwise pre defined indexer
            # past_protein = get_mean_at_time(past_time, system_state)[2]
        # end

        du[1] = -p[3]*u[1] + p[5]*hill_function(past_protein, p[1], p[2])
        du[2] = p[6]*u[1] - p[4]*u[2]
    end

    protein_history(p,t) = get_mean_at_time(t, system_state)[2]
    tspan = (Float64(system_state.current_time), Float64(system_state.current_time + system_state.observation_time_step))
    mean_prob = DDEProblem(
        state_space_mean_RHS,
        get_mean_at_time(system_state.current_time, system_state),
        protein_history,
        tspan,
        model_parameters; constant_lags=[system_state.delay]
    )
    mean_solution = solve(
        mean_prob,
        MethodOfSteps(Euler()),
        dt=1.,
        adaptive=false,
        saveat=1.,
        dtmin=1.,
        dtmax=1.
    )
    mean_solution_object = SolutionObject(mean_solution, tspan)
    push!(system_state.means, mean_solution_object)
    return system_state
end

function propagate_existing_off_diagonals!(
    system_state,
    off_diagonal_RHS,
    current_time,
    next_end_time,
)
    diag_tspan = (current_time, next_end_time)

    # we want to do P(s,t) -> P(s,t+nΔt) for s = t-τ:t
    for (off_diagonal_index, intermediate_time) in enumerate(system_state.off_diagonal_timepoints)
        # P(s,t)
        covariance_matrix_intermediate_to_current = get_off_diagonal_at_time_combination(
            intermediate_time,
            current_time,
            system_state
        )

        history(s, t) = get_specific_off_diagonal_value_at_time(t, system_state.off_diagonals[off_diagonal_index])

        off_diag_prob = DDEProblem(
            off_diagonal_RHS,
            covariance_matrix_intermediate_to_current,
            history,
            diag_tspan,
            intermediate_time; constant_lags=[system_state.delay]
        )

        off_diag_solution = solve(
            off_diag_prob,
            MethodOfSteps(Euler()),
            dt=1.,
            adaptive=false,
            saveat=1.,
            dtmin=1.,
            dtmax=1.
        )

        push!(system_state.off_diagonals[off_diagonal_index], SolutionObject(off_diag_solution, diag_tspan))

    end

    return system_state
end

function propagate_new_off_diagonals!(
    system_state,
    off_diagonal_RHS,
    last_off_diagonal_timepoint,
    next_end_time,
)
    # we want to do P(s,s) -> P(s,t+nΔt) for s = t:t+Δt
    # TODO
    current_off_diagonal_time_point = last_off_diagonal_timepoint + system_state.off_diagonal_timestep
    while current_off_diagonal_time_point <= next_end_time
        diag_tspan = (current_off_diagonal_time_point, next_end_time)
        
        # P(s,t)
        initial_variance = get_variance_at_time(current_off_diagonal_time_point, system_state)
       
        function off_diag_solution(t)
            return initial_variance
        end
        variance_tspan = (current_off_diagonal_time_point, current_off_diagonal_time_point)
        push!(system_state.off_diagonals, [SolutionObject(off_diag_solution, variance_tspan)])
        push!(system_state.off_diagonal_timepoints,current_off_diagonal_time_point)

        interpolation_object = generate_off_diagonal_history_function(current_off_diagonal_time_point, system_state)
        history(s, t) = interpolation_object(t)

        past_function(t) = interpolation_object(t)
        history_tspan = (
            system_state.off_diagonal_timepoints[1],
            current_off_diagonal_time_point,
        )
        pushfirst!(system_state.off_diagonals[end], SolutionObject(past_function, history_tspan))

        if current_off_diagonal_time_point != next_end_time

            off_diag_prob = DDEProblem(
                off_diagonal_RHS,
                initial_variance,
                history,
                diag_tspan,
                current_off_diagonal_time_point; constant_lags=[system_state.delay]
            )

            off_diag_solution = solve(
                off_diag_prob,
                MethodOfSteps(Euler()),
                dt=1.,
                adaptive=false,
                saveat=1.,
                dtmin=1.,
                dtmax=1.
            )

            push!(system_state.off_diagonals[end],SolutionObject(off_diag_solution, diag_tspan))
        end

        current_off_diagonal_time_point += system_state.off_diagonal_timestep
    end
    return system_state
end

function propagate_variance!(
    system_state,
    variance_RHS,
    current_time,
    next_end_time,
    model_parameters,
)
    initial_variance = get_variance_at_time(current_time, system_state)
    tspan = (current_time, next_end_time)
    variance_prob = ODEProblem(
        variance_RHS,
        initial_variance,
        tspan,
        model_parameters
    )
    
    variance_solution = solve(
        variance_prob,
        Euler(),
        dt=1.,
        adaptive=false,
        saveat=1.,
        dtmin=1.,
        dtmax=1.
    )
    
    variance_solution_object = SolutionObject(variance_solution, tspan)
    push!(system_state.variances, variance_solution_object)

    return system_state
end

"""
Predict state space variance to the next observation time index.
There are three separate steps in the variance prediction:
    (1) integrate P(t,s) to P(t+nΔt,s) for s = t-τ:t, where nΔt is the number of hidden
        states. (propagate_existing_off_diagonals!())
    (2) Integrate P(t,t) to P(t+nΔt,t+nΔt). (propagate_variance!())
    (3) Integrate P(t,t) to P(t,t+nΔt) for t in t:t+nΔt-1. (propagate_new_off_diagonals!())
"""
function predict_variance_and_off_diagonals!(
    system_state,
    model_parameters,
    )

    instant_jacobian = construct_instant_jacobian(model_parameters)
    
    function variance_RHS(dvariance, current_variance, params, t)
        past_time = t - system_state.delay
        past_protein = get_mean_at_time(past_time, system_state)[2]
        current_mean = get_mean_at_time(t, system_state)

        # P(t-τ, t)
        if past_time < 0 || t < 0
            past_to_now_diagonal_variance = zeros(2,2)
        else
            past_to_now_diagonal_variance = get_off_diagonal_at_time_combination(past_time, t, system_state)
        end
        
        delayed_jacobian = construct_delayed_jacobian(model_parameters, past_protein)

        variance_of_noise = [params[3]*current_mean[1]+params[5]*hill_function(past_protein, params[1], params[2]) 0.0;
                             0.0 params[6]*current_mean[1]+params[4]*current_mean[2]]

        dvariance .= instant_jacobian*current_variance + current_variance*instant_jacobian' +
             delayed_jacobian*past_to_now_diagonal_variance +
             past_to_now_diagonal_variance'*delayed_jacobian' +
             variance_of_noise
    end

    function off_diagonal_RHS(dcovariance, covariance, history, s, diag_t) # s is the intermediate time index
        past_time = diag_t - system_state.delay # t - τ
        past_protein = get_mean_at_time(past_time, system_state)[2]
        
        delayed_jacobian = construct_delayed_jacobian(model_parameters, past_protein)

        covariance_matrix_intermediate_to_past = history(s, past_time)

        dcovariance .= covariance*instant_jacobian' +
             covariance_matrix_intermediate_to_past*delayed_jacobian'
    end

    # TODO
    # implement a while loop here
    current_time = copy(system_state.current_time)
    last_off_diagonal_timepoint = system_state.off_diagonal_timepoints[end]
    next_observation_time = system_state.current_time + system_state.observation_time_step

    while current_time < next_observation_time
        next_end_time = min(next_observation_time, last_off_diagonal_timepoint + system_state.delay)

        system_state = propagate_existing_off_diagonals!(
            system_state,
            off_diagonal_RHS,
            current_time,
            next_end_time,
        )
        system_state = propagate_variance!(
            system_state,
            variance_RHS,
            current_time,
            next_end_time,
            model_parameters,
        )
        system_state = propagate_new_off_diagonals!(
            system_state,
            off_diagonal_RHS,
            last_off_diagonal_timepoint,
            next_end_time,
        )
        current_time = next_end_time
    end

    return system_state
    # in the case of τ < observation_time_step, we have to do the below procedure multiple times (ceil(observation/τ) times)
    # since otherwise certain P(t-τ,s) values will not exist #TODO bad explanation
    # initial_condition_times = Int.([current_number_of_states + states.discrete_delay*x for x in 0:ceil(states.number_of_hidden_states/states.discrete_delay)-1])
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
)
    system_state = predict_state_space_mean!(
        system_state,
        model_parameters,
    )

    system_state = predict_variance_and_off_diagonals!(
        system_state,
        model_parameters,
    )

    # move system_state to the next observation for the update step
    system_state = update_current_time_and_observation!(system_state)

    # println("Predict")
    # observation_transform = [1. 0.]
    # measurement_variance = 10_000.0
    # println("Current time: ", system_state.current_time)
    # println("mean at 0: ", get_mean_at_time(0.0, system_state))
    # println("mean: ",get_mean_at_time(system_state.current_time, system_state))
    # println("std: ",sqrt(dot(
    #         observation_transform,
    #         system_state.variances[end].at_time(system_state.current_time) * observation_transform',
    #     ) + measurement_variance))
    # println()

    return system_state
end # function

function update_mean!(
    system_state,
    most_recent_off_diagonal,
    observation_transform,
    helper_inverse,
)
    most_recent_mean = get_mean_at_time(system_state.current_time, system_state)
    adaptation_coefficient(t) = most_recent_off_diagonal(t)*observation_transform'*helper_inverse
    
    # need to copy current_observation since it get's updated later
    observation = copy(system_state.current_observation)

    function update_addition_function(t)
        adaptation_coefficient(t)*( observation -
            dot(observation_transform, most_recent_mean) )
    end
    for index in 1:length(system_state.means) # TODO don't need to update everything -- fix this
        system_state.means[index].at_time = system_state.means[index].at_time + update_addition_function
    end
    
    return system_state
end

function update_variance!(
    system_state,
    most_recent_off_diagonal,
    observation_transform,
    helper_inverse
)
    function update_addition_function(t)
        return -most_recent_off_diagonal(t)*observation_transform'*observation_transform*most_recent_off_diagonal(t)'*helper_inverse
    end

    for index in 1:length(system_state.variances) # TODO don't need to update everything -- fix this
        system_state.variances[index].at_time = system_state.variances[index].at_time + update_addition_function
    end

    return system_state
end

function update_off_diagonals!(
    system_state,
    most_recent_off_diagonal,
    observation_transform,
    helper_inverse,
)
    for (off_diagonal_index, off_diagonal_entry) in enumerate(system_state.off_diagonals)

        this_off_diagonal_timepoint = copy(system_state.off_diagonal_timepoints[off_diagonal_index])

        function update_addition_function(t)
            return -most_recent_off_diagonal(this_off_diagonal_timepoint) * 
            observation_transform' * observation_transform * 
            most_recent_off_diagonal(t)' * helper_inverse
        end

        for index in 1:length(off_diagonal_entry) # TODO don't need to update everything -- fix this
            off_diagonal_entry[index].at_time = off_diagonal_entry[index].at_time + update_addition_function
        end
    end
    return system_state
end

function get_most_recent_offdiagonal_as_function(system_state)
    most_recent_off_diagonals = fill(zeros(2,2), length(system_state.off_diagonal_timepoints))

    for (off_diagonal_index, off_diagonal_entry) in enumerate(system_state.off_diagonals)
        most_recent_off_diagonals[off_diagonal_index] = off_diagonal_entry[end].at_time(system_state.current_time)
    end


    return linear_interpolation(
        copy(system_state.off_diagonal_timepoints),
        most_recent_off_diagonals,
        # extrapolation_bc=Line()
        )
end

function generate_off_diagonal_history_function(off_diagonal_time, system_state)

    required_off_diagonals = fill(zeros(2,2), length(system_state.off_diagonal_timepoints))

    for (off_diagonal_index, off_diagonal_entry) in enumerate(system_state.off_diagonals)
        required_off_diagonals[off_diagonal_index] = off_diagonal_entry[end].at_time(off_diagonal_time)' # transpose
    end

    return linear_interpolation(
        copy(system_state.off_diagonal_timepoints),
        required_off_diagonals,
        )
end

function calculate_helper_inverse(
    predicted_final_covariance_matrix,
    observation_transform,
    measurement_variance,
)
    1.0 / (
        dot(
            observation_transform,
            predicted_final_covariance_matrix * observation_transform',
        ) + measurement_variance
    )
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
function kalman_update_step!(
    system_state,
    measurement_variance,
    observation_transform,
)
    # This is P(t+Δt,t+Δt) in the paper
    most_recent_variance = system_state.variances[end].at_time(system_state.current_time)

    # This is (FP_{t+Δt}F^T + Σ_ϵ)^{-1}
    helper_inverse = calculate_helper_inverse(
        most_recent_variance,
        observation_transform,
        measurement_variance,
    )

    # this is P(s, t+Δt) as a function of s
    most_recent_off_diagonal = get_most_recent_offdiagonal_as_function(system_state)

    system_state = update_mean!(
        system_state,
        most_recent_off_diagonal,
        observation_transform,
        helper_inverse,
    )

    system_state = update_variance!(
        system_state,
        most_recent_off_diagonal,
        observation_transform,
        helper_inverse,
    )

    system_state = update_off_diagonals!(
        system_state,
        most_recent_off_diagonal,
        observation_transform,
        helper_inverse,
    )

    # println("Update")
    # println("Current time: ", system_state.current_time)
    # println("mean at 0: ", get_mean_at_time(0.0,system_state))
    # println("var at 0: ", get_variance_at_time(0.0,system_state))
    # println("mean: ", system_state.means[end].at_time(system_state.current_time))
    # println("std: ",sqrt(dot(
    #         observation_transform,
    #         system_state.variances[end].at_time(system_state.current_time) * observation_transform',
    #     ) + measurement_variance))
    # println()

    return system_state
end