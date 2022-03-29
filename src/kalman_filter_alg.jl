Base.@kwdef struct ModelParameters
    repression_threshold::Real = 10000.0
    hill_coefficient::Real = 5.0
    mRNA_degradation_rate::Real = log(2)/30
    protein_degradation_rate::Real = log(2)/90
    basal_transcription_rate::Real = 1.0
    translation_rate::Real = 1.0
    time_delay::Real = 29.0
end

struct StateSpace
    mean::AbstractArray{<:Real}
    variance::AbstractArray{<:Real}
end

mutable struct StateAndDistributions
    state_space::StateSpace
    distributions::Array{Normal{Float64}}
end

struct TimeConstructor
    number_of_observations::Real
    observation_time_step::Real
    discretisation_time_step::Real
    discrete_delay::Real
    initial_number_of_states::Real
    number_of_hidden_states::Real
    total_number_of_states::Real
end

"""
Function which returns an instance of TimeConstructor, which is defines the numbers of
states for various uses in the Kalman filter.
"""
function TimeConstructorFunction(
    protein_at_observations::AbstractArray{<:Real},
    time_delay::Real,
    discretisation_time_step::Real = 1.0
    )

    number_of_observations = size(protein_at_observations,1)
    observation_time_step = protein_at_observations[2,1]-protein_at_observations[1,1]

    discrete_delay = Int64(round(time_delay/discretisation_time_step))
    initial_number_of_states = discrete_delay + 1
    number_of_hidden_states = Int64(round(observation_time_step/discretisation_time_step))

    total_number_of_states = initial_number_of_states + (number_of_observations - 1)*number_of_hidden_states

    return TimeConstructor(
        number_of_observations,
        observation_time_step,
        discretisation_time_step,
        discrete_delay,
        initial_number_of_states,
        number_of_hidden_states,
        total_number_of_states
    )
end

"""
Calculate the mean and variance of the Normal approximation of the state space for a given
time point in the Kalman filtering algorithm
"""
function distribution_prediction_at_given_time(
    state_space::StateSpace,
    states::TimeConstructor,
    given_time::Int,
    observation_transform::AbstractArray{<:Real},
    measurement_variance::Real
    )

    mean_prediction = dot(observation_transform,state_space.mean[given_time,2:3])
    last_predicted_covariance_matrix = state_space.variance[[given_time,states.total_number_of_states+given_time],
                                                            [given_time,states.total_number_of_states+given_time]]
    variance_prediction = dot(observation_transform,last_predicted_covariance_matrix*transpose(observation_transform)) + measurement_variance

    return Normal(mean_prediction,sqrt(variance_prediction))
end

# if no time given, use initial number of states
function distribution_prediction_at_given_time(
    state_space::StateSpace,
    states::TimeConstructor,
    observation_transform::AbstractArray{<:Real},
    measurement_variance::Real
    )

    mean_prediction = dot(observation_transform,state_space.mean[states.initial_number_of_states,2:3])
    last_predicted_covariance_matrix = state_space.variance[[states.initial_number_of_states,states.total_number_of_states+states.initial_number_of_states],
                                                            [states.initial_number_of_states,states.total_number_of_states+states.initial_number_of_states]]
    variance_prediction = dot(observation_transform,last_predicted_covariance_matrix*transpose(observation_transform)) + measurement_variance

    return Normal(mean_prediction,sqrt(variance_prediction))
end

"""
Calculate the current number of states for a given observation time
"""
function calculate_current_number_of_states(
    current_observation_time::Real,
    states::TimeConstructor,
    time_delay::Real
    )
    return Int64(round(current_observation_time/states.observation_time_step))*states.number_of_hidden_states + states.initial_number_of_states
end

"""
Perform a delay-adjusted non-linear stochastic Kalman filter based on observation of protein
copy numbers. This implements the filter described by Calderazzo et al., Bioinformatics (2018).

# Arguments

- `protein_at_observations::AbstractArray{<:Real}`: Observed protein. The dimension is n x 2, where n is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number at
    that time. The filter assumes that observations are generated with a fixed, regular time interval.

- `model_parameters::ModelParameters`: A ModelParameters object containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

- `measurement_variance::Real`: The variance in our measurement. This is given by Sigma epsilon in Calderazzo et. al. (2018).

# Returns

- `state_space_and_distributions::StateAndDistributions`: TODO
"""
function kalman_filter(
    protein_at_observations::AbstractArray{<:Real},
    model_parameters::ModelParameters,
    measurement_variance::Real
    )

    time_delay = model_parameters.time_delay
    observation_transform = [0. 1.]
    states = TimeConstructorFunction(protein_at_observations,time_delay)
    # initialise state space and distribution predictions
    state_space_and_distributions = kalman_filter_state_space_initialisation(protein_at_observations,
                                                                             model_parameters,
                                                                             measurement_variance)

    # loop through observations and at each observation apply the Kalman prediction step and then the update step
    for observation_index in 1:size(protein_at_observations,1)-1
        current_observation = protein_at_observations[1 + observation_index,:]
        state_space = kalman_prediction_step(state_space_and_distributions.state_space,
                                             states,
                                             current_observation,
                                             model_parameters)

        current_number_of_states = calculate_current_number_of_states(current_observation[1],states,time_delay)
        # between the prediction and update steps we record the normal distributions for our likelihood
        state_space_and_distributions.distributions[observation_index+1] = distribution_prediction_at_given_time(state_space,
                                                                                                                 states,
                                                                                                                 current_number_of_states,
                                                                                                                 observation_transform,
                                                                                                                 measurement_variance)
        state_space = kalman_update_step(state_space,
                                         states,
                                         current_observation,
                                         time_delay,
                                         measurement_variance)
    end # for
    return state_space_and_distributions
end # function

"""
Initialse the state space mean for a given set of time states and the system steady state
"""
function initialise_state_space_mean(
    states::TimeConstructor,
    steady_state
    )

    state_space_mean = Array{Float64}(undef,(states.total_number_of_states,3))

    state_space_mean[:,1] .= LinRange(-states.discrete_delay,
                                      states.total_number_of_states-states.discrete_delay-1,
                                      states.total_number_of_states)
    state_space_mean[1:states.initial_number_of_states,2] .= steady_state[1]
    state_space_mean[1:states.initial_number_of_states,3] .= steady_state[2]

    return state_space_mean
end

"""
Initialse the state space variance for a given set of time states and the system steady state
"""
function initialise_state_space_variance(
    states::TimeConstructor,
    steady_state;
    mRNA_scaling=20.0,
    protein_scaling=100.0
    )

    state_space_variance = zeros((2*(states.total_number_of_states),2*(states.total_number_of_states)));

    initial_mRNA_variance = steady_state[1]*mRNA_scaling
    initial_protein_variance = steady_state[2]*protein_scaling
    # diagm(diagind(A)[1 .<= diagind(A).< initial_number_of_states])
    for diag_index in 1:states.initial_number_of_states
        state_space_variance[diag_index,diag_index] = initial_mRNA_variance
        state_space_variance[diag_index + states.total_number_of_states,diag_index + states.total_number_of_states] = initial_protein_variance
    end #for
    return state_space_variance
end

"""
A function for initialisation of the state space mean and variance, and update for the "negative" times that
are a result of the time delay. Initialises the negative times using the steady state of the deterministic system,
and then updates them with kalman_update_step.

# Arguments

- `protein_at_observations::AbstractArray{<:Real}`: Observed protein. The dimension is n x 2, where n is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number at
    that time. The filter assumes that observations are generated with a fixed, regular time interval.

- `model_parameters::ModelParameters`: A ModelParameters object containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

- `measurement_variance::Real`: The variance in our measurement. This is given by Sigma epsilon in Calderazzo et. al. (2018).

# Returns

- `state_and_distributions::StateAndDistributions`: TODO
"""
function kalman_filter_state_space_initialisation(
    protein_at_observations::AbstractArray{<:Real},
    model_parameters::ModelParameters,
    measurement_variance::Real = 10.0
    )

    time_delay = model_parameters.time_delay

    states = TimeConstructorFunction(protein_at_observations,time_delay) # hidden discretisation_time_step in this function
    steady_state = calculate_steady_state_of_ode(model_parameters)

    # construct state space
    state_space_mean = initialise_state_space_mean(states,steady_state)
    state_space_variance = initialise_state_space_variance(states,steady_state)
    state_space = StateSpace(state_space_mean, state_space_variance)

    observation_transform = [0.0 1.0]

    # initialise distributions
    predicted_observation_distributions = Array{Normal{Real}}(undef,states.number_of_observations)
    predicted_observation_distributions[1] = distribution_prediction_at_given_time(state_space,
                                                                                   states,
                                                                                   observation_transform,
                                                                                   measurement_variance)


    # update the past ("negative time")
    current_observation = protein_at_observations[1,:]
    state_space = kalman_update_step(state_space,
                                     states,
                                     current_observation,
                                     time_delay,
                                     measurement_variance)

    return StateAndDistributions(state_space, predicted_observation_distributions)
end # function

"""
Perform the Kalman filter prediction about future observation, based on current knowledge i.e. current
state space mean and variance. This gives rho_{t+delta t-tau:t+delta t} and P_{t+delta t-tau:t+delta t},
using the differential equations in supplementary section 4 of Calderazzo et al., Bioinformatics (2018),
approximated using a forward Euler scheme.

# Arguments

- `state_space::StateSpace`: TODO

- `states::TimeConstructor`: TODO

- `current_observation::AbstractArray{<:Real}`: TODO

- `model_parameters::ModelParameters`: A ModelParameters object containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

# Returns

- `state_space::StateSpace`: TODO
"""
function kalman_prediction_step(
    state_space::StateSpace,
    states::TimeConstructor,
    current_observation::AbstractArray{<:Real},
    model_parameters::ModelParameters,
    )
    ## name the model parameters
    @unpack repression_threshold,
            hill_coefficient,
            mRNA_degradation_rate,
            protein_degradation_rate,
            basal_transcription_rate,
            translation_rate,
            time_delay = model_parameters

    # this is the number of states at t, i.e. before predicting towards t+observation_time_step
    current_number_of_states = calculate_current_number_of_states(current_observation[1]-states.number_of_hidden_states,
                                                                  states,time_delay)


    # we initialise all our matrices outside of the main for loop
    # this is P(t,t)
    current_covariance_matrix = Array{Float64}(undef,(2,2));#zeros((2,2))
    # this is P(t-tau,t) in page 5 of the supplementary material of Calderazzo et. al.
    covariance_matrix_past_to_now = Array{Float64}(undef,(2,2));#zeros((2,2))
    # this is P(t,t-tau) in page 5 of the supplementary material of Calderazzo et. al.
    covariance_matrix_now_to_past = Array{Float64}(undef,(2,2));#zeros((2,2))
    # This corresponds to P(s,t) in the Calderazzo paper
    covariance_matrix_intermediate_to_current = Array{Float64}(undef,(2,2));#zeros((2,2))
    # This corresponds to P(s,t-tau)
    covariance_matrix_intermediate_to_past = Array{Float64}(undef,(2,2));#zeros((2,2))
    # This corresponds to P(s,t+delta t)
    covariance_matrix_intermediate_to_next = Array{Float64}(undef,(2,2));#zeros((2,2))

    # derivations for the following are found in Calderazzo et. al. (2018)
    # g is [[-mRNA_degradation_rate,0],                  *[M(t),
    #       [translation_rate,-protein_degradation_rate]] [P(t)]
    # and its derivative will be called instant_jacobian
    # f is [[basal_transcription_rate*hill_function(past_protein)],0]
    # and its derivative with respect to the past state will be called delayed_jacobian
    # the matrix A in the paper will be called variance_of_noise
    instant_jacobian = [-mRNA_degradation_rate 0.0;
                        translation_rate -protein_degradation_rate]
    instant_jacobian_transpose = transpose(instant_jacobian)

    for next_time_index in (current_number_of_states + 1):(current_number_of_states + states.number_of_hidden_states)
        current_time_index = next_time_index - 1 # this corresponds to t
        past_time_index = current_time_index - states.discrete_delay # this corresponds to t-tau
        # indexing with 1:3 for numba
        current_mean = state_space.mean[current_time_index,[2,3]]
        past_protein = state_space.mean[past_time_index,3]

        hill_function_value = 1.0/(1.0+(past_protein/repression_threshold)^hill_coefficient)

        hill_function_derivative_value = -(hill_coefficient*(past_protein/repression_threshold)^(hill_coefficient - 1))/(
                                            repression_threshold*(1.0+(past_protein/repression_threshold)^hill_coefficient)^2)

        # jacobian of f is derivative of f with respect to past state ([past_mRNA, past_protein])
        delayed_jacobian = [0.0 basal_transcription_rate*hill_function_derivative_value;
                            0.0 0.0]
        delayed_jacobian_transpose = transpose(delayed_jacobian)

        ## derivative of mean is contributions from instant reactions + contributions from past reactions
        derivative_of_mean = ( [-mRNA_degradation_rate 0.0;
                                translation_rate -protein_degradation_rate]*current_mean +
                               [basal_transcription_rate*hill_function_value, 0])

        next_mean = current_mean .+ states.discretisation_time_step .* derivative_of_mean
        # ensures the prediction is non negative
        next_mean[next_mean.<0] .= 0
        # indexing with 1:3 for numba
        state_space.mean[next_time_index,[2,3]] .= next_mean
        # in the next lines we use for loop instead of np.ix_-like indexing for numba
        current_covariance_matrix = state_space.variance[[current_time_index,
                                                          states.total_number_of_states+current_time_index],
                                                          [current_time_index,
                                                           states.total_number_of_states+current_time_index]]

        # this is P(t-tau,t) in page 5 of the supplementary material of Calderazzo et. al
        covariance_matrix_past_to_now .= state_space.variance[[past_time_index,
                                                              states.total_number_of_states+past_time_index],
                                                              [current_time_index,
                                                               states.total_number_of_states+current_time_index]]

        # this is P(t,t-tau) in page 5 of the supplementary material of Calderazzo et. al.
        covariance_matrix_now_to_past .= state_space.variance[[current_time_index,
                                                              states.total_number_of_states+current_time_index],
                                                              [past_time_index,
                                                               states.total_number_of_states+past_time_index]]

        variance_change_current_contribution = ( instant_jacobian*current_covariance_matrix .+
                                                 current_covariance_matrix*instant_jacobian_transpose )

        variance_change_past_contribution = ( delayed_jacobian*covariance_matrix_past_to_now .+
                                              covariance_matrix_now_to_past*delayed_jacobian_transpose )

        variance_of_noise = [mRNA_degradation_rate*current_mean[1]+basal_transcription_rate*hill_function_value 0;
                             0 translation_rate*current_mean[1]+protein_degradation_rate*current_mean[2]]

        derivative_of_variance = ( variance_change_current_contribution .+
                                   variance_change_past_contribution .+
                                   variance_of_noise )

        # P(t+Deltat,t+Deltat)
        next_covariance_matrix = current_covariance_matrix .+ states.discretisation_time_step .* derivative_of_variance
        # ensure that the diagonal entries are non negative
        # this is a little annoying to do in Julia, but here we create a mask matrix with a 1 at any negative diagonal entries
        next_covariance_matrix[diagm(next_covariance_matrix[diagind(next_covariance_matrix)].<0)] .= 0

        state_space.variance[[next_time_index,
                              states.total_number_of_states+next_time_index],
                             [next_time_index,
                              states.total_number_of_states+next_time_index]] .= next_covariance_matrix

        ## now we need to update the cross correlations, P(s,t) in the Calderazzo paper
        # the range needs to include t, since we want to propagate P(t,t) into P(t,t+Deltat)
        for intermediate_time_index in past_time_index:current_time_index
            # This corresponds to P(s,t) in the Calderazzo paper
            covariance_matrix_intermediate_to_current .= state_space.variance[[intermediate_time_index,
                                                                              states.total_number_of_states+intermediate_time_index],
                                                                             [current_time_index,
                                                                              states.total_number_of_states+current_time_index]]
            # This corresponds to P(s,t-tau)
            covariance_matrix_intermediate_to_past .= state_space.variance[[intermediate_time_index,
                                                                           states.total_number_of_states+intermediate_time_index],
                                                                           [past_time_index,
                                                                            states.total_number_of_states+past_time_index]]


            covariance_derivative = ( covariance_matrix_intermediate_to_current*instant_jacobian_transpose .+
                                      covariance_matrix_intermediate_to_past*delayed_jacobian_transpose )

            # This corresponds to P(s,t+Deltat) in the Calderazzo paper
            covariance_matrix_intermediate_to_next .= covariance_matrix_intermediate_to_current .+ states.discretisation_time_step.*covariance_derivative

            # Fill in the big matrix
            state_space.variance[[intermediate_time_index,
                                  states.total_number_of_states+intermediate_time_index],
                                 [next_time_index,
                                  states.total_number_of_states+next_time_index]] .= covariance_matrix_intermediate_to_next
            # Fill in the big matrix with transpose arguments, i.e. P(t+Deltat, s) - works if initialised symmetrically
            state_space.variance[[next_time_index,
                                  states.total_number_of_states+next_time_index],
                                 [intermediate_time_index,
                                  states.total_number_of_states+intermediate_time_index]] .= transpose(covariance_matrix_intermediate_to_next)
        end # intermediate time index for
    end # for (next time index)
    return StateSpace(state_space.mean, state_space.variance)
end # function

# useful function layout from Ti

# function kalman_prediction_step(...)
#     get_parameters()...
#     initialize_stuff!()
#     for current_step = ...
#         grads = compute_gradients(...)
#         update!(variables, grads)
#     end
#     return results
# end

"""
Perform the Kalman filter update step on the predicted mean and variance, given a new observation.
This implements the equations at the beginning of page 4 in Calderazzo et al., Bioinformatics (2018).
This assumes that the observations are collected at fixed time intervals.

# Arguments

- `state_space::StateSpace`: TODO

- `states::TimeConstructor`: TODO

- `current_observation::AbstractArray{<:Real}`: TODO

- `time_delay::Real`: TODO

- `measurement_variance::Real`: TODO

# Returns

- `state_space::StateSpace`: TODO

"""
function kalman_update_step(
    state_space::StateSpace,
    states::TimeConstructor,
    current_observation::AbstractArray{<:Real},
    time_delay::Real,
    measurement_variance::Real
    )

    current_number_of_states = calculate_current_number_of_states(current_observation[1],states,time_delay)

    # predicted_state_space_mean until delay, corresponds to
    # rho(t+Deltat-delay:t+deltat). Includes current value and discrete_delay past values
    # funny indexing with 1:3 instead of (1,2) to make numba happy
    shortened_state_space_mean = state_space.mean[(current_number_of_states-states.discrete_delay):current_number_of_states,[2,3]]

    # put protein values underneath mRNA values, to make vector of means (rho)
    # consistent with variance (P)
    stacked_state_space_mean = vcat(shortened_state_space_mean[:,1],shortened_state_space_mean[:,2])

    # funny indexing with 1:3 instead of (1,2) to make numba happy
    predicted_final_state_space_mean = copy(state_space.mean[current_number_of_states,[2,3]])

    # extract covariance matrix up to delay
    # corresponds to P(t+Deltat-delay:t+deltat,t+Deltat-delay:t+deltat)
    mRNA_indices_to_keep = (current_number_of_states - states.discrete_delay):(current_number_of_states)
    protein_indices_to_keep = (states.total_number_of_states + current_number_of_states - states.discrete_delay):(states.total_number_of_states + current_number_of_states)
    all_indices_up_to_delay = vcat(mRNA_indices_to_keep, protein_indices_to_keep)

    # using for loop indexing for numba
    shortened_covariance_matrix = state_space.variance[all_indices_up_to_delay,all_indices_up_to_delay]
    # extract P(t+Deltat-delay:t+deltat,t+Deltat)
    shortened_covariance_matrix_past_to_final = shortened_covariance_matrix[:,[states.discrete_delay+1,end]]

    # and P(t+Deltat,t+Deltat-delay:t+deltat)
    shortened_covariance_matrix_final_to_past = shortened_covariance_matrix[[states.discrete_delay+1,end],:]

    # This is F in the paper
    observation_transform = [0.0 1.0]

    # This is P(t+Deltat,t+Deltat) in the paper
    predicted_final_covariance_matrix = state_space.variance[[current_number_of_states,
                                                              states.total_number_of_states+current_number_of_states],
                                                             [current_number_of_states,
                                                              states.total_number_of_states+current_number_of_states]]

    # This is (FP_{t+Deltat}F^T + Sigma_e)^-1
    helper_inverse = 1.0./(dot(observation_transform,predicted_final_covariance_matrix*transpose(observation_transform)) + measurement_variance)

    # This is C in the paper
    adaptation_coefficient = sum(dot.(shortened_covariance_matrix_past_to_final,observation_transform),dims=2).*helper_inverse

    # This is rho*
    updated_stacked_state_space_mean = ( stacked_state_space_mean .+
                                         (adaptation_coefficient*(current_observation[2] -
                                                                 dot(observation_transform,predicted_final_state_space_mean))) )
    # ensures the the mean mRNA and Protein are non negative
    updated_stacked_state_space_mean[updated_stacked_state_space_mean.<0] .= 0
    # unstack the rho into two columns, one with mRNA and one with protein
    updated_state_space_mean = hcat(updated_stacked_state_space_mean[1:(states.discrete_delay+1)],
                                    updated_stacked_state_space_mean[(states.discrete_delay+2):end])
    # Fill in the updated values
    # funny indexing with 1:3 instead of (1,2) to make numba happy
    state_space.mean[(current_number_of_states-states.discrete_delay):current_number_of_states,[2,3]] .= updated_state_space_mean

    # This is P*
    updated_shortened_covariance_matrix = ( shortened_covariance_matrix .-
                                            adaptation_coefficient*observation_transform*shortened_covariance_matrix_final_to_past )
    # ensure that the diagonal entries are non negative
    # np.fill_diagonal(updated_shortened_covariance_matrix,np.maximum(np.diag(updated_shortened_covariance_matrix),0))
    updated_shortened_covariance_matrix[diagm(updated_shortened_covariance_matrix[diagind(updated_shortened_covariance_matrix)].<0)] .= 0

    # Fill in updated values
    state_space.variance[all_indices_up_to_delay,all_indices_up_to_delay] .= updated_shortened_covariance_matrix

    return StateSpace(state_space.mean, state_space.variance)
end # function
