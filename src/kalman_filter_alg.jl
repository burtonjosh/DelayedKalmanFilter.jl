struct TimeConstructor{T<:Integer}
    number_of_observations::T
    observation_time_step::T
    discretisation_time_step::T
    discrete_delay::T
    initial_number_of_states::T
    number_of_hidden_states::T
    total_number_of_states::T
end

"""
Function which returns an instance of TimeConstructor, which is defines the numbers of
states for various uses in the Kalman filter.
"""
function time_constructor_function(
    protein_at_observations::Matrix{<:AbstractFloat},
    τ::AbstractFloat,
    discretisation_time_step::Integer = 1,
)

    number_of_observations = size(protein_at_observations, 1)
    observation_time_step =
        Int(protein_at_observations[2, 1] - protein_at_observations[1, 1])

    discrete_delay = ceil(Int,τ / discretisation_time_step)
    initial_number_of_states = discrete_delay + 1
    number_of_hidden_states = Int64(round(observation_time_step / discretisation_time_step))

    total_number_of_states =
        initial_number_of_states + (number_of_observations - 1) * number_of_hidden_states

    return TimeConstructor{Integer}(
        number_of_observations,
        observation_time_step,
        discretisation_time_step,
        discrete_delay,
        initial_number_of_states,
        number_of_hidden_states,
        total_number_of_states,
    )
end

"""
Helper function to return the state space mean for a given (real valued) time between t-τ and t+Δobs,
where t is the current number of states.
"""
function state_space_mean_indexer(
    state_space_mean,
    time,
    current_number_of_states,
    states
)
    array_index = length(state_space_mean) + min(0,floor(Int,(time-current_number_of_states)/states.observation_time_step))
    return state_space_mean[array_index](time)
end


"""
Calculate the mean and variance of the Normal approximation of the state space for a given
time point in the Kalman filtering algorithm and return it as a Normal distribution
"""
function distribution_prediction_at_given_time(
    state_space_mean,
    state_space_variance,
    continuous_state_space_variance,
    states::TimeConstructor,
    given_time::Integer,
    observation_transform::Matrix{<:AbstractFloat},
    measurement_variance::AbstractFloat,
)
    mean_prediction = dot(observation_transform,
        state_space_mean_indexer(
            state_space_mean,
            given_time-(states.discrete_delay+1),
            given_time-(states.discrete_delay+1),
            states)
    )
    
    last_predicted_covariance_matrix = state_space_variance[
        [given_time, states.total_number_of_states + given_time],
        [given_time, states.total_number_of_states + given_time],
    ]

    # println(last_predicted_covariance_matrix)
    # println(continuous_state_space_variance[1](
    #     given_time-(states.discrete_delay+1),
    #     given_time-(states.discrete_delay+1)
    #     )
    # )
    variance_prediction =
        dot(
            observation_transform,
            last_predicted_covariance_matrix * observation_transform',
        ) + measurement_variance

    return Normal(mean_prediction, sqrt(variance_prediction))
end

"""
Calculate the current number of states for a given observation time
"""
function calculate_current_number_of_states(
    current_observation_time::AbstractFloat,
    states::TimeConstructor,
    τ::AbstractFloat,
)
    return Int(round(current_observation_time / states.observation_time_step)) *
           states.number_of_hidden_states + states.initial_number_of_states
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
- `state_space_mean::Matrix{<:AbstractFloat}`: An N x 3 matrix, where N is the total number of states. The columns are time, mRNA
    and protein respectively.

- `state_space_variance::Matrix{<:AbstractFloat}`: An 2N x 2N matrix where N is the total number of states. It is constructed as a 2 x 2 block
    matrix, where the blocks give the covariance of (mRNA, mRNA), (mRNA, protein), (protein, mRNA), and (protein, protein) for all times (t,s) where
    abs(t -s) <= τ, the transcriptional time delay.

- `predicted_observation_distributions::Array{Normal{Float64}}`: An array of length N, whose entries are Normal distributions with mean and variance
    equal to the state space mean and variance predictions for the corresponding time point.

# Example
```jldoctest
julia> using DelayedKalmanFilter

julia> protein = [0. 105.; 10. 100.; 20. 98.]; # times are 0., 10., 20., and protein levels are 105., 100., and 98. respectively

julia> model_parameters = [100.0, 5.0, 0.1, 0.1, 1.0, 1.0, 15.0];

julia> measurement_variance = 1000.0;

julia> ss_mean, ss_var, _, distributions = kalman_filter(
           protein,
           model_parameters,
           measurement_variance
       );

julia> distributions[1]
Distributions.Normal{Float64}(μ=77.80895986786031, σ=93.7064351407417)
```
"""
function kalman_filter(
    protein_at_observations::Matrix{<:AbstractFloat},
    model_parameters::Vector{<:AbstractFloat},
    measurement_variance::AbstractFloat,
)

    τ = model_parameters[7]
    # F in the paper
    observation_transform = [0.0 1.0]
    states = time_constructor_function(protein_at_observations, τ)
    # initialise state space and distribution predictions
    state_space_mean, state_space_variance, continuous_state_space_variance, predicted_observation_distributions =
    kalman_filter_state_space_initialisation(
        protein_at_observations,
        model_parameters,
        states,
        observation_transform,
        measurement_variance,
    )

    # loop through observations and at each observation apply the Kalman prediction step and then the update step
    for observation_index = 2:size(protein_at_observations, 1)
        current_observation = protein_at_observations[observation_index, :]
        state_space_mean, state_space_variance, continuous_state_space_variance = kalman_prediction_step!(
            state_space_mean,
            state_space_variance,
            continuous_state_space_variance,
            states,
            current_observation,
            model_parameters,
        )
        current_number_of_states = calculate_current_number_of_states(
            current_observation[1],
            states,
            τ
        )
        # between the prediction and update steps we record the normal distributions for our likelihood
        predicted_observation_distributions[observation_index] =
            distribution_prediction_at_given_time(
                state_space_mean,
                state_space_variance,
                continuous_state_space_variance,
                states,
                current_number_of_states,
                observation_transform,
                measurement_variance,
            )
        state_space_mean, state_space_variance, continuous_state_space_variance = kalman_update_step!(
            state_space_mean,
            state_space_variance,
            continuous_state_space_variance,
            states,
            current_observation,
            τ,
            measurement_variance,
            observation_transform,
        )
    end # for
    return state_space_mean, state_space_variance, continuous_state_space_variance, predicted_observation_distributions
end # function

"""
Initialse the state space mean for a given set of time states and the ODE system's steady state
"""
function initialise_state_space_mean(states::TimeConstructor, steady_state,τ)

    function initial_continuous_mean_function(t)
        t <= 0.0 ? steady_state : [0.0, 0.0]
    end

    return fill(initial_continuous_mean_function,ceil(Int,τ/states.observation_time_step) + ceil(Int,states.observation_time_step/τ))
end

"""
Initialse the state space variance for a given set of time states and the ODE system's steady state
"""
function initialise_state_space_variance(
    states::TimeConstructor,
    steady_state,
    τ;
    mRNA_scaling::AbstractFloat = 20.0,
    protein_scaling::AbstractFloat = 100.0,
)

    state_space_variance = zeros(#Matrix{Float64}(undef,
        2*states.total_number_of_states,
        2*states.total_number_of_states,
    )

    diag_indices = diagind(state_space_variance)
    mRNA_indices = 1:states.initial_number_of_states
    protein_indices =
        1+states.total_number_of_states:1+states.total_number_of_states+states.initial_number_of_states

    state_space_variance[diag_indices[mRNA_indices]] .= steady_state[1] * mRNA_scaling
    state_space_variance[diag_indices[protein_indices]] .= steady_state[2] * protein_scaling

    function initial_continuous_variance_function(s,t)
        t <= 0 || s <= 0 ? [steady_state[1]*mRNA_scaling 0.0; 0.0 steady_state[2]*protein_scaling] : [0. 0. ; 0. 0.]
    end
    
    continuous_state_space_variance = fill(initial_continuous_variance_function,(3,ceil(Int,τ/states.observation_time_step) + ceil(Int,states.observation_time_step/τ)))

    return state_space_variance, continuous_state_space_variance
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

- `states::TimeConstructor`: A TimeConstructor whose fields define various values relevant to number of states, e.g. discrete_delay,
    total_number_of_states, etc.

- `observation_transform`: A 1 x 2 matrix corresponding to the transformation from observed data to molecule number, for mRNA and protein
    respectively.

- `measurement_variance::Real`: The variance in our measurement. This is given by Sigma epsilon in Calderazzo et. al. (2018).

# Returns
- `state_space_mean::Matrix{<:AbstractFloat}`: An N x 3 matrix, where N is the total number of states. The columns are time, mRNA
    and protein respectively.

- `state_space_variance::Matrix{<:AbstractFloat}`: An 2N x 2N matrix where N is the total number of states. It is constructed as a 2 x 2 block
    matrix, where the blocks give the covariance of (mRNA, mRNA), (mRNA, protein), (protein, mRNA), and (protein, protein) for all times (t,s) where
    abs(t -s) <= τ, the transcriptional time delay.

- `predicted_observation_distributions::Array{Normal{Float64}}`: An array of length n, whose entries are Normal distributions with mean and variance
    equal to the state space mean and variance predictions for the corresponding time point.
"""
function kalman_filter_state_space_initialisation(
    protein_at_observations::Matrix{<:AbstractFloat},
    model_parameters::Vector{<:AbstractFloat},
    states::TimeConstructor,
    observation_transform,
    measurement_variance::AbstractFloat = 10.0,
)

    τ = model_parameters[7]

    steady_state = calculate_steady_state_of_ode(model_parameters)

    # construct state space
    state_space_mean = initialise_state_space_mean(states, steady_state, τ)
    state_space_variance, continuous_state_space_variance = initialise_state_space_variance(states, steady_state, τ)

    # initialise distributions
    predicted_observation_distributions =
        fill(Normal(),states.number_of_observations)#Array{Normal{Float64}}(undef, states.number_of_observations)
    predicted_observation_distributions[1] = distribution_prediction_at_given_time(
        state_space_mean,
        state_space_variance,
        continuous_state_space_variance,
        states,
        states.initial_number_of_states,
        observation_transform,
        measurement_variance,
    )
    # update the past ("negative time")
    current_observation = protein_at_observations[1, :]
    state_space_mean, state_space_variance, continuous_state_space_variance = kalman_update_step!(
        state_space_mean,
        state_space_variance,
        continuous_state_space_variance,
        states,
        current_observation,
        τ,
        measurement_variance,
        observation_transform,
    )

    return state_space_mean, state_space_variance, continuous_state_space_variance, predicted_observation_distributions
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

"""
Predict state space mean to the next observation time index
"""
function predict_state_space_mean!(
    state_space_mean,
    current_number_of_states,
    model_parameters,
    states
    )

    initial_condition_times = [Float64(current_number_of_states)]# + states.discrete_delay*x for x in 0:ceil(states.number_of_hidden_states/states.discrete_delay)-1]

    for initial_condition_state in initial_condition_times
        
        function state_space_mean_RHS(du,u,h,p,t) # is there some way for this function to not be nested? does it matter?
            past_index = t - p[7]
            # history function in case τ < number_of_hidden_states
            if past_index >= tspan[1]
                past_protein = h(p,past_index;idxs=2)
            # otherwise pre defined indexer
            else
                past_protein = state_space_mean_indexer(state_space_mean, past_index,current_number_of_states-(states.discrete_delay+1)-states.number_of_hidden_states,states)[2]
            end
    
            du[1] = -p[3]*u[1] + p[5]*hill_function(past_protein, p[1], p[2])
            du[2] = p[6]*u[1] - p[4]*u[2]
        end
    
        h(p,t;idxs::Int) = 1.0
        # tspan = (initial_condition_state,min(initial_condition_state + states.discrete_delay,current_number_of_states+states.number_of_hidden_states)) .- (states.discrete_delay+1) # TODO get right times
        tspan = (initial_condition_state,current_number_of_states+states.number_of_hidden_states) .- (states.discrete_delay+1)
        mean_prob = DDEProblem(state_space_mean_RHS,
                            state_space_mean_indexer(state_space_mean,initial_condition_state-(states.discrete_delay+1),current_number_of_states-(states.discrete_delay+1),states),#current_mean
                            h,
                            tspan,
                            model_parameters; constant_lags=[model_parameters[7]])
        mean_solution = solve(mean_prob,MethodOfSteps(Euler()),dt=1.,adaptive=false,saveat=1.,dtmin=1.,dtmax=1.)
        state_space_mean = [state_space_mean[2:end]...,mean_solution]
    end

    return state_space_mean
end

function predict_variance_first_step!(
    initial_condition_state,
    current_number_of_states,
    off_diagonal_state_space_variance_RHS,
    state_space_variance,
    continuous_state_space_variance,
    states
)
    diag_tspan = (Float64(initial_condition_state), Float64(min(initial_condition_state+states.discrete_delay,current_number_of_states + states.number_of_hidden_states)))
    # solve first batch of off diagonals
    # we want to do P(s,t) -> P(s,t+nΔt) for s = t-τ:t
    for intermediate_time_index in max(initial_condition_state - states.discrete_delay, states.discrete_delay+1):initial_condition_state
        covariance_matrix_intermediate_to_current = state_space_variance[[intermediate_time_index,
                                                                            states.total_number_of_states+intermediate_time_index],
                                                                            [initial_condition_state,
                                                                            states.total_number_of_states+initial_condition_state]]

        off_diag_prob = ODEProblem(off_diagonal_state_space_variance_RHS,
                                    covariance_matrix_intermediate_to_current,# P(s,t)
                                    diag_tspan,
                                    intermediate_time_index)#model_parameters)
        off_diag_solution = solve(off_diag_prob,Euler(),dt=1.,adaptive=false,saveat=1.,dtmin=1.,dtmax=1.)

        # Fill in the big matrix
        for index in initial_condition_state+1:min(initial_condition_state+states.discrete_delay,current_number_of_states + states.number_of_hidden_states)
            if index - intermediate_time_index <= states.discrete_delay # this is hacky -- fix above to do less computations
            state_space_variance[[intermediate_time_index,
                                    states.total_number_of_states+intermediate_time_index],
                                    [index,
                                    states.total_number_of_states+index]] = off_diag_solution(index)

            state_space_variance[[index,
                                    states.total_number_of_states+index],
                                    [intermediate_time_index,
                                    states.total_number_of_states+intermediate_time_index]] = off_diag_solution(index)'
            end
        end # fill in matrix for
    end # intermediate time index for
end

function predict_variance_second_step!(
    initial_condition_state,
    current_number_of_states,
    state_space_variance_RHS,
    state_space_variance,
    continuous_state_space_variance,
    model_parameters,
    states
)
    initial_covariance = state_space_variance[[initial_condition_state,
                                               states.total_number_of_states+initial_condition_state],
                                              [initial_condition_state,
                                               states.total_number_of_states+initial_condition_state]]
    tspan = (Float64(initial_condition_state),Float64(min(initial_condition_state+states.discrete_delay,current_number_of_states + states.number_of_hidden_states)))
    diagonal_variance_prob = ODEProblem(state_space_variance_RHS,
                                        initial_covariance,
                                        tspan,
                                        model_parameters)
    diagonal_variance_solution = solve(diagonal_variance_prob,Euler(),dt=1.,adaptive=false,saveat=1.,dtmin=1.,dtmax=1.)
    for index in initial_condition_state+1:min(initial_condition_state+states.discrete_delay,current_number_of_states + states.number_of_hidden_states)
        state_space_variance[[index,
                                states.total_number_of_states+index],
                                [index,
                                states.total_number_of_states+index]] = diagonal_variance_solution(index)
    end
end

function predict_variance_third_step!(
    initial_condition_state,
    current_number_of_states,
    off_diagonal_state_space_variance_RHS,
    state_space_variance,
    continuous_state_space_variance,
    states
)
    for intermediate_time_index in initial_condition_state+1:min(initial_condition_state+states.discrete_delay,current_number_of_states + states.number_of_hidden_states)-1
        covariance_matrix_intermediate = state_space_variance[[intermediate_time_index,
                                                                states.total_number_of_states+intermediate_time_index],
                                                                [intermediate_time_index,
                                                                states.total_number_of_states+intermediate_time_index]]

        diag_tspan = (Float64(intermediate_time_index),Float64(min(initial_condition_state+states.discrete_delay,current_number_of_states + states.number_of_hidden_states)))
        off_diag_prob = ODEProblem(off_diagonal_state_space_variance_RHS,
                                    covariance_matrix_intermediate,# P(s,s)
                                    diag_tspan,
                                    intermediate_time_index)#model_parameters)
        off_diag_solution = solve(off_diag_prob,Euler(),dt=1.,adaptive=false,saveat=1.,dtmin=1.,dtmax=1.)

        # Fill in the big matrix
        for index in intermediate_time_index+1:min(initial_condition_state+states.discrete_delay,current_number_of_states + states.number_of_hidden_states)
            state_space_variance[[intermediate_time_index,
                                    states.total_number_of_states+intermediate_time_index],
                                    [index,
                                    states.total_number_of_states+index]] = off_diag_solution(index)
            state_space_variance[[index,
                                    states.total_number_of_states+index],
                                    [intermediate_time_index,
                                    states.total_number_of_states+intermediate_time_index]] = off_diag_solution(index)'
        end # fill in matrix for
    end # intermediate time index for
end

"""
Predict state space variance to the next observation time index.
There are three separate steps in the variance prediction:
    (1) integrate P(t,s) to P(t+nΔt,s) for s = t-τ:t, where nΔt is the number of hidden
        states. (horizontal/vertical)
    (2) Integrate P(t,t) to P(t+nΔt,t+nΔt). (diagonal)
    (3) Integrate P(t,t) to P(t,t+nΔt) for t in t:t+nΔt-1. (horizontal/vertical)
"""
function predict_state_space_variance!(
    state_space_mean,
    state_space_variance,
    continuous_state_space_variance,
    current_number_of_states,
    model_parameters,
    states,
    instant_jacobian
    )
    
    function state_space_variance_RHS(du,u,p,t)
        # past_protein = h(p,t-p[7])
        past_index = Int(t) - states.discrete_delay
        past_protein = state_space_mean_indexer(state_space_mean, past_index-(states.discrete_delay+1),current_number_of_states-(states.discrete_delay+1),states)[2]
        current_mean = state_space_mean_indexer(state_space_mean, t-(states.discrete_delay+1),current_number_of_states-(states.discrete_delay+1),states)

        past_to_now_diagonal_variance = state_space_variance[[past_index,states.total_number_of_states+past_index],
                                                             [Int(t),states.total_number_of_states+Int(t)]]
        delayed_jacobian = construct_delayed_jacobian(model_parameters, past_protein)

        variance_of_noise = [p[3]*current_mean[1]+p[5]*hill_function(past_protein, p[1], p[2]) 0.0;
                             0.0 p[6]*current_mean[1]+p[4]*current_mean[2]]

        du .= instant_jacobian*u + u*instant_jacobian' +
             delayed_jacobian*past_to_now_diagonal_variance +
             past_to_now_diagonal_variance'*delayed_jacobian' +
             variance_of_noise
    end # function

    function off_diagonal_state_space_variance_RHS(du,u,p,diag_t) # p is the intermediate time index s
        past_index = Int(diag_t) - states.discrete_delay # t - τ
        if past_index < 1
            past_protein = 0.0
        else
            past_protein = state_space_mean_indexer(state_space_mean, past_index-(states.discrete_delay+1),current_number_of_states-(states.discrete_delay+1),states)[2]
        end

        delayed_jacobian = construct_delayed_jacobian(model_parameters, past_protein)

        if diag_t < 1 || past_index < 1
            covariance_matrix_intermediate_to_past = zeros(2,2)
        else
            covariance_matrix_intermediate_to_past = state_space_variance[[p,
                                                                           states.total_number_of_states+p],
                                                                          [past_index,
                                                                           states.total_number_of_states+past_index]]
       end
        du .= u*instant_jacobian' +
             covariance_matrix_intermediate_to_past*delayed_jacobian'

    end # function

    # in the case of τ < observation_time_step, we have to do the below procedure multiple times (ceil(observation/τ) times)
    # since otherwise certain P(t-τ,s) values will not exist #TODO bad explanation
    initial_condition_times = Int.([current_number_of_states + states.discrete_delay*x for x in 0:ceil(states.number_of_hidden_states/states.discrete_delay)-1])
    for initial_condition_state in initial_condition_times
        # (1) Integrate P(t,s) to P(t+nΔt,s) for s = t-τ:t.
        predict_variance_first_step!(
            initial_condition_state,
            current_number_of_states,
            off_diagonal_state_space_variance_RHS,
            state_space_variance,
            continuous_state_space_variance,
            states
        )
        # (2) Integrate P(t,t) to P(t+nΔt,t+nΔt).
        predict_variance_second_step!(
            initial_condition_state,
            current_number_of_states,
            state_space_variance_RHS,
            state_space_variance,
            continuous_state_space_variance,
            model_parameters,
            states
        )
        # (3) Integrate P(t,t) to P(t,t+nΔt) for t in t:t+nΔt-1.
        predict_variance_third_step!(
            initial_condition_state,
            current_number_of_states,
            off_diagonal_state_space_variance_RHS,
            state_space_variance,
            continuous_state_space_variance,
            states
        )
    end # initial_condition_times for
end

"""
Perform the Kalman filter prediction about future observation, based on current knowledge i.e. current
state space mean and variance. This gives rho_{t+delta t-tau:t+delta t} and P_{t+delta t-tau:t+delta t},
using the differential equations in supplementary section 4 of Calderazzo et al., Bioinformatics (2018),
approximated using a forward Euler scheme.

# Arguments

- `state_space_mean::Matrix{<:AbstractFloat}`: An N x 3 matrix, where N is the total number of states. The columns are time, mRNA
    and protein respectively.

- `state_space_variance::Matrix{<:AbstractFloat}`: An 2N x 2N matrix where N is the total number of states. It is constructed as a 2 x 2 block
    matrix, where the blocks give the covariance of (mRNA, mRNA), (mRNA, protein), (protein, mRNA), and (protein, protein) for all times (t,s) where
    abs(t -s) <= τ, the transcriptional time delay.

- `states::TimeConstructor`: A TimeConstructor whose fields define various values relevant to number of states, e.g. discrete_delay,
    total_number_of_states, etc.

- `current_observation::Vector{<:AbstractFloat}`: The current time point and protein observation which acts as the initial condition for the
    prediction.

- `model_parameters::ModelParameters`: A ModelParameters object containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate,
    translation rate, time delay.

# Returns
- `state_space_mean::Matrix{<:AbstractFloat}`: An N x 3 matrix, where N is the total number of states. The columns are time, mRNA
    and protein respectively. With each prediction new values are saved to the relevant entries in the matrix.

- `state_space_variance::Matrix{<:AbstractFloat}`: An 2N x 2N matrix where N is the total number of states. It is constructed as a 2 x 2 block
    matrix, where the blocks give the covariance of (mRNA, mRNA), (mRNA, protein), (protein, mRNA), and (protein, protein) for all times (t,s) where
    abs(t -s) <= τ, the transcriptional time delay. With each prediction new values are saved to the relevant entries in the matrix.
"""
function kalman_prediction_step!(
    state_space_mean,
    state_space_variance,
    continuous_state_space_variance,
    states::TimeConstructor,
    current_observation::Vector{<:AbstractFloat},
    model_parameters::Vector{<:AbstractFloat},
)

    τ = model_parameters[7]

    # @unpack P₀, h, μₘ, μₚ, αₘ, αₚ, τ = model_parameters
    # this is the number of states at t, i.e. before predicting towards t+observation_time_step
    current_number_of_states = calculate_current_number_of_states(
        current_observation[1] - states.number_of_hidden_states,
        states,
        τ,
    )

    instant_jacobian = construct_instant_jacobian(model_parameters)

    state_space_mean = predict_state_space_mean!(
        state_space_mean,
        current_number_of_states,
        model_parameters,
        states
        )

    predict_state_space_variance!(
        state_space_mean,
        state_space_variance,
        continuous_state_space_variance,
        current_number_of_states,
        model_parameters,
        states,
        instant_jacobian
        )
    return state_space_mean, state_space_variance, continuous_state_space_variance
end # function

function update_mean!(
    state_space_mean,
    current_number_of_states,
    states,
    adaptation_coefficient,
    current_observation,
    observation_transform,
)

    predicted_final_state_space_mean = state_space_mean_indexer(
        state_space_mean,
        current_number_of_states-(states.discrete_delay+1),
        current_number_of_states-(states.discrete_delay+1),
        states
    )

    adaptation_coefficient_mRNA = LinearInterpolation(
        current_observation[1]-(states.discrete_delay):current_observation[1],
        adaptation_coefficient[1:states.discrete_delay+1]
        )
    adaptation_coefficient_protein = LinearInterpolation(
        current_observation[1]-(states.discrete_delay):current_observation[1],
        adaptation_coefficient[states.discrete_delay+2:end]
        )

    function update_addition(t)
        [adaptation_coefficient_mRNA(t),
         adaptation_coefficient_protein(t)] * ( current_observation[2] -
         dot(observation_transform, predicted_final_state_space_mean) )
    end

    updated_state_space_mean = state_space_mean .+ update_addition
    
    return updated_state_space_mean
end

function calculate_helper_inverse(
    observation_transform,
    predicted_final_covariance_matrix,
    measurement_variance,
)

    1.0 / (
        dot(
            observation_transform,
            predicted_final_covariance_matrix * observation_transform',
        ) + measurement_variance
    )

end

function calculate_adaptation_coefficient(cov_matrix, observation_transform, helper_inverse)
    sum(dot.(cov_matrix, observation_transform), dims = 2) * helper_inverse
end

function adaptation_coefficient_function(
    continuous_state_space_variance,
    index,
    t,
    s,
    observation_transform,
    helper_inverse
)
    continuous_state_space_variance[index](t,s)*observation_transform'*helper_inverse
end

function calculate_shortened_covariance_matrix(
    state_space_variance,
    current_number_of_states,
    states::TimeConstructor,
)

    mRNA_indices_to_keep =
        (current_number_of_states-states.discrete_delay):(current_number_of_states)
    protein_indices_to_keep =
        (states.total_number_of_states+current_number_of_states-states.discrete_delay):(states.total_number_of_states+current_number_of_states)
    all_indices_up_to_delay = vcat(mRNA_indices_to_keep, protein_indices_to_keep)

    return state_space_variance[all_indices_up_to_delay, all_indices_up_to_delay],
    all_indices_up_to_delay
end

"""
Return the covariance matrix at the current (last predicted) number of states
"""
function calculate_final_covariance_matrix(
    state_space_variance,
    continuous_state_space_variance,
    current_number_of_states,
    states,
)
    return state_space_variance[
        [
            current_number_of_states,
            states.total_number_of_states + current_number_of_states,
        ],
        [
            current_number_of_states,
            states.total_number_of_states + current_number_of_states,
        ],
    ], continuous_state_space_variance[1](
        current_number_of_states-(states.discrete_delay+1),
        current_number_of_states-(states.discrete_delay+1))
end

function update_variance!(
    state_space_variance,
    continuous_state_space_variance,
    shortened_covariance_matrix,
    all_indices_up_to_delay,
    states,
    adaptation_coefficient,
    observation_transform,
    helper_inverse
)

    updated_shortened_covariance_matrix = (
        shortened_covariance_matrix -
        adaptation_coefficient *
        observation_transform *
        shortened_covariance_matrix[[states.discrete_delay + 1, end], :]
    )

    idx = diagind(updated_shortened_covariance_matrix)
    updated_shortened_covariance_matrix[idx] = max.(updated_shortened_covariance_matrix[idx], 0.0)

    # Fill in updated values
    state_space_variance[all_indices_up_to_delay, all_indices_up_to_delay] =
        updated_shortened_covariance_matrix

    #### continuous variance part 
    function variance_update_addition(t,s)
        # return negative part 
        -adaptation_coefficient_function(
            continuous_state_space_variance,
            1,
            t,
            s,
            observation_transform,
            helper_inverse
        )*observation_transform*shortened_covariance_matrix[[states.discrete_delay + 1, end], :] # TODO need to fix this dimension mismatch
        # the problem is with the index part, don't want to have to specify that - can it be made contextual?
    end

    updated_continuous_state_space_variance = continuous_state_space_variance .+ variance_update_addition


    # function update_addition(t)
    #     [adaptation_coefficient_mRNA(t),
    #      adaptation_coefficient_protein(t)] * ( current_observation[2] -
    #      dot(observation_transform, predicted_final_state_space_mean) )
    # end

    # updated_state_space_mean = state_space_mean .+ update_addition



    return state_space_variance, updated_continuous_state_space_variance
end

"""
Perform the Kalman filter update step on the predicted mean and variance, given a new observation.
This implements the equations at the beginning of page 4 in Calderazzo et al., Bioinformatics (2018).
This assumes that the observations are collected at fixed time intervals.

# Arguments

- `state_space_mean::Matrix{<:AbstractFloat}`: An N x 3 matrix, where N is the total number of states. The columns are time, mRNA
    and protein respectively.

- `state_space_variance::Matrix{<:AbstractFloat}`: An 2N x 2N matrix where N is the total number of states. It is constructed as a 2 x 2 block
    matrix, where the blocks give the covariance of (mRNA, mRNA), (mRNA, protein), (protein, mRNA), and (protein, protein) for all times (t,s) where
    abs(t -s) <= τ, the transcriptional time delay.

- `states::TimeConstructor`: A TimeConstructor whose fields define various values relevant to number of states, e.g. discrete_delay,
    total_number_of_states, etc.

- `current_observation::Vector{<:AbstractFloat}`: The current time point and protein observation which acts as the initial condition for the
    prediction.

- `τ::Real`: The time delay parameter, model_parameters[7].

- `measurement_variance::Real`: The variance which defines the measurement error, it is ``Σ_ϵ`` in the equation ``y = Fx + Σ_ϵ``.

# Returns

- `state_space_mean::Matrix{<:AbstractFloat}`: An N x 3 matrix, where N is the total number of states. The columns are time, mRNA
    and protein respectively. At each update step the relevant entries in the matrix are updated according to the update defined in
    Calderazzo et al., Bioinformatics (2018).

- `state_space_variance::Matrix{<:AbstractFloat}`: An 2N x 2N matrix where N is the total number of states. It is constructed as a 2 x 2 block
    matrix, where the blocks give the covariance of (mRNA, mRNA), (mRNA, protein), (protein, mRNA), and (protein, protein) for all times (t,s) where
    abs(t -s) <= τ, the transcriptional time delay. At each update step the relevant entries in the matrix are updated according to the update defined in
    Calderazzo et al., Bioinformatics (2018).

"""
function kalman_update_step!(
    state_space_mean,
    state_space_variance,
    continuous_state_space_variance,
    states::TimeConstructor,
    current_observation::Vector{<:AbstractFloat},
    τ::AbstractFloat,
    measurement_variance::AbstractFloat,
    observation_transform,
)

    current_number_of_states =
        calculate_current_number_of_states(current_observation[1], states, τ)

    # extract covariance matrix up to delay and indices
    # corresponds to P(t+Deltat-delay:t+deltat,t+Deltat-delay:t+deltat)
    shortened_covariance_matrix, all_indices_up_to_delay =
        calculate_shortened_covariance_matrix(
            state_space_variance,
            current_number_of_states,
            states,
        )

    # This is P(t+Deltat,t+Deltat) in the paper
    predicted_final_covariance_matrix, predicted_final_continuous_covariance_matrix = calculate_final_covariance_matrix(
        state_space_variance,
        continuous_state_space_variance,
        current_number_of_states,
        states,
    )

    # println("final disc ", predicted_final_covariance_matrix)
    # println("final cont ", predicted_final_continuous_covariance_matrix)

    # This is (FP_{t+Deltat}F^T + Sigma_e)^-1
    helper_inverse = calculate_helper_inverse(
        observation_transform,
        predicted_final_covariance_matrix,
        measurement_variance,
    )

    # helper_inverse_continuous = calculate_helper_inverse(
    #     observation_transform,
    #     predicted_final_continuous_covariance_matrix,
    #     measurement_variance,
    # )

    # This is C in the paper
    adaptation_coefficient = calculate_adaptation_coefficient(
        shortened_covariance_matrix[:, [states.discrete_delay + 1, end]],
        observation_transform,
        helper_inverse,
    )

    println("adap disc, ",adaptation_coefficient[[29,end]])
    println("adap cont, ",adaptation_coefficient_function(
    continuous_state_space_variance,
    1,
    0,
    0,
    observation_transform,
    helper_inverse
))

    # this is ρ*
    state_space_mean = update_mean!(
        state_space_mean,
        current_number_of_states,
        states,
        adaptation_coefficient,
        current_observation,
        observation_transform,
    )
    # This is P*
    state_space_variance, continuous_state_space_variance = update_variance!(
        state_space_variance,
        continuous_state_space_variance,
        shortened_covariance_matrix,
        all_indices_up_to_delay,
        states,
        adaptation_coefficient,
        observation_transform,
        helper_inverse
    )

    return state_space_mean, state_space_variance, continuous_state_space_variance
end # function
