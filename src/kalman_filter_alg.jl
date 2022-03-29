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
    mean
    variance
end

mutable struct StateAndDistributions
    state_space
    distributions
end

struct TimeConstructor
    number_of_observations
    observation_time_step
    discretisation_time_step
    discrete_delay
    number_of_hidden_states
    total_number_of_states
end

"""
Calculate the mean and variance of the Normal approximation of the state space for a given
time point in the Kalman filtering algorithm
"""
function distribution_prediction_at_given_time(
    state_space::StateSpace,
    given_time::Int,
    total_number_of_states::Int,
    observation_transform
    )

    mean_prediction = dot(observation_transform,state_space.mean[given_time,2:3])
    last_predicted_covariance_matrix = state_space.variance[[given_time,total_number_of_states+given_time],
                                                            [given_time,total_number_of_states+given_time]]
    variance_prediction = dot(observation_transform,last_predicted_covariance_matrix*transpose(observation_transform)) + measurement_variance

    return Normal(mean_prediction,sqrt(variance_prediction))
end

"""
Function which returns an instance of TimeConstructor, which is defines the numbers of
states for various uses in the Kalman filter.
"""
function TimeConstructorFunction(
    protein_at_observations,
    time_delay,
    discretisation_time_step = 1.0
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
Perform a delay-adjusted non-linear stochastic Kalman filter based on observation of protein
copy numbers. This implements the filter described by Calderazzo et al., Bioinformatics (2018).

# Arguments

- `protein_at_observations::Array{Float64,2}`: Observed protein. The dimension is n x 2, where n is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number at
    that time. The filter assumes that observations are generated with a fixed, regular time interval.

- `model_parameters::Array{Float64,1}`: An array containing the model parameters in the following order:
    repression threshold, hill coefficient, mRNA degradation rate,protein degradation rate, basal transcription rate, translation rate,
    transcription delay.

- `measurement_variance::Float64`: The variance in our measurement. This is given by Sigma epsilon in Calderazzo et. al. (2018).

# Returns

- `state_space_mean::Array{Float64,2}`: An array of dimension n x 3, where n is the number of inferred time points.
    The first column is time, the second column is the mean mRNA, and the third
    column is the mean protein. Time points are generated every minute

- `state_space_variance::Array{Float64,2}`: An array of dimension 2n x 2n.
          [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
            cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

- `state_space_mean_derivative::Array{Float64,3}`: An array of dimension n x m x 2, where n is the number of inferred time points,
    and m is the number of parameters. The m columns in the second dimension are the
    derivative of the state mean with respect to each parameter. The two elements in
    the third dimension represent the derivative of mRNA and protein respectively

- `state_space_variance_derivative::Array{Float64,3}`: An array of dimension 7 x 2n x 2n.
          [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/dtheta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/dtheta,
            d[cov( mRNA(t0:tn),protein(t0:tn) )/]dtheta, d[cov( protein(t0:tn),protein(t0:tn) )]/dtheta ]

- `predicted_observation_distributions::Array{Float64,2}`: An array of dimension n x 3 where n is the number of observation time points.
    The first column is time, the second and third columns are the mean and variance
    of the distribution of the expected observations at each time point, respectively.

- `predicted_observation_mean_derivatives::Array{Float64,3}`: An array of dimension n x m x 2, where n is the number of observation time points,
    and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
    space mean at each observation time point, wrt each parameter

- `predicted_observation_variance_derivatives::Array{Float64,4}`: An array of dimension n x m x 2 x 2, where n is the number of observation time points,
    and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
    space variance at each observation time point, wrt each parameter
"""
function kalman_filter(
    protein_at_observations,
    model_parameters::ModelParameters,
    measurement_variance::Real = 10
    )

    time_delay = model_parameters.time_delay

    number_of_states = TimeConstructor(protein_at_observations,time_delay)
    ## this may not be needed
    number_of_observations = number_of_states.number_of_observations
    observation_time_step = number_of_states.observation_time_step
    discretisation_time_step = number_of_states.discretisation_time_step
    discrete_delay = number_of_states.discrete_delay
    initial_number_of_states = number_of_states.initial_number_of_states
    number_of_hidden_states = number_of_states.number_of_hidden_states
    total_number_of_states = number_of_states.total_number_of_states

    state_space_and_distributions = kalman_filter_state_space_initialisation(protein_at_observations,
                                                                             model_parameters,
                                                                             measurement_variance)
    # loop through observations and at each observation apply the Kalman prediction step and then the update step
    # for observation_index, current_observation in enumerate(protein_at_observations[1:]):
    for observation_index in 1:(size(protein_at_observations,1)-1)

        current_observation = protein_at_observations[1 + observation_index,:]
        state_space = kalman_prediction_step(state_space_and_distributions.state_space,
                                             current_observation,
                                             model_parameters,
                                             observation_time_step)

        current_number_of_states = Int64(round(current_observation[1]/observation_time_step))*number_of_hidden_states + initial_number_of_states

    # between the prediction and update steps we record the mean and sd for our likelihood, and the derivatives of the mean and variance for the
    # derivative of the likelihood wrt the parameters

        state_space_and_distributions.distributions[observation_index+1] = distribution_prediction_at_given_time(state_space,
                                                                                                                 current_number_of_states,
                                                                                                                 total_number_of_states,
                                                                                                                 observation_transform)


        state_space = kalman_update_step(state_space,
                                         current_observation,
                                         time_delay,
                                         observation_time_step,
                                         measurement_variance)
    end # for
    return state_space_and_distributions
end # function

"""
    A function for initialisation of the state space mean and variance, and update for the "negative" times that
     are a result of the time delay. Initialises the negative times using the steady state of the deterministic system,
     and then updates them with kalman_update_step.

    Parameters
    ----------

    protein_at_observations : numpy array.
        Observed protein. The dimension is n x 2, where n is the number of observation time points.
        The first column is the time, and the second column is the observed protein copy number at
        that time. The filter assumes that observations are generated with a fixed, regular time interval.

    model_parameters : numpy array.
        An array containing the model parameters in the following order:
        repression_threshold, hill_coefficient, mRNA_degradation_rate,
        protein_degradation_rate, basal_transcription_rate, translation_rate,
        transcription_delay.

    measurement_variance : float.
        The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

    Returns
    -------

    state_space_mean : numpy array.
        An array of dimension n x 3, where n is the number of inferred time points.
        The first column is time, the second column is the mean mRNA, and the third
        column is the mean protein. Time points are generated every minute

    state_space_variance : numpy array.
        An array of dimension 2n x 2n.
              [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
                cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

    state_space_mean_derivative : numpy array.
        An array of dimension n x m x 2, where n is the number of inferred time points,
        and m is the number of parameters. The m columns in the second dimension are the
        derivative of the state mean with respect to each parameter. The two elements in
        the third dimension represent the derivative of mRNA and protein respectively

    state_space_variance_derivative : numpy array.
        An array of dimension 7 x 2n x 2n.
              [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
                d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

    predicted_observation_distributions : numpy array.
        An array of dimension n x 3 where n is the number of observation time points.
        The first column is time, the second and third columns are the mean and variance
        of the distribution of the expected observations at each time point, respectively

    predicted_observation_mean_derivatives : numpy array.
        An array of dimension n x m x 2, where n is the number of observation time points,
        and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
        space mean at each observation time point, wrt each parameter

    predicted_observation_variance_derivatives : numpy array.
        An array of dimension n x m x 2 x 2, where n is the number of observation time points,
        and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
        space variance at each observation time point, wrt each parameter
    """
function kalman_filter_state_space_initialisation(protein_at_observations,model_parameters::ModelParameters,measurement_variance::Real = 10.0)
    time_delay = model_parameters.time_delay

    number_of_states = TimeConstructor(protein_at_observations,time_delay) # hidden discretisation_time_step in this function
    ## this may not be needed
    number_of_observations = number_of_states.number_of_observations
    observation_time_step = number_of_states.observation_time_step
    discretisation_time_step = number_of_states.discretisation_time_step
    discrete_delay = number_of_states.discrete_delay
    initial_number_of_states = number_of_states.initial_number_of_states
    number_of_hidden_states = number_of_states.number_of_hidden_states
    total_number_of_states = number_of_states.total_number_of_states

    steady_state = calculate_steady_state_of_ode(model_parameters)

    # initialise state space mean -- new function
    state_space_mean = Array{Float64}(undef,(total_number_of_states,3));#zeros((total_number_of_states,3))
    state_space_mean[1:initial_number_of_states,2] .= steady_state[1]
    state_space_mean[1:initial_number_of_states,3] .= steady_state[2]

    final_observation_time = protein_at_observations[end,1]
    # assign time entries
    state_space_mean[:,1] .= LinRange(protein_at_observations[1,1]-discrete_delay,final_observation_time,total_number_of_states)

    # initialise initial covariance matrix -- new function
    state_space_variance = zeros((2*(total_number_of_states),2*(total_number_of_states)));

    # set the mRNA and protein variance at negative times to the LNA approximation
    # scaling factors for mRNA and protein respectively. For example, observation might be fluorescence,
    # so the scaling would correspond to how light intensity relates to molecule number.
    initial_mRNA_scaling = 20.0
    initial_protein_scaling = 100.0
    initial_mRNA_variance = state_space_mean[1,2]*initial_mRNA_scaling
    initial_protein_variance = state_space_mean[1,3]*initial_protein_scaling
    # diagm(diagind(A)[1 .<= diagind(A).< initial_number_of_states])
    for diag_index in 1:initial_number_of_states
        state_space_variance[diag_index,diag_index] = initial_mRNA_variance
        state_space_variance[diag_index + total_number_of_states,diag_index + total_number_of_states] = initial_protein_variance
    end #for

    # construct state space
    state_space = StateSpace(state_space_mean, state_space_variance)

    observation_transform = [0.0 1.0]

    # initialise distributions
    predicted_observation_distributions = Array{Float64}(undef,number_of_observations);#zeros(number_of_observations,3)
    predicted_observation_distributions[1] = distribution_prediction_at_given_time(state_space,
                                                                                   initial_number_of_states,
                                                                                   total_number_of_states,
                                                                                   observation_transform)


    # update the past ("negative time")
    current_observation = protein_at_observations[1,:]

    state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_update_step(state_space,
                                                                                                                              current_observation,
                                                                                                                              time_delay,
                                                                                                                              observation_time_step,
                                                                                                                              measurement_variance)

    return StateAndDistributions(state_space, predicted_observation_distributions)
end # function

"""
A function which updates the mean and variance for the distributions which describe the likelihood of
our observations, given some model parameters.

Parameters
----------

predicted_observation_distributions : numpy array.
    An array of dimension n x 3 where n is the number of observation time points.
    The first column is time, the second and third columns are the mean and variance
    of the distribution of the expected observations at each time point, respectively

current_observation : int.
    Observed protein at the current time. The dimension is 1 x 2.
    The first column is the time, and the second column is the observed protein copy number at
    that time

state_space_mean : numpy array
    An array of dimension n x 3, where n is the number of inferred time points.
    The first column is time, the second column is the mean mRNA, and the third
    column is the mean protein. Time points are generated every minute

state_space_variance : numpy array.
    An array of dimension 2n x 2n.
          [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
            cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

current_number_of_states : float.
    The current number of (hidden and observed) states upto the current observation time point.
    This includes the initial states (with negative time).

total_number_of_states : float.
    The total number of states that will be predicted by the kalman_filter function

measurement_variance : float.
    The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

observation_index : int.
    The index for the current observation time in the main kalman_filter loop

Returns
-------

predicted_observation_distributions[observation_index + 1] : numpy array.
    An array of dimension 1 x 3.
    The first column is time, the second and third columns are the mean and variance
    of the distribution of the expected observations at the current time point, respectively.
"""
function kalman_observation_distribution_parameters(predicted_observation_distributions,
                                                    current_observation,
                                                    state_space_mean,
                                                    state_space_variance,
                                                    current_number_of_states,
                                                    total_number_of_states,
                                                    measurement_variance,
                                                    observation_index)

    observation_transform = [0.0 1.0]

    predicted_observation_distributions[observation_index+1,1] = current_observation[1]
    predicted_observation_distributions[observation_index+1,2] = dot(observation_transform,state_space_mean[current_number_of_states,[2,3]])

    last_predicted_covariance_matrix = state_space_variance[[current_number_of_states,
                                                             total_number_of_states+current_number_of_states],
                                                            [current_number_of_states,
                                                             total_number_of_states+current_number_of_states]]

    predicted_observation_distributions[observation_index+1,3] = dot(observation_transform,last_predicted_covariance_matrix*transpose(observation_transform)) .+ measurement_variance

    return predicted_observation_distributions[observation_index+1]
end # function

"""

Parameters
----------

predicted_observation_mean_derivatives : numpy array.
    An array of dimension n x m x 2, where n is the number of observation time points,
    and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
    space mean at each observation time point, wrt each parameter

predicted_observation_variance_derivatives : numpy array.
    An array of dimension n x m x 2 x 2, where n is the number of observation time points,
    and m is the number of parameters. This gives the (non-updated) predicted derivative of the state
    space variance at each observation time point, wrt each parameter
    return state_space_mean,
current_observation : numpy array.
    A 1 x 2 array which describes the observation of protein at the current time point. The first
    column is time, and the second column is the protein level

state_space_mean_derivative : numpy array.
    An array of dimension n x m x 2, where n is the number of inferred time points,
    and m is the number of parameters. The m columns in the second dimension are the
    derivative of the state mean with respect to each parameter. The two elements in
    the third dimension represent the derivative of mRNA and protein respectively

state_space_variance_derivative : numpy array.
    An array of dimension 7 x 2n x 2n.
          [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
            d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

current_number_of_states : float.
    The current number of (hidden and observed) states upto the current observation time point.
    This includes the initial states (with negative time).

total_number_of_states : float.
    The total number of (observed and hidden) states, used to index the variance matrix

observation_index : int.
    The index for the current observation time in the main kalman_filter loop

Returns
-------

    predicted_observation_mean_derivatives[observation_index + 1] : numpy array.
        An array of dimension 7 x 2, which contains the derivative of the mean mRNA
        and protein wrt each parameter at the current observation time point


    predicted_observation_variance_derivatives[observation_index + 1] : numpy array.
        An array of dimension 7 x 2 x 2, which describes the derivative of the state
        space variance wrt each parameter for the current time point
"""
function kalman_observation_derivatives(predicted_observation_mean_derivatives,
                                   predicted_observation_variance_derivatives,
                                   current_observation,
                                   state_space_mean_derivative,
                                   state_space_variance_derivative,
                                   current_number_of_states,
                                   total_number_of_states,
                                   observation_index)
    predicted_observation_mean_derivatives[observation_index+1,:,:] .= state_space_mean_derivative[current_number_of_states,:,:]
    predicted_observation_variance_derivatives[observation_index+1,:,:,:] .= state_space_variance_derivative[:,
                                                                                                            [current_number_of_states,
                                                                                                             total_number_of_states+current_number_of_states],
                                                                                                            [current_number_of_states,
                                                                                                             total_number_of_states+current_number_of_states]]
    return predicted_observation_mean_derivatives[observation_index+1], predicted_observation_variance_derivatives[observation_index+1]
end # function

"""
Perform the Kalman filter prediction about future observation, based on current knowledge i.e. current
state space mean and variance. This gives rho_{t+delta t-tau:t+delta t} and P_{t+delta t-tau:t+delta t},
using the differential equations in supplementary section 4 of Calderazzo et al., Bioinformatics (2018),
approximated using a forward Euler scheme.

TODO: update variable descriptions
Parameters
----------

state_space_mean : numpy array.
    The dimension is n x 3, where n is the number of states until the current time.
    The first column is time, the second column is mean mRNA, and the third column is mean protein. It
    represents the information based on observations we have already made.

state_space_variance : numpy array.
    The dimension is 2n x 2n, where n is the number of states until the current time. The definition
    is identical to the one provided in the Kalman filter function, i.e.
        [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
          cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

state_space_mean_derivative : numpy array.
    An array of dimension n x m x 2, where n is the number of inferred time points,
    and m is the number of parameters. The m columns in the second dimension are the
    derivative of the state mean with respect to each parameter. The two elements in
    the third dimension represent the derivative of mRNA and protein respectively

state_space_variance_derivative : numpy array.
    An array of dimension 7 x 2n x 2n.
          [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
            d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

current_observation : numpy array.
    The dimension is 1 x 2, where the first entry is time, and the second is the protein observation.

model_parameters : numpy array.
    An array containing the model parameters. The order is identical to the one provided in the
    Kalman filter function documentation, i.e.
    repression_threshold, hill_coefficient, mRNA_degradation_rate,
    protein_degradation_rate, basal_transcription_rate, translation_rate,
    transcription_delay.

observation_time_step : float.
    This gives the time between each experimental observation. This is required to know how far
    the function should predict.

Returns
-------
predicted_state_space_mean : numpy array.
    The dimension is n x 3, where n is the number of previous observations until the current time.
    The first column is time, the second column is mean mRNA, and the third column is mean protein.

predicted_state_space_variance : numpy array.
The dimension is 2n x 2n, where n is the number of previous observations until the current time.
    [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
      cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

state_space_mean_derivative : numpy array.
    An array of dimension n x m x 2, where n is the number of inferred time points,
    and m is the number of parameters. The m columns in the second dimension are the
    derivative of the state mean with respect to each parameter. The two elements in
    the third dimension represent the derivative of mRNA and protein respectively

state_space_variance_derivative : numpy array.
    An array of dimension 7 x 2n x 2n.
          [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
            d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]
"""
# kalman_prediction_step(state_space_and_distributions.state_space,
#                                      current_observation,
#                                      model_parameters,
#                                      observation_time_step)
function kalman_prediction_step(
    state_space::StateSpace,
    current_observation,
    model_parameters::ModelParameters,
    observation_time_step
    )
    # This is the time step dt in the forward euler scheme
    discretisation_time_step = 1.0
    ## name the model parameters
    #@unpack repression_threshold, hill_coefficient, ... = model_parameters
    repression_threshold = model_parameters[1]
    hill_coefficient = model_parameters[2]
    mRNA_degradation_rate = model_parameters[3]
    protein_degradation_rate = model_parameters[4]
    basal_transcription_rate = model_parameters[5]
    translation_rate = model_parameters[6]
    transcription_delay = model_parameters[7]

    discrete_delay = Int64(round(transcription_delay/discretisation_time_step))
    number_of_hidden_states = Int64(round(observation_time_step/discretisation_time_step))

    # this is the number of states at t, i.e. before predicting towards t+observation_time_step
    current_number_of_states = (Int64(round(current_observation[1]/observation_time_step))-1)*number_of_hidden_states + discrete_delay + 1
    total_number_of_states = size(state_space_mean,1)

    ## next_time_index corresponds to 't+Deltat' in the propagation equation on page 5 of the supplementary
    ## material in the calderazzo paper

    # we initialise all our matrices outside of the main for loop for improved performance
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

    ## probably get rid
    # # this is d_rho(t)/d_theta
    # next_mean_derivative = Array{Float64}(undef,(7,2));#zeros((7,2))
    # # this is d_P(t,t)/d_theta
    # current_covariance_derivative_matrix = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))
    # # this is d_P(t-tau,t)/d_theta
    # covariance_derivative_matrix_past_to_now = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))
    # # this is d_P(t,t-tau)/d_theta
    # covariance_derivative_matrix_now_to_past = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))
    # # d_P(t+Deltat,t+Deltat)/d_theta
    # next_covariance_derivative_matrix = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))
    # # initialisation for the common part of the derivative of P(t,t) for each parameter
    # common_state_space_variance_derivative_element = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))
    # # This corresponds to d_P(s,t)/d_theta in the Calderazzo paper
    # covariance_matrix_derivative_intermediate_to_current = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))
    # # This corresponds to d_P(s,t-tau)/d_theta
    # covariance_matrix_derivative_intermediate_to_past = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))
    # # This corresponds to d_P(s,t+Deltat)/d_theta in the Calderazzo paper
    # covariance_matrix_derivative_intermediate_to_next = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))
    # # initialisation for the common part of the derivative of P(s,t) for each parameter
    # common_intermediate_state_space_variance_derivative_element = Array{Float64}(undef,(7,2,2));#zeros((7,2,2))

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

    for next_time_index in (current_number_of_states + 1):(current_number_of_states + number_of_hidden_states)
        current_time_index = next_time_index - 1 # this corresponds to t
        past_time_index = current_time_index - discrete_delay # this corresponds to t-tau
        # indexing with 1:3 for numba
        current_mean = state_space_mean[current_time_index,[2,3]]
        past_protein = state_space_mean[past_time_index,3]

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

        next_mean = current_mean .+ discretisation_time_step .* derivative_of_mean
        # ensures the prediction is non negative
        next_mean[next_mean.<0] .= 0
        # indexing with 1:3 for numba
        state_space_mean[next_time_index,[2,3]] .= next_mean
        # in the next lines we use for loop instead of np.ix_-like indexing for numba
        current_covariance_matrix = state_space_variance[[current_time_index,
                                                          total_number_of_states+current_time_index],
                                                          [current_time_index,
                                                           total_number_of_states+current_time_index]]

        # this is P(t-tau,t) in page 5 of the supplementary material of Calderazzo et. al
        covariance_matrix_past_to_now .= state_space_variance[[past_time_index,
                                                              total_number_of_states+past_time_index],
                                                              [current_time_index,
                                                               total_number_of_states+current_time_index]]

        # this is P(t,t-tau) in page 5 of the supplementary material of Calderazzo et. al.
        covariance_matrix_now_to_past .= state_space_variance[[current_time_index,
                                                              total_number_of_states+current_time_index],
                                                              [past_time_index,
                                                               total_number_of_states+past_time_index]]

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
        next_covariance_matrix = current_covariance_matrix .+ discretisation_time_step .* derivative_of_variance
        # ensure that the diagonal entries are non negative
        # this is a little annoying to do in Julia, but here we create a mask matrix with a 1 at any negative diagonal entries
        next_covariance_matrix[diagm(next_covariance_matrix[diagind(next_covariance_matrix)].<0)] .= 0

        state_space_variance[[next_time_index,
                              total_number_of_states+next_time_index],
                             [next_time_index,
                              total_number_of_states+next_time_index]] .= next_covariance_matrix

        ## now we need to update the cross correlations, P(s,t) in the Calderazzo paper
        # the range needs to include t, since we want to propagate P(t,t) into P(t,t+Deltat)
        for intermediate_time_index in past_time_index:current_time_index
            # This corresponds to P(s,t) in the Calderazzo paper
            covariance_matrix_intermediate_to_current .= state_space_variance[[intermediate_time_index,
                                                                              total_number_of_states+intermediate_time_index],
                                                                             [current_time_index,
                                                                              total_number_of_states+current_time_index]]
            # This corresponds to P(s,t-tau)
            covariance_matrix_intermediate_to_past .= state_space_variance[[intermediate_time_index,
                                                                           total_number_of_states+intermediate_time_index],
                                                                           [past_time_index,
                                                                            total_number_of_states+past_time_index]]


            covariance_derivative = ( covariance_matrix_intermediate_to_current*instant_jacobian_transpose .+
                                      covariance_matrix_intermediate_to_past*delayed_jacobian_transpose )

            # This corresponds to P(s,t+Deltat) in the Calderazzo paper
            covariance_matrix_intermediate_to_next .= covariance_matrix_intermediate_to_current .+ discretisation_time_step.*covariance_derivative

            # Fill in the big matrix
            state_space_variance[[intermediate_time_index,
                                  total_number_of_states+intermediate_time_index],
                                 [next_time_index,
                                  total_number_of_states+next_time_index]] .= covariance_matrix_intermediate_to_next
            # Fill in the big matrix with transpose arguments, i.e. P(t+Deltat, s) - works if initialised symmetrically
            state_space_variance[[next_time_index,
                                  total_number_of_states+next_time_index],
                                 [intermediate_time_index,
                                  total_number_of_states+intermediate_time_index]] .= transpose(covariance_matrix_intermediate_to_next)
        end # intermediate time index for
    end # for (next time index)
    return StateSpace(state_space_mean, state_space_variance)
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

TODO: update variable descriptions
Parameters
----------

state_space_mean : numpy array.
    The dimension is n x 3, where n is the number of states until the current time.
    The first column is time, the second column is mean mRNA, and the third column is mean protein.

state_space_variance : numpy array.
    The dimension is 2n x 2n, where n is the number of states until the current time.
        [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
          cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ]

state_space_mean_derivative : numpy array.
    An array of dimension n x m x 2, where n is the number of inferred time points,
    and m is the number of parameters. The m columns in the second dimension are the
    derivative of the state mean with respect to each parameter. The two elements in
    the third dimension represent the derivative of mRNA and protein respectively

state_space_variance_derivative : numpy array.
    An array of dimension 7 x 2n x 2n.
          [ d[cov( mRNA(t0:tn),mRNA(t0:tn) )]/d_theta,    d[cov( protein(t0:tn),mRNA(t0:tn) )]/d_theta,
            d[cov( mRNA(t0:tn),protein(t0:tn) )/]d_theta, d[cov( protein(t0:tn),protein(t0:tn) )]/d_theta ]

current_observation : numpy array.
    The dimension is 1 x 2, where the first entry is time, and the second is the protein observation.

time_delay : float.
    The fixed transciptional time delay in the system. This tells us how far back we need to update our
    state space estimates.

observation_time_step : float.
    The fixed time interval between protein observations.

measurement_variance : float.
    The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

Returns
-------

state_space_mean : numpy array.
    The dimension is n x 3, where the first column is time, and the second and third columns are the mean
    mRNA and mean protein levels respectively. This corresponds to rho* in
    Calderazzo et al., Bioinformatics (2018).

state_space_variance : numpy array.
    This corresponds to P* in Calderazzo et al., Bioinformatics (2018).
    The dimension is 2n x 2n, where n is the number of states until the current time.
        [ cov( mRNA(t0:tn),mRNA(t0:tn) ),    cov( protein(t0:tn),mRNA(t0:tn) ),
          cov( mRNA(t0:tn),protein(t0:tn) ), cov( protein(t0:tn),protein(t0:tn) ].
"""
function kalman_update_step(
    state_space::StateSpace,
    current_observation,
    time_delay,
    observation_time_step,
    measurement_variance
    )

    discretisation_time_step = state_space_mean[2,1] - state_space_mean[1,1]
    discrete_delay = Int64(round(time_delay/discretisation_time_step))
    number_of_hidden_states = Int64(round(observation_time_step/discretisation_time_step))

    # this is the number of states at t+Deltat, i.e. after predicting towards t+observation_time_step
    current_number_of_states = (Int64(round(current_observation[1]/observation_time_step)))*number_of_hidden_states + discrete_delay+1
    total_number_of_states = size(state_space_mean,1)

    # predicted_state_space_mean until delay, corresponds to
    # rho(t+Deltat-delay:t+deltat). Includes current value and discrete_delay past values
    # funny indexing with 1:3 instead of (1,2) to make numba happy
    shortened_state_space_mean = state_space_mean[(current_number_of_states-discrete_delay):current_number_of_states,[2,3]]

    # put protein values underneath mRNA values, to make vector of means (rho)
    # consistent with variance (P)
    stacked_state_space_mean = vcat(shortened_state_space_mean[:,1],shortened_state_space_mean[:,2])

    # funny indexing with 1:3 instead of (1,2) to make numba happy
    predicted_final_state_space_mean = copy(state_space_mean[current_number_of_states,[2,3]])

    # extract covariance matrix up to delay
    # corresponds to P(t+Deltat-delay:t+deltat,t+Deltat-delay:t+deltat)
    mRNA_indices_to_keep = (current_number_of_states - discrete_delay):(current_number_of_states)
    protein_indices_to_keep = (total_number_of_states + current_number_of_states - discrete_delay):(total_number_of_states + current_number_of_states)
    all_indices_up_to_delay = vcat(mRNA_indices_to_keep, protein_indices_to_keep)

    # using for loop indexing for numba
    shortened_covariance_matrix = state_space_variance[all_indices_up_to_delay,all_indices_up_to_delay]
    # extract P(t+Deltat-delay:t+deltat,t+Deltat)
    shortened_covariance_matrix_past_to_final = shortened_covariance_matrix[:,[discrete_delay+1,end]]

    # and P(t+Deltat,t+Deltat-delay:t+deltat)
    shortened_covariance_matrix_final_to_past = shortened_covariance_matrix[[discrete_delay+1,end],:]

    # This is F in the paper
    observation_transform = [0.0 1.0]

    # This is P(t+Deltat,t+Deltat) in the paper
    predicted_final_covariance_matrix = state_space_variance[[current_number_of_states,
                                                              total_number_of_states+current_number_of_states],
                                                             [current_number_of_states,
                                                              total_number_of_states+current_number_of_states]]

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
    updated_state_space_mean = hcat(updated_stacked_state_space_mean[1:(discrete_delay+1)],
                                    updated_stacked_state_space_mean[(discrete_delay+2):end])
    # Fill in the updated values
    # funny indexing with 1:3 instead of (1,2) to make numba happy
    state_space_mean[(current_number_of_states-discrete_delay):current_number_of_states,[2,3]] .= updated_state_space_mean

    # This is P*
    updated_shortened_covariance_matrix = ( shortened_covariance_matrix .-
                                            adaptation_coefficient*observation_transform*shortened_covariance_matrix_final_to_past )
    # ensure that the diagonal entries are non negative
    # np.fill_diagonal(updated_shortened_covariance_matrix,np.maximum(np.diag(updated_shortened_covariance_matrix),0))
    updated_shortened_covariance_matrix[diagm(updated_shortened_covariance_matrix[diagind(updated_shortened_covariance_matrix)].<0)] .= 0

    # Fill in updated values
    state_space_variance[all_indices_up_to_delay,all_indices_up_to_delay] .= updated_shortened_covariance_matrix

    return StateSpace(state_space_mean, state_space_variance)
end # function
