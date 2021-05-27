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
function kalman_filter(protein_at_observations,model_parameters,measurement_variance = 10)
    time_delay = model_parameters[7]

    number_of_observations = size(protein_at_observations,1)
    observation_time_step = protein_at_observations[2,1]-protein_at_observations[1,1]
    # This is the time step dt in the forward euler scheme
    discretisation_time_step = 1.0
    # This is the delay as an integer multiple of the discretization timestep so that we can index with it
    discrete_delay = Int64(round(time_delay/discretisation_time_step))
    number_of_hidden_states = Int64(round(observation_time_step/discretisation_time_step))
    initial_number_of_states = discrete_delay + 1
    total_number_of_states = initial_number_of_states + (number_of_observations - 1)*number_of_hidden_states
    # scaling factors for mRNA and protein respectively. For example, observation might be fluorescence,
    # so the scaling would correspond to how light intensity relates to molecule number.
    observation_transform = [0.0 1.0]

    state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives = kalman_filter_state_space_initialisation(protein_at_observations,
                                                                                                                                                                                                                                                                             model_parameters,
                                                                                                                                                                                                                                                                             measurement_variance)
    # loop through observations and at each observation apply the Kalman prediction step and then the update step
    # for observation_index, current_observation in enumerate(protein_at_observations[1:]):
    for observation_index in 1:(size(protein_at_observations,1)-1)
        current_observation = protein_at_observations[1 + observation_index,:]
        saving_path = string(pwd(),"/test/output/");
        if observation_index == 1
            writedlm(string(saving_path,"state_space_mean_before_pred.csv"),state_space_mean,",");
            writedlm(string(saving_path,"state_space_variance_before_pred.csv"),state_space_variance,",");
            writedlm(string(saving_path,"state_space_mean_div_before_pred.csv"),state_space_mean_derivative,",");
            writedlm(string(saving_path,"state_space_variance_div_before_pred.csv"),state_space_variance_derivative,",");
        end
        state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_prediction_step(state_space_mean,
                                                                                                                                  state_space_variance,
                                                                                                                                  state_space_mean_derivative,
                                                                                                                                  state_space_variance_derivative,
                                                                                                                                  current_observation,
                                                                                                                                  model_parameters,
                                                                                                                                  observation_time_step)
        if observation_index == 1
            writedlm(string(saving_path,"state_space_mean_after_pred.csv"),state_space_mean,",");
            writedlm(string(saving_path,"state_space_variance_after_pred.csv"),state_space_variance,",");
            writedlm(string(saving_path,"state_space_mean_div_after_pred.csv"),state_space_mean_derivative,",");
            writedlm(string(saving_path,"state_space_variance_div_after_pred.csv"),state_space_variance_derivative,",");
        end

        current_number_of_states = Int64(round(current_observation[1]/observation_time_step))*number_of_hidden_states + initial_number_of_states

    # between the prediction and update steps we record the mean and sd for our likelihood, and the derivatives of the mean and variance for the
    # derivative of the likelihood wrt the parameters
        predicted_observation_distributions[observation_index+1] = kalman_observation_distribution_parameters(predicted_observation_distributions,
                                                                                                            current_observation,
                                                                                                            state_space_mean,
                                                                                                            state_space_variance,
                                                                                                            current_number_of_states,
                                                                                                            total_number_of_states,
                                                                                                            measurement_variance,
                                                                                                            observation_index)

        predicted_observation_mean_derivatives[observation_index+1], predicted_observation_variance_derivatives[observation_index+1] = kalman_observation_derivatives(predicted_observation_mean_derivatives,
                                                                                                                                                                  predicted_observation_variance_derivatives,
                                                                                                                                                                  current_observation,
                                                                                                                                                                  state_space_mean_derivative,
                                                                                                                                                                  state_space_variance_derivative,
                                                                                                                                                                  current_number_of_states,
                                                                                                                                                                  total_number_of_states,
                                                                                                                                                                  observation_index)

        state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_update_step(state_space_mean,
                                                                                                                                  state_space_variance,
                                                                                                                                  state_space_mean_derivative,
                                                                                                                                  state_space_variance_derivative,
                                                                                                                                  current_observation,
                                                                                                                                  time_delay,
                                                                                                                                  observation_time_step,
                                                                                                                                  measurement_variance)
    end # for
    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives
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
function kalman_filter_state_space_initialisation(protein_at_observations,model_parameters,measurement_variance = 10)
    time_delay = model_parameters[7]

    # This is the time step dt in the forward euler scheme
    discretisation_time_step = 1.0
    # This is the delay as an integer multiple of the discretization timestep so that we can index with it
    discrete_delay = Int64(round(time_delay/discretisation_time_step))

    observation_time_step = protein_at_observations[2,1]-protein_at_observations[1,1]
    number_of_observations = size(protein_at_observations,1)

    # 'synthetic' observations, which allow us to update backwards in time
    number_of_hidden_states = Int64(round(observation_time_step/discretisation_time_step))

    ## initialise "negative time" with the mean and standard deviations of the LNA
    initial_number_of_states = discrete_delay + 1
    total_number_of_states = initial_number_of_states + (number_of_observations - 1)*number_of_hidden_states

    state_space_mean = zeros((total_number_of_states,3))
    state_space_mean[1:initial_number_of_states,2] .= 41.41785876041631 # placeholder until steady state function is implemented
    state_space_mean[1:initial_number_of_states,3] .= 5377.800549410291 # placeholder until steady state function is implemented
                                                            # hes5.calculate_steady_state_of_ode(repression_threshold=model_parameters[1],
                                                            #                                    hill_coefficient=model_parameters[2],
                                                            #                                    mRNA_degradation_rate=model_parameters[3],
                                                            #                                    protein_degradation_rate=model_parameters[4],
                                                            #                                    basal_transcription_rate=model_parameters[5],
                                                            #                                    translation_rate=model_parameters[6])

    final_observation_time = protein_at_observations[end,1]
    # assign time entries
    state_space_mean[:,1] = LinRange(protein_at_observations[1,1]-discrete_delay,final_observation_time,total_number_of_states)

    # initialise initial covariance matrix
    state_space_variance = zeros((2*(total_number_of_states),2*(total_number_of_states)))

    # set the mRNA and protein variance at negative times to the LNA approximation
    initial_mRNA_scaling = 20.0
    initial_protein_scaling = 100.0
    initial_mRNA_variance = state_space_mean[1,2]*initial_mRNA_scaling
    initial_protein_variance = state_space_mean[1,3]*initial_protein_scaling
    # diagm(diagind(A)[1 .<= diagind(A).< initial_number_of_states])
    for diag_index in 1:initial_number_of_states
        state_space_variance[diag_index,diag_index] = initial_mRNA_variance
        state_space_variance[diag_index + total_number_of_states,diag_index + total_number_of_states] = initial_protein_variance
    end #for

    observation_transform = [0.0 1.0]

    predicted_observation_distributions = zeros(number_of_observations,3)
    predicted_observation_distributions[1,1] = 0
    predicted_observation_distributions[1,2] = dot(observation_transform,state_space_mean[initial_number_of_states,2:3])

    last_predicted_covariance_matrix = state_space_variance[[initial_number_of_states,total_number_of_states+initial_number_of_states],
                                                            [initial_number_of_states,total_number_of_states+initial_number_of_states]]

    predicted_observation_distributions[1,3] = dot(observation_transform,last_predicted_covariance_matrix*transpose(observation_transform)) + measurement_variance

    ####################################################################
    ####################################################################
    ##
    ## initialise derivative arrays
    ##
    ####################################################################
    ####################################################################
    #
    state_space_mean_derivative = zeros(total_number_of_states,7,2)
    #
    repression_threshold = model_parameters[1]
    hill_coefficient = model_parameters[2]
    mRNA_degradation_rate = model_parameters[3]
    protein_degradation_rate = model_parameters[4]
    basal_transcription_rate = model_parameters[5]
    translation_rate = model_parameters[6]
    transcription_delay = model_parameters[7]

    steady_state_protein = state_space_mean[1,3]

    hill_function_value = 1.0/(1.0+(steady_state_protein/repression_threshold)^hill_coefficient)
    hill_function_derivative_value_wrt_protein = - hill_coefficient*(steady_state_protein/repression_threshold)^(hill_coefficient - 1)/( repression_threshold*
                                                   ((1.0+(steady_state_protein/repression_threshold)^hill_coefficient)^2))
    protein_derivative_denominator_scalar = (basal_transcription_rate*translation_rate)/(mRNA_degradation_rate*protein_degradation_rate)
    initial_protein_derivative_denominator = (protein_derivative_denominator_scalar*hill_function_derivative_value_wrt_protein) - 1

    # assign protein derivative first, since mRNA derivative is given as a function of protein derivative

    hill_function_derivative_value_wrt_repression = (hill_coefficient*(steady_state_protein/repression_threshold)^hill_coefficient)/( repression_threshold*
                                                   ((1.0+(steady_state_protein/repression_threshold)^hill_coefficient)^2))
    hill_function_derivative_value_wrt_hill_coefficient = - (log(steady_state_protein/repression_threshold)*(steady_state_protein/repression_threshold)^hill_coefficient)/(
                                                            (1.0+(steady_state_protein/repression_threshold)^hill_coefficient)^2)
    # repression threshold
    state_space_mean_derivative[1:initial_number_of_states,1,2] .= - (protein_derivative_denominator_scalar*hill_function_derivative_value_wrt_repression)/(
                                                                    initial_protein_derivative_denominator)

    state_space_mean_derivative[1:initial_number_of_states,1,1] .= (protein_degradation_rate/translation_rate)*state_space_mean_derivative[1,1,2]

    # hill coefficient
    state_space_mean_derivative[1:initial_number_of_states,2,2] .= - (protein_derivative_denominator_scalar*hill_function_derivative_value_wrt_hill_coefficient)/(
                                                                    initial_protein_derivative_denominator)

    state_space_mean_derivative[1:initial_number_of_states,2,1] .= (protein_degradation_rate/translation_rate)*state_space_mean_derivative[1,2,2]

    # mRNA degradation
    state_space_mean_derivative[1:initial_number_of_states,3,2] .= (protein_derivative_denominator_scalar*hill_function_value)/(
                                                                  mRNA_degradation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[1:initial_number_of_states,3,1] .= (protein_degradation_rate/translation_rate)*state_space_mean_derivative[1,3,2]

    # protein degradation
    state_space_mean_derivative[1:initial_number_of_states,4,2] .= (protein_derivative_denominator_scalar*hill_function_value)/(
                                                                  protein_degradation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[1:initial_number_of_states,4,1] .= (steady_state_protein + protein_degradation_rate*state_space_mean_derivative[1,4,2])/translation_rate

    # basal transcription
    state_space_mean_derivative[1:initial_number_of_states,5,2] .= -(protein_derivative_denominator_scalar*hill_function_value)/(
                                                                   basal_transcription_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[1:initial_number_of_states,5,1] .= (protein_degradation_rate/translation_rate)*state_space_mean_derivative[1,5,2]

    # translation
    state_space_mean_derivative[1:initial_number_of_states,6,2] .= -(protein_derivative_denominator_scalar*hill_function_value)/(
                                                                   translation_rate*initial_protein_derivative_denominator)

    state_space_mean_derivative[1:initial_number_of_states,6,1] .= -(protein_degradation_rate/translation_rate)*((steady_state_protein/translation_rate) -
                                                                                                               state_space_mean_derivative[1,6,2])
    # transcriptional delay
    state_space_mean_derivative[1:initial_number_of_states,7,2] .= 0
    state_space_mean_derivative[1:initial_number_of_states,7,1] .= 0

    state_space_variance_derivative = zeros(7,2*total_number_of_states,2*total_number_of_states)
    for parameter_index in 1:7
        for diagonal_index in 1:initial_number_of_states
        state_space_variance_derivative[parameter_index,diagonal_index,diagonal_index] = initial_mRNA_scaling*state_space_mean_derivative[1,parameter_index,1]
        state_space_variance_derivative[parameter_index,diagonal_index+total_number_of_states,diagonal_index+total_number_of_states] = initial_protein_scaling*state_space_mean_derivative[1,parameter_index,2]
        end # for
    end # for
    predicted_observation_mean_derivatives = zeros((number_of_observations,7,2))
    predicted_observation_mean_derivatives[1] = state_space_mean_derivative[initial_number_of_states]

    predicted_observation_variance_derivatives = zeros((number_of_observations,7,2,2))
    for parameter_index in 1:7
        predicted_observation_variance_derivatives[1,parameter_index,[1,2],[1,2]] = state_space_variance_derivative[parameter_index,[initial_number_of_states,
                                                                                                                          total_number_of_states+initial_number_of_states],
                                                                                                                         [initial_number_of_states,
                                                                                                                          total_number_of_states+initial_number_of_states]]
    end # for

    # update the past ("negative time")
    current_observation = protein_at_observations[1,:]
    state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative = kalman_update_step(state_space_mean,
                                                                                                                              state_space_variance,
                                                                                                                              state_space_mean_derivative,
                                                                                                                              state_space_variance_derivative,
                                                                                                                              current_observation,
                                                                                                                              time_delay,
                                                                                                                              observation_time_step,
                                                                                                                              measurement_variance)

    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives
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

    predicted_observation_distributions[observation_index+1,3] = dot(observation_transform,last_predicted_covariance_matrix*transpose(observation_transform)) + measurement_variance

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
    predicted_observation_mean_derivatives[observation_index,:,:] = copy(state_space_mean_derivative[current_number_of_states,:,:])
    for parameter_index in 1:7
        predicted_observation_variance_derivatives[observation_index,parameter_index,[1,2],[1,2]] = copy(state_space_variance_derivative[parameter_index,
                                                                                                                                    [current_number_of_states,
                                                                                                                                     total_number_of_states+current_number_of_states],
                                                                                                                                     [current_number_of_states,
                                                                                                                                      total_number_of_states+current_number_of_states]])
    end # for
    return predicted_observation_mean_derivatives[observation_index], predicted_observation_variance_derivatives[observation_index]
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
function kalman_prediction_step(state_space_mean,
                           state_space_variance,
                           state_space_mean_derivative,
                           state_space_variance_derivative,
                           current_observation,
                           model_parameters,
                           observation_time_step)
    # This is the time step dt in the forward euler scheme
    discretisation_time_step = 1.0
    ## name the model parameters
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
    current_covariance_matrix = zeros((2,2))
    # this is P(t-tau,t) in page 5 of the supplementary material of Calderazzo et. al.
    covariance_matrix_past_to_now = zeros((2,2))
    # this is P(t,t-tau) in page 5 of the supplementary material of Calderazzo et. al.
    covariance_matrix_now_to_past = zeros((2,2))
    # This corresponds to P(s,t) in the Calderazzo paper
    covariance_matrix_intermediate_to_current = zeros((2,2))
    # This corresponds to P(s,t-tau)
    covariance_matrix_intermediate_to_past = zeros((2,2))

    # this is d_rho(t)/d_theta
    next_mean_derivative = zeros((7,2))
    # this is d_P(t,t)/d_theta
    current_covariance_derivative_matrix = zeros((7,2,2))
    # this is d_P(t-tau,t)/d_theta
    covariance_derivative_matrix_past_to_now = zeros((7,2,2))
    # this is d_P(t,t-tau)/d_theta
    covariance_derivative_matrix_now_to_past = zeros((7,2,2))
    # d_P(t+Deltat,t+Deltat)/d_theta
    next_covariance_derivative_matrix = zeros((7,2,2))
    # initialisation for the common part of the derivative of P(t,t) for each parameter
    common_state_space_variance_derivative_element = zeros((7,2,2))
    # This corresponds to d_P(s,t)/d_theta in the Calderazzo paper
    covariance_matrix_derivative_intermediate_to_current = zeros((7,2,2))
    # This corresponds to d_P(s,t-tau)/d_theta
    covariance_matrix_derivative_intermediate_to_past = zeros((7,2,2))
    # This corresponds to d_P(s,t+Deltat)/d_theta in the Calderazzo paper
    covariance_matrix_derivative_intermediate_to_next = zeros((7,2,2))
    # initialisation for the common part of the derivative of P(s,t) for each parameter
    common_intermediate_state_space_variance_derivative_element = zeros((7,2,2))

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

        next_mean = current_mean + discretisation_time_step*derivative_of_mean
        # ensures the prediction is non negative
        next_mean[next_mean.<0] .= 0
        # indexing with 1:3 for numba
        state_space_mean[next_time_index,[2,3]] = next_mean
        # in the next lines we use for loop instead of np.ix_-like indexing for numba
        current_covariance_matrix = state_space_variance[[current_time_index,
                                                          total_number_of_states+current_time_index],
                                                          [current_time_index,
                                                           total_number_of_states+current_time_index]]

        # this is P(t-tau,t) in page 5 of the supplementary material of Calderazzo et. al
        covariance_matrix_past_to_now = state_space_variance[[past_time_index,
                                                              total_number_of_states+past_time_index],
                                                              [current_time_index,
                                                               total_number_of_states+current_time_index]]

        # this is P(t,t-tau) in page 5 of the supplementary material of Calderazzo et. al.
        covariance_matrix_now_to_past = state_space_variance[[current_time_index,
                                                              total_number_of_states+current_time_index],
                                                              [past_time_index,
                                                               total_number_of_states+past_time_index]]

        variance_change_current_contribution = ( instant_jacobian*current_covariance_matrix +
                                                 current_covariance_matrix*instant_jacobian_transpose )

        variance_change_past_contribution = ( delayed_jacobian*covariance_matrix_past_to_now +
                                              covariance_matrix_now_to_past*delayed_jacobian_transpose )

        variance_of_noise = [mRNA_degradation_rate*current_mean[1]+basal_transcription_rate*hill_function_value 0;
                             0 translation_rate*current_mean[1]+protein_degradation_rate*current_mean[2]]

        derivative_of_variance = ( variance_change_current_contribution +
                                   variance_change_past_contribution +
                                   variance_of_noise )

        # P(t+Deltat,t+Deltat)
        next_covariance_matrix = current_covariance_matrix + discretisation_time_step*derivative_of_variance
        # ensure that the diagonal entries are non negative
        # this is a little annoying to do in Julia, but here we create a mask matrix with a 1 at any negative diagonal entries
        next_covariance_matrix[diagm(next_covariance_matrix[diagind(next_covariance_matrix)].<0)] .= 0

        state_space_variance[[next_time_index,
                              total_number_of_states+next_time_index],
                             [next_time_index,
                              total_number_of_states+next_time_index]] = next_covariance_matrix

        ## now we need to update the cross correlations, P(s,t) in the Calderazzo paper
        # the range needs to include t, since we want to propagate P(t,t) into P(t,t+Deltat)
        for intermediate_time_index in past_time_index:current_time_index
            # This corresponds to P(s,t) in the Calderazzo paper
            covariance_matrix_intermediate_to_current = state_space_variance[[intermediate_time_index,
                                                                              total_number_of_states+intermediate_time_index],
                                                                             [current_time_index,
                                                                              total_number_of_states+current_time_index]]
            # This corresponds to P(s,t-tau)
            covariance_matrix_intermediate_to_past = state_space_variance[[intermediate_time_index,
                                                                           total_number_of_states+intermediate_time_index],
                                                                           [past_time_index,
                                                                            total_number_of_states+past_time_index]]


            covariance_derivative = ( covariance_matrix_intermediate_to_current*instant_jacobian_transpose +
                                      covariance_matrix_intermediate_to_past*delayed_jacobian_transpose )

            # This corresponds to P(s,t+Deltat) in the Calderazzo paper
            covariance_matrix_intermediate_to_next = covariance_matrix_intermediate_to_current + discretisation_time_step*covariance_derivative

            # Fill in the big matrix
            state_space_variance[[intermediate_time_index,
                                  total_number_of_states+intermediate_time_index],
                                 [next_time_index,
                                  total_number_of_states+next_time_index]] = covariance_matrix_intermediate_to_next
            # Fill in the big matrix with transpose arguments, i.e. P(t+Deltat, s) - works if initialised symmetrically
            state_space_variance[[next_time_index,
                                  total_number_of_states+next_time_index],
                                 [intermediate_time_index,
                                  total_number_of_states+intermediate_time_index]] = transpose(covariance_matrix_intermediate_to_next)
        end # intermediate time index for

    #################################
    ####
    #### prediction step for the derivatives of the state space mean and variance wrt each parameter
    ####
    #################################

    ###
    ### state space mean derivatives
    ###
        current_mean_derivative = state_space_mean_derivative[current_time_index,:,[1,2]]
        past_mean_derivative = state_space_mean_derivative[past_time_index,:,[1,2]]
        past_protein_derivative = state_space_mean_derivative[past_time_index,:,2]

        # calculate predictions for derivative of mean wrt each parameter
        # repression threshold
        hill_function_derivative_value_wrt_repression = (hill_coefficient*(past_protein/repression_threshold)^hill_coefficient)/(
                                                         repression_threshold*(1.0+(past_protein/repression_threshold)^hill_coefficient)^2)

        repression_derivative = ( instant_jacobian*current_mean_derivative[1,:] +
                                  delayed_jacobian*past_mean_derivative[1,:] +
                                  [basal_transcription_rate*hill_function_derivative_value_wrt_repression, 0.0] )

        next_mean_derivative[1,:] = current_mean_derivative[1,:] + discretisation_time_step*repression_derivative

        # hill coefficient
        hill_function_derivative_value_wrt_hill_coefficient = - (log(past_protein/repression_threshold)*(past_protein/repression_threshold)^hill_coefficient)/(
                                                                 (1.0+(past_protein/repression_threshold)^hill_coefficient)^2)

        hill_coefficient_derivative = ( instant_jacobian*current_mean_derivative[2,:] +
                                        delayed_jacobian*past_mean_derivative[2,:] +
                                        [basal_transcription_rate*hill_function_derivative_value_wrt_hill_coefficient, 0.0] )

        next_mean_derivative[2,:] = current_mean_derivative[2,:] + discretisation_time_step*hill_coefficient_derivative

        # mRNA degradation rate
        mRNA_degradation_rate_derivative = ( instant_jacobian*current_mean_derivative[3,:] +
                                             delayed_jacobian*past_mean_derivative[3,:] +
                                             [-current_mean[1], 0.0] )

        next_mean_derivative[3,:] = current_mean_derivative[3,:] + discretisation_time_step*mRNA_degradation_rate_derivative

        # protein degradation rate
        protein_degradation_rate_derivative = ( instant_jacobian*current_mean_derivative[4,:] +
                                                delayed_jacobian*past_mean_derivative[4,:] +
                                                [0.0, -current_mean[2]] )

        next_mean_derivative[4,:] = current_mean_derivative[4,:] + discretisation_time_step*protein_degradation_rate_derivative

        # basal transcription rate
        basal_transcription_rate_derivative = ( instant_jacobian*current_mean_derivative[5,:] +
                                                delayed_jacobian*past_mean_derivative[5,:] +
                                                [hill_function_value, 0.0])

        next_mean_derivative[5,:] = current_mean_derivative[5,:] + discretisation_time_step*basal_transcription_rate_derivative

        # translation rate
        translation_rate_derivative = ( instant_jacobian*current_mean_derivative[6,:] +
                                        delayed_jacobian*past_mean_derivative[6,:] +
                                        [0.0, current_mean[1]] )

        next_mean_derivative[6,:] = current_mean_derivative[6,:] + discretisation_time_step*translation_rate_derivative

        # transcriptional delay
        transcription_delay_derivative = ( instant_jacobian*current_mean_derivative[7,:] +
                                           delayed_jacobian*past_mean_derivative[7,:] )

        next_mean_derivative[7,:] = current_mean_derivative[7,:] + discretisation_time_step*transcription_delay_derivative

        # assign the predicted derivatives to our state_space_mean_derivative array
        state_space_mean_derivative[next_time_index,:,:] = next_mean_derivative

        ###
        ### state space variance derivatives
        ###

        # in the next lines we use for loop instead of np.ix_-like indexing for numba
        for parameter_index in 1:7
            # this is d_P(t,t)/d_theta
            current_covariance_derivative_matrix[parameter_index,:,:] = state_space_variance_derivative[parameter_index,
                                                                                                        [current_time_index,
                                                                                                         total_number_of_states+current_time_index],
                                                                                                        [current_time_index,
                                                                                                         total_number_of_states+current_time_index]]

             # this is d_P(t-tau,t)/d_theta
             covariance_derivative_matrix_past_to_now[parameter_index,:,:] = state_space_variance_derivative[parameter_index,
                                                                                                             [past_time_index,
                                                                                                              total_number_of_states+past_time_index],
                                                                                                              [current_time_index,
                                                                                                              total_number_of_states+current_time_index]]
             # this is d_P(t,t-tau)/d_theta
             covariance_derivative_matrix_now_to_past[parameter_index,:,:] = state_space_variance_derivative[parameter_index,
                                                                                                             [current_time_index,
                                                                                                              total_number_of_states+current_time_index],
                                                                                                             [past_time_index,
                                                                                                              total_number_of_states+past_time_index]]
          # the derivative is quite long and slightly different for each parameter, meaning it's difficult to
          # code this part with a loop. For each parameter we divide it in to it's constituent parts. There is one
          # main part in common for every derivative which is defined here as common_state_space_variance_derivative_element
              common_state_space_variance_derivative_element[parameter_index,:,:] = ( instant_jacobian*current_covariance_derivative_matrix[parameter_index,:,:] +
                                                                                      current_covariance_derivative_matrix[parameter_index,:,:]*instant_jacobian_transpose +
                                                                                      delayed_jacobian*covariance_derivative_matrix_past_to_now[parameter_index,:,:] +
                                                                                      covariance_derivative_matrix_now_to_past[parameter_index,:,:]*delayed_jacobian_transpose )
          end # for

        ## d_P(t+Deltat,t+Deltat)/d_theta

        hill_function_second_derivative_value = hill_coefficient*(past_protein/repression_threshold)^hill_coefficient*(
                                                (past_protein/repression_threshold)^hill_coefficient +
                                                hill_coefficient*(((past_protein/repression_threshold)^hill_coefficient)-1)+1)/(
                                                past_protein^2*(1.0+(past_protein/repression_threshold)^hill_coefficient)^3)
        # repression threshold
        # this refers to d(f'(p(t-tau)))/dp_0
        hill_function_second_derivative_value_wrt_repression = hill_coefficient^2*((past_protein/repression_threshold)^(hill_coefficient)-1)*(past_protein/repression_threshold)^(hill_coefficient-1)/(
                                                                repression_threshold^2*((1.0+(past_protein/repression_threshold)^hill_coefficient)^3))

        # instant_jacobian_derivative_wrt_repression = 0
        delayed_jacobian_derivative_wrt_repression = ( [0.0 basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[1,2]; 0.0 0.0] +
                                                       [0.0 basal_transcription_rate*hill_function_second_derivative_value_wrt_repression; 0.0 0.0] )
        delayed_jacobian_derivative_wrt_repression_transpose = transpose(delayed_jacobian_derivative_wrt_repression)

        instant_noise_derivative_wrt_repression = ( [mRNA_degradation_rate*current_mean_derivative[1,1] 0.0;
                                                     0.0 translation_rate*current_mean_derivative[1,1] + protein_degradation_rate*current_mean_derivative[1,2]] )

        delayed_noise_derivative_wrt_repression = ( [basal_transcription_rate*(hill_function_derivative_value*past_mean_derivative[1,2] + hill_function_derivative_value_wrt_repression) 0.0;
                                                     0.0 0.0] )

        derivative_of_variance_wrt_repression_threshold = ( common_state_space_variance_derivative_element[1,:,:] +
                                                            delayed_jacobian_derivative_wrt_repression*covariance_matrix_past_to_now +
                                                            covariance_matrix_now_to_past*delayed_jacobian_derivative_wrt_repression_transpose +
                                                            instant_noise_derivative_wrt_repression + delayed_noise_derivative_wrt_repression )

        next_covariance_derivative_matrix[1,:,:] = current_covariance_derivative_matrix[1,:,:] + discretisation_time_step*derivative_of_variance_wrt_repression_threshold

        # hill coefficient
        # this refers to d(f'(p(t-tau)))/dh
        hill_function_second_derivative_value_wrt_hill_coefficient = ((past_protein/repression_threshold)^hill_coefficient)*((-(past_protein/repression_threshold)^hill_coefficient) +
                                                                        hill_coefficient*(((past_protein/repression_threshold)^hill_coefficient)-1)*log(past_protein/repression_threshold)-1)/(
                                                                        past_protein*(1.0+(past_protein/repression_threshold)^hill_coefficient)^3)

        # instant_jacobian_derivative_wrt_hill_coefficient = 0
        delayed_jacobian_derivative_wrt_hill_coefficient = ( [0.0 basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[2,2]; 0.0 0.0] +
                                                             [0.0 basal_transcription_rate*hill_function_second_derivative_value_wrt_hill_coefficient; 0.0 0.0] )

        instant_noise_derivative_wrt_hill_coefficient = [mRNA_degradation_rate*current_mean_derivative[2,1] 0.0;
                                                         0.0 translation_rate*current_mean_derivative[2,1] + protein_degradation_rate*current_mean_derivative[2,2]]

        delayed_noise_derivative_wrt_hill_coefficient = [basal_transcription_rate*hill_function_derivative_value*past_mean_derivative[2,2] + hill_function_derivative_value_wrt_hill_coefficient 0.0;
                                                         0.0 0.0]

        derivative_of_variance_wrt_hill_coefficient = ( common_state_space_variance_derivative_element[2,:,:] +
                                                        delayed_jacobian_derivative_wrt_hill_coefficient*covariance_matrix_past_to_now +
                                                        covariance_matrix_now_to_past*transpose(delayed_jacobian_derivative_wrt_hill_coefficient) +
                                                        instant_noise_derivative_wrt_hill_coefficient + delayed_noise_derivative_wrt_hill_coefficient )

        next_covariance_derivative_matrix[2,:,:] = current_covariance_derivative_matrix[2,:,:] + discretisation_time_step*(derivative_of_variance_wrt_hill_coefficient)

        # mRNA degradation rate
        instant_jacobian_derivative_wrt_mRNA_degradation = [-1.0 0.0; 0.0 0.0]
        delayed_jacobian_derivative_wrt_mRNA_degradation = [0.0 basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[3,2]; 0.0 0.0]
        instant_noise_derivative_wrt_mRNA_degradation = [mRNA_degradation_rate*current_mean_derivative[3,1] + current_mean[1] 0.0;
                                                         0.0 translation_rate*current_mean_derivative[3,1] + protein_degradation_rate*current_mean_derivative[3,2]]

        delayed_noise_derivative_wrt_mRNA_degradation = [basal_transcription_rate*(hill_function_derivative_value*past_mean_derivative[3,2]) 0.0;
                                                         0.0 0.0]

        derivative_of_variance_wrt_mRNA_degradation = ( common_state_space_variance_derivative_element[3,:,:] +
                                                        instant_jacobian_derivative_wrt_mRNA_degradation*current_covariance_matrix +
                                                        current_covariance_matrix*transpose(instant_jacobian_derivative_wrt_mRNA_degradation) +
                                                        delayed_jacobian_derivative_wrt_mRNA_degradation*covariance_matrix_past_to_now +
                                                        covariance_matrix_now_to_past*transpose(delayed_jacobian_derivative_wrt_mRNA_degradation) +
                                                        instant_noise_derivative_wrt_mRNA_degradation + delayed_noise_derivative_wrt_mRNA_degradation )

        next_covariance_derivative_matrix[3,:,:] = current_covariance_derivative_matrix[3,:,:] + discretisation_time_step*(derivative_of_variance_wrt_mRNA_degradation)

        # protein degradation rate
        instant_jacobian_derivative_wrt_protein_degradation = [0.0 0.0; 0.0 -1.0]
        delayed_jacobian_derivative_wrt_protein_degradation = [0.0 basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[4,2];0.0 0.0]
        instant_noise_derivative_wrt_protein_degradation = [mRNA_degradation_rate*current_mean_derivative[4,1] 0.0;
                                                            0.0 translation_rate*current_mean_derivative[4,1] + protein_degradation_rate*current_mean_derivative[4,2] + current_mean[2]]

        delayed_noise_derivative_wrt_protein_degradation = [basal_transcription_rate*(hill_function_derivative_value*past_mean_derivative[4,2]) 0.0;
                                                            0.0 0.0]

        derivative_of_variance_wrt_protein_degradation = ( common_state_space_variance_derivative_element[4,:,:] +
                                                           instant_jacobian_derivative_wrt_protein_degradation*current_covariance_matrix +
                                                           current_covariance_matrix*transpose(instant_jacobian_derivative_wrt_protein_degradation) +
                                                           delayed_jacobian_derivative_wrt_protein_degradation*covariance_matrix_past_to_now +
                                                           covariance_matrix_now_to_past*transpose(delayed_jacobian_derivative_wrt_protein_degradation) +
                                                           instant_noise_derivative_wrt_protein_degradation + delayed_noise_derivative_wrt_protein_degradation )

        next_covariance_derivative_matrix[4,:,:] = current_covariance_derivative_matrix[4,:,:] + discretisation_time_step*(derivative_of_variance_wrt_protein_degradation)

        # basal transcription rate
        # instant_jacobian_derivative_wrt_basal_transcription = 0
        delayed_jacobian_derivative_wrt_basal_transcription = ( [0.0 basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[5,2];0.0 0.0] +
                                                                [0.0 hill_function_derivative_value; 0.0 0.0] )
        instant_noise_derivative_wrt_basal_transcription = [mRNA_degradation_rate*current_mean_derivative[5,1] 0.0;
                                                            0.0 translation_rate*current_mean_derivative[5,1] + protein_degradation_rate*current_mean_derivative[5,2]]

        delayed_noise_derivative_wrt_basal_transcription = [basal_transcription_rate*hill_function_derivative_value*past_mean_derivative[5,2] + hill_function_value 0.0;
                                                            0.0 0.0]

        derivative_of_variance_wrt_basal_transcription = ( common_state_space_variance_derivative_element[5,:,:] +
                                                           delayed_jacobian_derivative_wrt_basal_transcription*covariance_matrix_past_to_now +
                                                           covariance_matrix_now_to_past*transpose(delayed_jacobian_derivative_wrt_basal_transcription) +
                                                           instant_noise_derivative_wrt_basal_transcription + delayed_noise_derivative_wrt_basal_transcription )

        next_covariance_derivative_matrix[5,:,:] = current_covariance_derivative_matrix[5,:,:] + discretisation_time_step*(derivative_of_variance_wrt_basal_transcription)

        # translation rate
        instant_jacobian_derivative_wrt_translation_rate = [0.0 0.0;1.0 0.0]
        delayed_jacobian_derivative_wrt_translation_rate = [0.0 basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[6,2]; 0.0 0.0]
        instant_noise_derivative_wrt_translation_rate = [mRNA_degradation_rate*current_mean_derivative[6,1] 0.0;
                                                         0.0 translation_rate*current_mean_derivative[6,1] + protein_degradation_rate*current_mean_derivative[6,2] + current_mean[1]]

        delayed_noise_derivative_wrt_translation_rate = [basal_transcription_rate*hill_function_derivative_value*past_mean_derivative[6,2] 0.0;
                                                         0.0 0.0]

        derivative_of_variance_wrt_translation_rate = ( common_state_space_variance_derivative_element[6,:,:] +
                                                        instant_jacobian_derivative_wrt_translation_rate*current_covariance_matrix +
                                                        current_covariance_matrix*transpose(instant_jacobian_derivative_wrt_translation_rate) +
                                                        delayed_jacobian_derivative_wrt_translation_rate*covariance_matrix_past_to_now +
                                                        covariance_matrix_now_to_past*transpose(delayed_jacobian_derivative_wrt_translation_rate) +
                                                        instant_noise_derivative_wrt_translation_rate + delayed_noise_derivative_wrt_translation_rate )

        next_covariance_derivative_matrix[6,:,:] = current_covariance_derivative_matrix[6,:,:] + discretisation_time_step*(derivative_of_variance_wrt_translation_rate)

        # transcriptional delay
        # instant_jacobian_derivative_wrt_transcription_delay = 0
        delayed_jacobian_derivative_wrt_transcription_delay = [0.0 basal_transcription_rate*hill_function_second_derivative_value*past_mean_derivative[7,2]; 0.0 0.0]
        instant_noise_derivative_wrt_transcription_delay = [mRNA_degradation_rate*current_mean_derivative[7,1] 0.0;
                                                            0.0 translation_rate*current_mean_derivative[7,1] + protein_degradation_rate*current_mean_derivative[7,2]]

        delayed_noise_derivative_wrt_transcription_delay = [basal_transcription_rate*hill_function_derivative_value*past_mean_derivative[7,2] 0.0;
                                                            0.0 0.0]

        derivative_of_variance_wrt_transcription_delay = ( common_state_space_variance_derivative_element[7,:,:] +
                                                           delayed_jacobian_derivative_wrt_transcription_delay*covariance_matrix_past_to_now +
                                                           covariance_matrix_now_to_past*transpose(delayed_jacobian_derivative_wrt_transcription_delay) +
                                                           instant_noise_derivative_wrt_transcription_delay + delayed_noise_derivative_wrt_transcription_delay )

        next_covariance_derivative_matrix[7,:,:] = current_covariance_derivative_matrix[7,:,:] + discretisation_time_step*(derivative_of_variance_wrt_transcription_delay)

        for parameter_index in 1:7
            state_space_variance_derivative[parameter_index,[next_time_index,
                                                              total_number_of_states+next_time_index],
                                                             [next_time_index,
                                                              total_number_of_states+next_time_index]] = next_covariance_derivative_matrix[parameter_index,:,:]
        end # for

        ## now we need to update the cross correlations, d_P(s,t)/d_theta in the Calderazzo paper
        # the range needs to include t, since we want to propagate d_P(t,t)/d_theta into d_P(t,t+Deltat)/d_theta
        for intermediate_time_index in past_time_index:current_time_index
            # This corresponds to d_P(s,t)/d_theta in the Calderazzo paper
            # for loops instead of np.ix_-like indexing
            for parameter_index in 1:7
                covariance_matrix_derivative_intermediate_to_current[parameter_index,:,:] = state_space_variance_derivative[parameter_index,[intermediate_time_index,
                                                                                                                                             total_number_of_states+intermediate_time_index],
                                                                                                                                            [current_time_index,
                                                                                                                                             total_number_of_states+current_time_index]]
                 # This corresponds to d_P(s,t-tau)/d_theta
                 covariance_matrix_derivative_intermediate_to_past[parameter_index,:,:] = state_space_variance_derivative[parameter_index,[intermediate_time_index,
                                                                                                                                           total_number_of_states+intermediate_time_index],
                                                                                                                                           [past_time_index,
                                                                                                                                           total_number_of_states+past_time_index]]
            end #for

            # Again, this derivative is slightly different for each parameter, meaning it's difficult to
            # code this part with a loop. For each parameter we divide it in to it's constituent parts. There is one
            # main part in common for every derivative which is defined here as common_intermediate_state_space_variance_derivative_element
            for parameter_index in 1:7
                common_intermediate_state_space_variance_derivative_element[parameter_index,:,:] = ( covariance_matrix_derivative_intermediate_to_current[parameter_index,:,:]*
                                                                                                        instant_jacobian_transpose +
                                                                                                  covariance_matrix_derivative_intermediate_to_past[parameter_index,:,:]*
                                                                                                        delayed_jacobian_transpose )
            end # for
            # repression threshold
            derivative_of_intermediate_variance_wrt_repression_threshold = ( common_intermediate_state_space_variance_derivative_element[1,:,:] +
                                                                             covariance_matrix_intermediate_to_past*delayed_jacobian_derivative_wrt_repression_transpose )

            covariance_matrix_derivative_intermediate_to_next[1,:,:] = covariance_matrix_derivative_intermediate_to_current[1,:,:] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_repression_threshold)
            # if next_time_index == 31 && intermediate_time_index+total_number_of_states == 71
            #     println(covariance_matrix_derivative_intermediate_to_next[1,:,:])
            # end #if
            # hill coefficient
            derivative_of_intermediate_variance_wrt_hill_coefficient = ( common_intermediate_state_space_variance_derivative_element[2,:,:] +
                                                                         covariance_matrix_intermediate_to_past*transpose(delayed_jacobian_derivative_wrt_hill_coefficient) )

            covariance_matrix_derivative_intermediate_to_next[2,:,:] = covariance_matrix_derivative_intermediate_to_current[2,:,:] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_hill_coefficient)

            # mRNA degradation rate
            derivative_of_intermediate_variance_wrt_mRNA_degradation = ( common_intermediate_state_space_variance_derivative_element[3,:,:] +
                                                                         covariance_matrix_intermediate_to_current*transpose(instant_jacobian_derivative_wrt_mRNA_degradation) +
                                                                         covariance_matrix_intermediate_to_past*transpose(delayed_jacobian_derivative_wrt_mRNA_degradation) )

            covariance_matrix_derivative_intermediate_to_next[3,:,:] = covariance_matrix_derivative_intermediate_to_current[3,:,:] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_mRNA_degradation)

            # protein degradation rate
            derivative_of_intermediate_variance_wrt_protein_degradation = ( common_intermediate_state_space_variance_derivative_element[4,:,:] +
                                                                            covariance_matrix_intermediate_to_current*transpose(instant_jacobian_derivative_wrt_protein_degradation) +
                                                                            covariance_matrix_intermediate_to_past*transpose(delayed_jacobian_derivative_wrt_protein_degradation) )

            covariance_matrix_derivative_intermediate_to_next[4,:,:] = covariance_matrix_derivative_intermediate_to_current[4,:,:] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_protein_degradation)

            # basal transcription rate
            derivative_of_intermediate_variance_wrt_basal_transcription = ( common_intermediate_state_space_variance_derivative_element[5,:,:] +
                                                                            covariance_matrix_intermediate_to_past*transpose(delayed_jacobian_derivative_wrt_basal_transcription) )

            covariance_matrix_derivative_intermediate_to_next[5,:,:] = covariance_matrix_derivative_intermediate_to_current[5,:,:] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_basal_transcription)

            # translation rate
            derivative_of_intermediate_variance_wrt_translation_rate = ( common_intermediate_state_space_variance_derivative_element[6,:,:] +
                                                                         covariance_matrix_intermediate_to_current*transpose(instant_jacobian_derivative_wrt_translation_rate) +
                                                                         covariance_matrix_intermediate_to_past*transpose(delayed_jacobian_derivative_wrt_translation_rate) )

            covariance_matrix_derivative_intermediate_to_next[6,:,:] = covariance_matrix_derivative_intermediate_to_current[6,:,:] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_translation_rate)

            # transcriptional delay
            derivative_of_intermediate_variance_wrt_transcription_delay = ( common_intermediate_state_space_variance_derivative_element[7,:,:] +
                                                                            covariance_matrix_intermediate_to_past*transpose(delayed_jacobian_derivative_wrt_transcription_delay) )

            covariance_matrix_derivative_intermediate_to_next[7,:,:] = covariance_matrix_derivative_intermediate_to_current[7,:,:] + discretisation_time_step*(derivative_of_intermediate_variance_wrt_transcription_delay)

            # Fill in the big matrix
            for parameter_index in 1:7
                # println(size(covariance_matrix_derivative_intermediate_to_next))
                state_space_variance_derivative[parameter_index,[intermediate_time_index,
                                                                 total_number_of_states+intermediate_time_index],
                                                                [next_time_index,
                                                                 total_number_of_states+next_time_index]] = covariance_matrix_derivative_intermediate_to_next[parameter_index,:,:]
                if next_time_index == 31 && intermediate_time_index+total_number_of_states == 71
                    println(state_space_variance_derivative[parameter_index,[intermediate_time_index,
                                                                 total_number_of_states+intermediate_time_index],
                                                                [next_time_index,
                                                                 total_number_of_states+next_time_index]])

                    println(covariance_matrix_derivative_intermediate_to_next[parameter_index,:,:])
                end #if
                # transpose arguments
                state_space_variance_derivative[parameter_index,[next_time_index,
                                                                 total_number_of_states+next_time_index],
                                                                [intermediate_time_index,
                                                                 total_number_of_states+intermediate_time_index]] = covariance_matrix_derivative_intermediate_to_next[parameter_index,:,:]
            end # for
        end # for (intermediate time index)
    end # for (next time index)
    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative
end # function

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
function kalman_update_step(state_space_mean,
                       state_space_variance,
                       state_space_mean_derivative,
                       state_space_variance_derivative,
                       current_observation,
                       time_delay,
                       observation_time_step,
                       measurement_variance)

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
    helper_inverse = 1.0/(dot(observation_transform,predicted_final_covariance_matrix*transpose(observation_transform)) + measurement_variance)

    # This is C in the paper
    adaptation_coefficient = sum(dot.(shortened_covariance_matrix_past_to_final,observation_transform),dims=2)*helper_inverse

    # This is rho*
    updated_stacked_state_space_mean = ( stacked_state_space_mean +
                                         (adaptation_coefficient*(current_observation[2] -
                                                                 dot(observation_transform,predicted_final_state_space_mean))) )
    # ensures the the mean mRNA and Protein are non negative
    updated_stacked_state_space_mean[updated_stacked_state_space_mean.<0] .= 0
    # unstack the rho into two columns, one with mRNA and one with protein
    updated_state_space_mean = hcat(updated_stacked_state_space_mean[1:(discrete_delay+1)],
                                    updated_stacked_state_space_mean[(discrete_delay+2):end])
    # Fill in the updated values
    # funny indexing with 1:3 instead of (1,2) to make numba happy
    state_space_mean[(current_number_of_states-discrete_delay):current_number_of_states,[2,3]] = updated_state_space_mean

    # This is P*
    updated_shortened_covariance_matrix = ( shortened_covariance_matrix -
                                            adaptation_coefficient*observation_transform*shortened_covariance_matrix_final_to_past )
    # ensure that the diagonal entries are non negative
    # np.fill_diagonal(updated_shortened_covariance_matrix,np.maximum(np.diag(updated_shortened_covariance_matrix),0))
    updated_shortened_covariance_matrix[diagm(updated_shortened_covariance_matrix[diagind(updated_shortened_covariance_matrix)].<0)] .= 0

    # Fill in updated values
    # replacing the following line with a loop for numba
    # state_space_variance[all_indices_up_to_delay,
    #                    all_indices_up_to_delay.transpose()] = updated_shortened_covariance_matrix
    state_space_variance[all_indices_up_to_delay,all_indices_up_to_delay] = updated_shortened_covariance_matrix
    #
    # for shortened_row_index, long_row_index in enumerate(all_indices_up_to_delay):
    #     for shortened_column_index, long_column_index in enumerate(all_indices_up_to_delay):
    #         state_space_variance[long_row_index,long_column_index] = updated_shortened_covariance_matrix[shortened_row_index,
    #                                                                                                      shortened_column_index]

    ##########################################
    ## derivative updates
    ##########################################

    # funny indexing with 0:2 instead of (0,1) to make numba happy
    shortened_state_space_mean_derivative = state_space_mean_derivative[(current_number_of_states-discrete_delay):current_number_of_states,:,[1,2]]

    # put protein values underneath mRNA values, to make vector of mean derivatives (d_rho/d_theta)
    # consistent with variance (P)
    stacked_state_space_mean_derivative = zeros(7,2*(discrete_delay+1))

    # this gives us 7 rows (one for each parameter) of mRNA derivative values over time, followed by protein derivative values over time
    for parameter_index in 1:7
        stacked_state_space_mean_derivative[parameter_index,:] = vcat(shortened_state_space_mean_derivative[:,parameter_index,1],
                                                                      shortened_state_space_mean_derivative[:,parameter_index,2])
    end # for
    # funny indexing with 0:2 instead of (0,1) to make numba happy (this gives a 7 x 2 numpy array)
    predicted_final_state_space_mean_derivative = state_space_mean_derivative[current_number_of_states,:,[1,2]]

    # extract covariance derivative matrix up to delay
    # using for loop indexing for numba
    shortened_covariance_derivative_matrix = state_space_variance_derivative[:,all_indices_up_to_delay,all_indices_up_to_delay]

    # # extract d_P(t+Deltat-delay:t+deltat,t+Deltat)/d_theta, replacing ((discrete_delay),-1) with a splice for numba
    shortened_covariance_derivative_matrix_past_to_final = shortened_covariance_derivative_matrix[:,:,[discrete_delay+1,end]]
    shortened_covariance_derivative_matrix_final_to_past = shortened_covariance_derivative_matrix[:,[discrete_delay+1,end],:]

    # This is the derivative of P(t+Deltat,t+Deltat) in the paper
    predicted_final_covariance_derivative_matrix = state_space_variance_derivative[:,
                                                                                   [current_number_of_states,
                                                                                    total_number_of_states+current_number_of_states],
                                                                                   [current_number_of_states,
                                                                                    total_number_of_states+current_number_of_states]]

    # need derivative of the adaptation_coefficient
    adaptation_coefficient_derivative = zeros(7,length(all_indices_up_to_delay))
    for parameter_index in 1:7
        adaptation_coefficient_derivative[parameter_index,:] = (shortened_covariance_derivative_matrix_past_to_final[parameter_index,:,:]*transpose(observation_transform)*helper_inverse -
                                                             (shortened_covariance_matrix_past_to_final*transpose(observation_transform)*observation_transform*(
                                                             predicted_final_covariance_derivative_matrix[parameter_index,:,:]*transpose(observation_transform))*(helper_inverse^2) ))
    end #for
    test2 = shortened_covariance_matrix_past_to_final*transpose(observation_transform)*observation_transform*(
    predicted_final_covariance_derivative_matrix[3,:,:]*transpose(observation_transform))*(helper_inverse^2)
    # This is d_rho*/d_theta
    updated_stacked_state_space_mean_derivative = zeros(7,2*(discrete_delay+1))
    for parameter_index in 1:7
        updated_stacked_state_space_mean_derivative[parameter_index,:] = ( stacked_state_space_mean_derivative[parameter_index,:] +
                                                                         adaptation_coefficient_derivative[parameter_index,:]*(current_observation[2] -
                                                                         dot(observation_transform,predicted_final_state_space_mean)) -
                                                                         adaptation_coefficient*observation_transform*predicted_final_state_space_mean_derivative[parameter_index,:])
    end #for
    # unstack the rho into two columns, one with mRNA and one with protein
    updated_state_space_mean_derivative = zeros((discrete_delay+1),7,2)
    for parameter_index in 1:7
        updated_state_space_mean_derivative[:,parameter_index,:] = hcat(updated_stacked_state_space_mean_derivative[parameter_index,1:(discrete_delay+1)],
                                                                        updated_stacked_state_space_mean_derivative[parameter_index,(discrete_delay+2):end])
    end #for
    # Fill in the updated values
    state_space_mean_derivative[(current_number_of_states-discrete_delay):current_number_of_states,:,1:2] = updated_state_space_mean_derivative

    # This is d_P*/d_theta
    updated_shortened_covariance_derivative_matrix = zeros(7,length(all_indices_up_to_delay),length(all_indices_up_to_delay))
    for parameter_index in 1:7
        updated_shortened_covariance_derivative_matrix[parameter_index,:,:] = ( shortened_covariance_derivative_matrix[parameter_index,:,:] -
                                                                                adaptation_coefficient_derivative[parameter_index,:]*observation_transform*(
                                                                                shortened_covariance_matrix_final_to_past) -
                                                                                adaptation_coefficient*observation_transform*shortened_covariance_derivative_matrix_final_to_past[parameter_index,:,:] )
    end #for
    # Fill in updated values
    state_space_variance_derivative[:,all_indices_up_to_delay,all_indices_up_to_delay] = updated_shortened_covariance_derivative_matrix

    return state_space_mean, state_space_variance, state_space_mean_derivative, state_space_variance_derivative
end # function
