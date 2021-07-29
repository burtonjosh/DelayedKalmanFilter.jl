"""
Calculates the negative log of the likelihood of our data given the paramters, using the Kalman filter. It uses the
predicted_observation_distributions from the kalman_filter function. The entries of this array in the second and
third columns represent the probability of the future observation of mRNA and Protein respectively, given our current knowledge.

Parameters
----------

protein_at_observations : numpy array.
    Observed protein. The dimension is m x n x 2, where m is the number of data sets, n is the
    number of observation time points. For each data set, the first column is the time,
    and the second column is the observed protein copy number at that time.

model_parameters : numpy array.
    An array containing the model parameters in the following order:
    repression_threshold, hill_coefficient, mRNA_degradation_rate,
    protein_degradation_rate, basal_transcription_rate, translation_rate,
    transcription_delay.

measurement_variance : float.
    The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

Returns
-------

log_likelihood : float.
    The log of the likelihood of the data.
"""
function calculate_log_likelihood_at_parameter_point(protein_at_observations,model_parameters,measurement_variance)
    if any(model_parameters .< 0)
        return -Inf
    end

    log_likelihood = 0
    _, _, _, _, predicted_observation_distributions, _, _ = kalman_filter(protein_at_observations,
                                                                          model_parameters,
                                                                          measurement_variance)
    observations = protein_at_observations[:,2]
    mean = predicted_observation_distributions[:,2]
    sd = sqrt.(predicted_observation_distributions[:,3])

    for observation_index in 1:length(observations)
        log_likelihood += logpdf(Normal(mean[observation_index],sd[observation_index]),observations[observation_index])
    end #for
    return log_likelihood
end # function


"""
Calculates the log of the likelihood, and the derivative of the negative log likelihood wrt each parameter, of our data given the
paramters, using the Kalman filter. It uses the predicted_observation_distributions, predicted_observation_mean_derivatives, and
predicted_observation_variance_derivatives from the kalman_filter function. It returns the log likelihood as in the
calculate_log_likelihood_at_parameter_point function, and also returns an array of the derivative wrt each parameter.

Parameters
----------

protein_at_observations : numpy array.
    Observed protein. The dimension is n x 2, where n is the number of observation time points.
    The first column is the time, and the second column is the observed protein copy number at
    that time.

model_parameters : numpy array.
    An array containing the moderowl parameters in the following order:
    repression_threshold, hill_coefficient, mRNA_degradation_rate,
    protein_degradation_rate, basal_transcription_rate, translation_rate,
    transcription_delay.

measurement_variance : float.
    The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

Returns
-------

log_likelihood : float.
    The log of the likelihood of the data.

log_likelihood_derivative : numpy array.
    The derivative of the log likelihood of the data, wrt each model parameter
"""
function calculate_log_likelihood_and_derivative_at_parameter_point(protein_at_observations,model_parameters,mean_protein,measurement_variance)
    number_of_parameters = size(model_parameters,1)

    _, _, _, _, predicted_observation_distributions, predicted_observation_mean_derivatives, predicted_observation_variance_derivatives = kalman_filter(protein_at_observations,
                                                                                                                                                        model_parameters,
                                                                                                                                                        measurement_variance)
    # calculate log likelihood as before
    log_likelihood = 0
    number_of_observations = size(protein_at_observations,1)
    observations = protein_at_observations[:,2]
    mean = predicted_observation_distributions[:,2]
    sd = sqrt.(predicted_observation_distributions[:,3])

    for observation_index in 1:length(observations)
        log_likelihood += logpdf(Normal(mean[observation_index],sd[observation_index]),observations[observation_index])
    end #for
    # now for the computation of the derivative of the negative log likelihood. An expression of this can be found
    # at equation (28) in Mbalawata, Särkkä, Haario (2013)
    observation_transform = [0.0 1.0]
    helper_inverse = 1.0./predicted_observation_distributions[:,3]

    log_likelihood_derivative = zeros(number_of_parameters)

    for parameter_index in 1:number_of_parameters
        for time_index in 1:number_of_observations
            log_likelihood_derivative[parameter_index] -= 0.5*((helper_inverse[time_index]*tr(observation_transform*predicted_observation_variance_derivatives[time_index,parameter_index,:,:]*
                                                               transpose(observation_transform))) -
                                                               (dot(helper_inverse[time_index]*transpose(observation_transform*predicted_observation_mean_derivatives[time_index,parameter_index,:,:]),
                                                               (observations[time_index] - mean[time_index]))) -
                                                               (helper_inverse[time_index]^2*(observations[time_index] - mean[time_index])^2*
                                                               dot(observation_transform,predicted_observation_variance_derivatives[time_index,parameter_index,:,:]*
                                                               transpose(observation_transform))) -
                                                               (dot(helper_inverse[time_index]*(observations[time_index] - mean[time_index]),
                                                               observation_transform*predicted_observation_mean_derivatives[time_index,parameter_index,:,:])))
         end #for
     end #for
    return log_likelihood, log_likelihood_derivative
end #function

"""
todo
"""
function log_likelihood_and_derivative_with_prior_and_transformation(protein_at_observations,proposed_position,mean_protein,measurement_variance)
    number_of_parameters = size(proposed_position,1)

    # reparameterise
    reparameterised_proposed_position = copy(proposed_position)
    reparameterised_proposed_position[[3,4,5,6]] .= exp.(reparameterised_proposed_position[[3,4,5,6]])

    # priors
    if any([pdf(Uniform(50,2*mean_protein),reparameterised_proposed_position[1]) == 0,
            pdf(Uniform(2,6),reparameterised_proposed_position[2]) == 0,
            pdf(Uniform(log(2)/150,log(2)/10),reparameterised_proposed_position[3]) == 0,
            pdf(Uniform(log(2)/150,log(2)/10),reparameterised_proposed_position[4]) == 0,
            pdf(Uniform(0.01,120),reparameterised_proposed_position[5]) == 0,
            pdf(Uniform(0.01,40),reparameterised_proposed_position[6]) == 0,
            pdf(Uniform(1,40),reparameterised_proposed_position[7]) == 0])
        return -Inf, zeros(number_of_parameters)
    end

    # quick and dirty clean up (make data start at '0')
    protein_at_observations[:,1] .-= protein_at_observations[1,1]
    log_likelihood, log_likelihood_derivative = calculate_log_likelihood_and_derivative_at_parameter_point(protein_at_observations,
                                                                                                           reparameterised_proposed_position,
                                                                                                           mean_protein,
                                                                                                           measurement_variance)
    for i in [3,4,5,6]
        log_likelihood_derivative[i] = reparameterised_proposed_position[i]*log_likelihood_derivative[i]
    end
    return log_likelihood, log_likelihood_derivative
end
