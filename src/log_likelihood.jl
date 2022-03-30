"""
Calculates the negative log of the likelihood of our data given the paramters, using the Kalman filter. It uses the
predicted_observation_distributions from the kalman_filter function. The entries of this array in the second and
third columns represent the probability of the future observation of mRNA and Protein respectively, given our current knowledge.

Parameters
----------

- `protein_at_observations::`: Observed protein. The dimension is n x 2, where n is the
    number of observation time points. For each data set, the first column is the time,
    and the second column is the observed protein copy number at that time.

- `model_parameters::ModelParameters`: An array containing the model parameters in the
    following order: P₀, h, μₘ,
    μₚ, αₘ, αₚ,
    transcription_delay.

measurement_variance : float.
    The variance in our measurement. This is given by Sigma_e in Calderazzo et. al. (2018).

Returns
-------

log_likelihood : float.
    The log of the likelihood of the data.
"""
function calculate_log_likelihood_at_parameter_point(
    protein_at_observations::AbstractArray{<:Real},
    model_parameters::ModelParameters,
    measurement_variance::Real,
)
    size(protein_at_observations, 2) == 2 ||
        throw(ArgumentError("observation matrix must be N × 2"))

    if any(
        [
            getfield(model_parameters, fieldname) for
            fieldname in fieldnames(ModelParameters)
        ] .<= 0,
    )
        return -Inf
    end

    state_space_and_distributions =
        kalman_filter(protein_at_observations, model_parameters, measurement_variance)
    observations = protein_at_observations[:, 2]
    return sum(logpdf.(state_space_and_distributions.distributions, observations))
end
