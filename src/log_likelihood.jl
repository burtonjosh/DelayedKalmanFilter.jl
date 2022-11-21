"""
Calculates the log likelihood of our data given the parameters, using a delay adjusted extended Kalman filter. It uses the
`predicted_observation_distributions` from the `kalman_filter` function. The entries of this array in the second and
third columns represent the probability of the future observation of mRNA and protein respectively, given our current knowledge.

Parameters
----------

- `protein_at_observations::Matrix{<:AbstractFloat}`: Observed protein. The dimension is N x 2, where N is the
    number of observation time points. For each data set, the first column is the time,
    and the second column is the observed protein copy number at that time.

- `model_parameters::Vector{<:AbstractFloat}`: An array containing the model parameters in the
    following order: P₀, h, μₘ, μₚ, αₘ, αₚ, τ.

- `measurement_variance::Float`: The variance in our measurement. This is given by ``Σ_ϵ`` Sigma_e in Calderazzo et. al. (2018).

Returns
-------

- `log_likelihood::AbstractFloat`.
"""
function calculate_log_likelihood_at_parameter_point(
    protein_at_observations::Matrix{<:AbstractFloat},
    model_parameters::Vector{<:AbstractFloat},
    measurement_variance::AbstractFloat,
)
    size(protein_at_observations, 2) == 2 ||
        throw(ArgumentError("observation matrix must be N × 2"))

    @assert all(model_parameters .>= 0.0) "all model parameters must be positive"

    _, distributions =
        kalman_filter(protein_at_observations, model_parameters, measurement_variance)
    observations = protein_at_observations[:, 2]

    return sum(logpdf.(Normal.(distributions[:,1],sqrt.(distributions[:,2])), observations))
end