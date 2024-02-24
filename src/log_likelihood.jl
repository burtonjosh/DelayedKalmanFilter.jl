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
function calculate_log_likelihood(data, params, measurement_variance; ode_solver = Tsit5())
  size(data, 2) == 2 || throw(ArgumentError("observation matrix must be N × 2"))
  all(params .>= 0.0) || throw(ErrorException("all model parameters must be positive"))
  try
    _, distributions = kalman_filter(data, params, measurement_variance; ode_solver)
    logpdf(MvNormal(distributions[:, 1], diagm(distributions[:, 2])), data[:, 2])
  catch e
    -Inf
  end
end
