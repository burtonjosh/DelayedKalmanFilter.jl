"""
Function which defines the HES5 ode system
"""
function hes_ode!(du, u, p, t)
  du[1] = p[5] * hill_function(u[2], p[1], p[2]) - p[3] * u[1]
  du[2] = p[6] * u[1] - p[4] * u[2]
end

"""
Calculate the Hill function for a given protein molecule number, repression threshold, and hill coefficient.

# Arguments

- `protein::AbstractFloat`

- `P₀::AbstractFloat`

- `h::AbstractFloat`
"""
function hill_function(protein, P₀, h)
  return (P₀^h) / (protein^h + P₀^h)
end

"""
The partial derivative of the Hill function with respect to the protein molecule number.

# Arguments

- `protein::AbstractFloat`

- `P₀::AbstractFloat`

- `h::AbstractFloat`
"""
function ∂hill∂p(protein, P₀, h)
  return -(h * P₀^h * protein^(h - 1)) / (protein^h + P₀^h)^2
end

"""
Calculate the steady state of the Hes5 ODE system, for a specific set of parameters.

# Arguments

- `P₀::AbstractFloat`

- `h::AbstractFloat`

- `μₘ::AbstractFloat`

- `μₚ::AbstractFloat`

- `αₘ::AbstractFloat`

- `αₚ::AbstractFloat`

# Returns

- `steady_state_solution::Array{AbstractFloat,1}`: A 2-element array, giving the steady state for the mRNA and protein respectively.
"""
function calculate_steady_state_of_ode(model_params; initial_guess=[40.0, 50000.0])
  prob = SteadyStateProblem(hes_ode!, initial_guess, model_params)
  nl_prob = NonlinearProblem(prob)
  return solve(nl_prob, DynamicSS(Tsit5())).u
  # return solve(prob, SSRootfind()).u
end
