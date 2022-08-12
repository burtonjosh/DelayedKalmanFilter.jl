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

- `protein::Real`

- `P₀::Real`

- `h::Real`
"""
function hill_function(protein, P₀, h)
    return (P₀^h) / (protein^h + P₀^h)
end

function ∂hill∂p(protein, P₀, h)
    return ForwardDiff.derivative(protein -> hill_function(protein, P₀, h), protein)
end

"""
Calculate the steady state of the Hes5 ODE system, for a specific set of parameters.

# Arguments

- `P₀::Float64`

- `h::Float64`

- `μₘ::Float64`

- `μₚ::Float64`

- `αₘ::Float64`

- `αₚ::Float64`

# Returns

- `steady_state_solution::Array{Float64,1}`: A 2-element array, giving the steady state for the mRNA and protein respectively.
"""
function calculate_steady_state_of_ode(
    model_params;
    initial_guess = [40.0, 50000.0],
)
    prob = SteadyStateProblem(hes_ode!, initial_guess, model_params)
    return solve(prob, SSRootfind())
end
