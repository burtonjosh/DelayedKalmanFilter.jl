"""
Function which defines the HES5 ode system
"""
function hes_ode!(du,u,p::ModelParameters,t)
   du[1] = p.basal_transcription_rate*hill_function(u[2],
      p.repression_threshold,
      p.hill_coefficient) - p.mRNA_degradation_rate*u[1]
   du[2] = p.translation_rate*u[1] - p.protein_degradation_rate*u[2]
end

"""
Calculate the Hill function for a given protein molecule number, repression threshold, and hill coefficient.

# Arguments

- `protein::Real`

- `repression_threshold::Real`

- `hill_coefficient::Real`
"""
function hill_function(protein,repression_threshold,hill_coefficient)
   return 1/(1 + (protein/repression_threshold)^hill_coefficient)
end

"""
Calculate the steady state of the Hes5 ODE system, for a specific set of parameters.

# Arguments

- `repression_threshold::Float64`

- `hill_coefficient::Float64`

- `mRNA_degradation_rate::Float64`

- `protein_degradation_rate::Float64`

- `basal_transcription_rate::Float64`

- `translation_rate::Float64`

# Returns

- `steady_state_solution::Array{Float64,1}`: A 2-element array, giving the steady state for the mRNA and protein respectively.
"""
function calculate_steady_state_of_ode(
   model_params::ModelParameters;
   initial_guess = [40.0,50000.0]
   )
   prob = SteadyStateProblem(hes_ode!, initial_guess, model_params)
   return solve(prob,SSRootfind())
end
