# this uses catalyst reaction networks -- probably not optimised for performance
# function calculate_steady_state_of_ode(repression_threshold,
#                                        hill_coefficient,
#                                        mRNA_degradation_rate,
#                                        protein_degradation_rate,
#                                        basal_transcription_rate,
#                                        translation_rate)
#    # define Hes5 reaction network
#    hes5_network = @reaction_network begin
#       hillr(protein,basal_transcription_rate,repression_threshold,hill_coefficient), ∅ --> mRNA
#       mRNA_degradation_rate, mRNA → ∅
#       protein_degradation_rate, protein → ∅
#       translation_rate, mRNA → mRNA + protein
#    end repression_threshold hill_coefficient mRNA_degradation_rate protein_degradation_rate basal_transcription_rate translation_rate
#
#    # parameters [α,K,n,δ,γ,β,μ]
#    parameters = (repression_threshold,
#                  hill_coefficient,
#                  mRNA_degradation_rate,
#                  protein_degradation_rate,
#                  basal_transcription_rate,
#                  translation_rate)
#    # initial condition [m,p]
#    u₀ = [40.,50000.]
#    tspan = (0.0,100.0)
#    # create the ODEProblem we want to solve
#    prob = ODEProblem(hes5_network, u₀,tspan,parameters)
#    sol = solve(prob)
#
# end #function

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

- `state_space_mean::Array{Float64,2}`: An array of dimension n x 3, where n is the number of inferred time points.
    The first column is time, the second column is the mean mRNA, and the third
    column is the mean protein. Time points are generated every minute
"""
function calculate_steady_state_of_ode(repression_threshold,
                                       hill_coefficient,
                                       mRNA_degradation_rate,
                                       protein_degradation_rate,
                                       basal_transcription_rate,
                                       translation_rate)

   hill_function(protein,repression_threshold,hill_coefficient) = 1/(1 + (protein/repression_threshold)^hill_coefficient)

   function hes_ode!(du,u,p,t)
      du[1] = p[5]*hill_function(u[2],p[1],p[2]) - p[3]*u[1]
      du[2] = p[6]*u[1] - p[4]*u[2]
   end #function

   parameters = (repression_threshold,
                 hill_coefficient,
                 mRNA_degradation_rate,
                 protein_degradation_rate,
                 basal_transcription_rate,
                 translation_rate)

   # initial condition [m,p]
   initial_guess = [40.,50000.]
   # create the SteadyStateProblem we want to solve
   prob = SteadyStateProblem(hes_ode!, initial_guess, parameters)
   # solve the problem
   steady_state_solution = solve(prob,SSRootfind())
end #function
