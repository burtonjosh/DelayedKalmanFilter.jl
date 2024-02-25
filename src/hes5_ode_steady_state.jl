"""
Function which defines the model ode system
"""
function model_ode!(du, u, p, t)
  du[1] = -p[1] * u[1]
end

function calculate_steady_state_of_ode(model_params; initial_guess = [1.0])
  prob = SteadyStateProblem(model_ode!, initial_guess, model_params)
  nl_prob = NonlinearProblem(prob)
  solve(nl_prob, DynamicSS(Tsit5())).u
end
