module DelayedKalmanFilter

using DelimitedFiles
using DifferentialEquations
using Distributions
using ForwardDiff
using Interpolations
using LinearAlgebra

include("kalman_filter_alg.jl")
include("log_likelihood.jl")
include("hes5_ode_steady_state.jl")

import Base.+  
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)
+(f::SciMLBase.ODESolution, g::Function) = (x...) -> f(x...) + g(x...)
+(f::OrdinaryDiffEq.ODECompositeSolution, g::Function) = (x...) -> f(x...) + g(x...)

export kalman_filter
export calculate_log_likelihood_at_parameter_point

end # module