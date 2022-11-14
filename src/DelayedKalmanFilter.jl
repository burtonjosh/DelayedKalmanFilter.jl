module DelayedKalmanFilter

using DataStructures
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

function +(f::SolutionObject, g::SolutionObject)
    SolutionObject(f.at_time + g.at_time, f.tspan)
end

export kalman_filter
export calculate_log_likelihood_at_parameter_point

end # module