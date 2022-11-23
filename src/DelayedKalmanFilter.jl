module DelayedKalmanFilter

using DataStructures
using DelimitedFiles
using DifferentialEquations
using Distributions
using Interpolations
using LinearAlgebra
using StaticArrays

include("kalman_filter_alg.jl")
include("log_likelihood.jl")
include("hes5_ode_steady_state.jl")
include("check_kalman_filter.jl")

import Base.+
+(f::Function, g::Function) = (x...) -> f(x...) + g(x...)
+(f::SciMLBase.ODESolution, g::Function) = (x...) -> f(x...) + g(x...)
+(f::OrdinaryDiffEq.ODECompositeSolution, g::Function) = (x...) -> f(x...) + g(x...)

export kalman_filter, check_filter
export get_mean_at_time, get_variance_at_time
export calculate_log_likelihood_at_parameter_point

end
