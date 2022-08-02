```@setup tutorial
using Plots
```

# Tutorial

The package exports two functions, [`kalman_filter`](@ref) and [`calculate_log_likelihood_at_parameter_point`](@ref).

First we can simulate some observations. To do this, we define a stochastic delay differential equation problem using 

```@example tutorial
using DelayedKalmanFilter, DifferentialEquations, StochasticDelayDiffEq, Plots

function hes_model_drift(du,u,h,p,t)
    P₀, n, μₘ, μₚ, αₘ, αₚ, τ = p
    du[1] = hillr(h(p,t-τ;idxs=2),αₘ,P₀,n) - μₘ*u[1]
    du[2] = αₚ*u[1] - μₚ*u[2]
end

function hes_model_noise(du,u,h,p,t)
    P₀, n, μₘ, μₚ, αₘ, αₚ, τ = p
    du[1] = sqrt(max(0.,hillr(h(p,t-τ;idxs=2),αₘ,P₀,n) + μₘ*u[1]))
    du[2] = sqrt(max(0,αₚ*u[1] + μₚ*u[2]))
end

h(p, t; idxs::Int) = 1.0;

p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.];
tspan=(0.,1720.);

prob = SDDEProblem(hes_model_drift, hes_model_noise, [30.,500.], h, tspan, p; saveat=10);
sol = solve(prob,RKMilCommute());

unobserved_data = Array(sol)[:,50:100];

protein_observations = unobserved_data[2,:] + 0.1*mean(unobserved_data[2,:])*randn(length(unobserved_data[2,:]));

scatter(unobserved_data[2,:],label="unobserved")
scatter!(observed_data,label="observed",xlabel="Time",ylabel="Protein")
```