# Tutorial

The package exports two functions, [`kalman_filter`](@ref) and [`calculate_log_likelihood_at_parameter_point`](@ref).

Let's load some colours from [Paul Tol](https://personal.sron.nl/~pault/) and the [`Plots`](https://docs.juliaplots.org/stable/) package for plotting.

```@example tutorial
using Plots

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;
```
First we can simulate some observations. To do this, we define a stochastic delay differential equation problem using the [StochasticDelayDiffEq.jl](https://github.com/SciML/StochasticDelayDiffEq.jl) package.

```@example tutorial
using DelayedKalmanFilter, DifferentialEquations, StochasticDelayDiffEq, Statistics

hillr(X, v, K, n) = v * (K^n) / (X^n + K^n)

function hes_model_drift(du,u,h,p,t)
    P₀, n, μₘ, μₚ, αₘ, αₚ, τ = p
    du[1] = hillr(h(p,t-τ;idxs=2),αₘ,P₀,n) - μₘ*u[1]
    du[2] = αₚ*u[1] - μₚ*u[2]
end

function hes_model_noise(du,u,h,p,t)
    P₀, n, μₘ, μₚ, αₘ, αₚ, τ = p
    du[1] = sqrt(max(0.,hillr(h(p,t-τ;idxs=2),αₘ,P₀,n) + μₘ*u[1]))
    du[2] = sqrt(max(0.,αₚ*u[1] + μₚ*u[2]))
end

h(p, t; idxs::Int) = 1.0;
```
We use a chemical Langevin equation to simulate our SDDE system, as can be seen from the definition of the noise function[^Gillespie2000].

[^Gillespie2000]: Daniel T. Gillespie (2000). The chemical Langevin equation. J. Chem. Phys., Volume 113, Issue 297, 21 June 2000. [https://doi.org/10.1063/1.481811](https://doi.org/10.1063/1.481811).

We choose a specific set of parameters and solve the [`SciMLBase.SDDEProblem`](https://diffeq.sciml.ai/stable/types/sdde_types/#SciMLBase.SDDEProblem), then plot the protein from `unobserved_data`, as well as the noisy `protein_observations`.

```@example tutorial
p = [3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.];
tspan=(0.,1720.);

prob = SDDEProblem(hes_model_drift, hes_model_noise, [30.,500.], h, tspan, p; saveat=10);
sol = solve(prob,RKMilCommute());

unobserved_data = Array(sol)[:,100:end];
measurement_std = 0.1*mean(unobserved_data[2,:])

protein = unobserved_data[2,:] + 
    measurement_std*randn(length(unobserved_data[2,:]));

times = 0:10:730
protein_observations = hcat(times,protein)

# unobserved protein
plot(times, unobserved_data[2,:],
    label="Unobserved",
    xlabel="Time (mins)",
    ylabel="Protein (molecule number)",
    linewidth=2,
    color=TolVibrantBlue)
# observed protein
scatter!(times,protein_observations[:,2],
    label="Observed",
    color=TolVibrantOrange)
```

Now we can call [`kalman_filter`](@ref) to get a state space mean and variance for each observation time point:

```@example tutorial
_, _, _, distributions = kalman_filter(protein_observations, p, measurement_std^2);
means = [distributions[i].μ for i in 1:length(distributions)];
stds = [distributions[i].σ for i in 1:length(distributions)];

# unobserved protein
plot(times, unobserved_data[2,:],
    label="Unobserved",
    xlabel="Time (mins)",
    ylabel="Protein (molecule number)",
    linewidth=2,
    color=TolVibrantBlue)

# kalman mean and sd
plot!(times,means,ribbon=stds,fillalpha=.1,
    label="Kalman",
    linewidth=2,
    color=TolVibrantMagenta)

# observed protein
scatter!(times,protein_observations[:,2],
    label="Observed",
    color=TolVibrantOrange)
```

We can also use the convenience function [`calculate_log_likelihood_at_parameter_point`](@ref) to get a log-likelihood value for this specific (ground truth) parameter set.

```@example tutorial
calculate_log_likelihood_at_parameter_point(protein_observations, p, measurement_std^2)
```

Using a different parameter set should give us a smaller value for the likelihood.

```@example tutorial
p_wrong = [2407.57, 3.4, log(2)/30, log(2)/90, 5.6, 21.7, 12.];
calculate_log_likelihood_at_parameter_point(protein_observations, p_wrong, measurement_std^2)
```