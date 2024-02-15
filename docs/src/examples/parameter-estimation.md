# Parameter Estimation

Some intro into doing parameter estimation

Let's use the same data set from the [Tutorial](@ref):

```@example parameter-estimation
using Plots, Random
Random.seed!(25)

begin
    black = RGBA{Float64}(colorant"#000000")
    TolVibrantOrange = RGBA{Float64}(colorant"#EE7733")
    TolVibrantBlue = RGBA{Float64}(colorant"#0077BB")
    TolVibrantMagenta = RGBA{Float64}(colorant"#EE3377")
end;

using DelayedKalmanFilter, StochasticDelayDiffEq, Statistics

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
plot(
    times,
    unobserved_data[2,:],
    label="Unobserved",
    linewidth=2,
    color=TolVibrantBlue)

# observed protein
scatter!(
    times,
    protein_observations[:,2],
    label="Observed",
    color=TolVibrantOrange
)

plot!(xlabel="Time (minutes)", ylabel="Protein molecule number")
```

Imagine we only have the data, and do not know the value of all the parameters of our model that generated this data. How can we determine their value?

Let's first assume we know all values but one, in this case ``\tau``. A very simple approach would be to calculate the likelihood for a range of values for each parameter, and choose the most likely parameter value in this way. For example:

```@example parameter-estimation
time_delays = LinRange(1., 45., 50)
many_p = [[3407.99, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, i] for i in time_delays]
ll_delays = [calculate_log_likelihood_at_parameter_point(protein_observations, p, measurement_std^2) for p in many_p]

plot(time_delays, ll_delays)
plot!(xlabel = "τ", ylabel = "log-likelihood")
```

As the number of parameters we want to infer increases, this approach becomes less and less viable. We can instead make use of some more advanced algorithm's for maximum likelihood estimation.

## Model definition

Something about probabilistic models, Turing allows us to define and work with them.

```@example parameter-estimation
using Turing, LinearAlgebra

@model function infer_repression(data, times, repression_mean, measurement_variance)
    P₀ ~ truncated(Normal(repression_mean, 500^2); lower=100., upper=repression_mean*2)
    
    _, distributions = kalman_filter(
        hcat(times, data),
        [P₀, 5.17, log(2)/30, log(2)/90, 15.86, 1.27, 30.],
        measurement_variance
    )
    data ~ MvNormal(distributions[:,1], diagm(distributions[:,2]))
end

model = infer_repression(
    protein_observations[:,2],
    protein_observations[:,1],
    mean(protein_observations[:,2]),
    measurement_std^2
)
```

## Maximum likelihood estimation

using [`Optimization.jl`](http://optimization.sciml.ai/stable/) we can take a Turing model and perform MLE.

```@example parameter-estimation
using Optim
mle_estimate = optimize(model, MLE())
```

## Pathfinder and HMC

Initialise with Pathfinder, do some sampling with NUTS(0.8).
