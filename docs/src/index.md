```@meta
CurrentModule = DelayedKalmanFilter
```

# DelayedKalmanFilter.jl: Approximate likelihood for delayed SDE's

This package calculates an approximate likelihood function for the delayed stochastic differential equations

```math
\begin{aligned}
\frac{\text{d}m}{\text{d}t} &= \alpha_mf(p(t-\tau)) - \mu_mm + \sqrt{\alpha_mf(p(t-\tau)) + \mu_mm} \xi_m,\\[0.6em]
\frac{\text{d}p}{\text{d}t} &= \alpha_pm - \mu_pp + \sqrt{\alpha_pm + \mu_pp} \xi_p,
\end{aligned}
```

with time-series data ``y_t`` defined under the observation process

```math
y_t = F x_t + \epsilon_t,
```

where ``x_t = [m(t), p(t)]^T``, ``F`` is a ``1 \times 2`` matrix, ``\epsilon_t \sim\mathcal{N}(0,\Sigma_\epsilon)`` and ``\Sigma_\epsilon \in \mathbb{R}``.

The function ``f`` is a repressive Hill function,

```math
f(p(t-\tau)) = \frac{1}{1 + [p(t-\tau)/P_0]^h},
```

and ``\xi_m`` and ``\xi_p`` are temporally uncorrelated, statistically independent Gaussian white noises.

Using a delay adjusted extended Kalman filter, we seek to define an approximate likelihood function

```math
\mathcal{L}(\boldsymbol{\theta}\mid\mathbf{y}),
```

where ``\boldsymbol{\theta}`` defines the parameters of the system.[^Calderazzo2019][^Burton2021].

With this function we can utilise either maximum likelihood estimation to obtain a point estimate

```math
\hat{\boldsymbol{\theta}} = \underset{\boldsymbol{\theta}}{\arg\max}\mathcal{L}(\boldsymbol{\theta}\mid\mathbf{y}),
```

or Markov chain Monte Carlo (MCMC) methods to estimate a posterior distribution for the parameters of our model. See [Parameter Estimation](@ref) for details.

[^Calderazzo2019]: Silvia Calderazzo, Marco Brancaccio, and Bärbel Finkenstädt (2018). Filtering and inference for stochastic oscillators with distributed delays. Bioinformatics, Volume 35, Issue 8, 15 April 2019. [https://doi.org/10.1093/bioinformatics/bty782](https://doi.org/10.1093/bioinformatics/bty782).

[^Burton2021]: Joshua Burton, Cerys S. Manning Magnus Rattray, Nancy Papalopulu, and Jochen Kursawe (2021). Inferring kinetic parameters of oscillatory gene regulation from single cell time-series data. J. R. Soc. Interface, Volume 18, Issue 182, 29 September 2021. [http://doi.org/10.1098/rsif.2021.0393](http://doi.org/10.1098/rsif.2021.0393).
