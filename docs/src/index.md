```@meta
CurrentModule = DelayedKalmanFilter
```

# DelayedKalmanFilter.jl: Approximate likelihood for delayed SDE's

This package calculates an approximate likelihood function for the delayed stochastic differential equations

```math
\begin{aligned}
\frac{\text{d}m}{\text{d}t} &= \alpha_mf(p(t-\tau)) - \mu_mm + \sqrt{\alpha_mf(p(t-\tau)) + \mu_mm}\xi_m,\\[0.6em]
\frac{\text{d}p}{\text{d}t} &= \alpha_pm - \mu_pp + \sqrt{\alpha_pm + \mu_pp}\xi_p,
\end{aligned}
```

with data ``y_t`` defined under the observation process

```math
y_t = F x_t + \\epsilon_t,
```

where ``F`` is a ``1 \\times 2`` matrix, ``\\epsilon_t \\sim\\mathcal{N}(0,\\Sigma_\\epsilon)`` and ``\\Sigma_\\epsilon \\in \\mathbb{R}``.

The function ``f`` is a repressive Hill function,

```math
f(p(t-\tau)) = \frac{1}{1 + [p(t-\tau)/P_0]^h}.
```

Using a delay adjusted extended Kalman filter, we seek to define an approximate likelihood function

```math
pi(\\mathbf{y}\\mid\\boldsymbol{\\theta}),
```

where ``\\boldsymbol{\\theta}`` defines the parameters of the system.[^Calderazzo2019][^Burton2021].

[^Calderazzo2019]: Silvia Calderazzo, Marco Brancaccio, and Bärbel Finkenstädt (2018). Filtering and inference for stochastic oscillators with distributed delays. Bioinformatics, Volume 35, Issue 8, 15 April 2019. [https://doi.org/10.1093/bioinformatics/bty782](https://doi.org/10.1093/bioinformatics/bty782).

[^Burton2021]: Joshua Burton, Cerys S. Manning Magnus Rattray, Nancy Papalopulu, and Jochen Kursawe (2021). Inferring kinetic parameters of oscillatory gene regulation from single cell time-series data. J. R. Soc. Interface, Volume 18, Issue 182, 29 September 2021. [http://doi.org/10.1098/rsif.2021.0393](http://doi.org/10.1098/rsif.2021.0393).