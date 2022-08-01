```@meta
CurrentModule = DelayedKalmanFilter
```

# DelayedKalmanFilter.jl: Approximate likelihood for delayed SDE's

This package calculates an approximate likelihood function for delayed stochastic differential equations[^Calderazzo2019][^Burton2021].

[^Calderazzo2019]: Silvia Calderazzo, Marco Brancaccio, and Bärbel Finkenstädt (2018). Filtering and inference for stochastic oscillators with distributed delays. Bioinformatics, Volume 35, Issue 8, 15 April 2019. [https://doi.org/10.1093/bioinformatics/bty782](https://doi.org/10.1093/bioinformatics/bty782).

[^Burton2021]: Joshua Burton, Cerys S. Manning Magnus Rattray, Nancy Papalopulu, and Jochen Kursawe (2021). Inferring kinetic parameters of oscillatory gene regulation from single cell time-series data. J. R. Soc. Interface, Volume 18, Issue 182, 29 September 2021. [http://doi.org/10.1098/rsif.2021.0393](http://doi.org/10.1098/rsif.2021.0393).