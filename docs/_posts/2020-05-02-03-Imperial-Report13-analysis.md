---
title: "Replication study: Estimating number of inections and impact of NPIs on COVID-19 in European countries (Imperial Report 13)"
author: Tor Erlend Fjelde
draft: true
---

We, i.e. the [TuringLang team](https://turing.ml/dev/team/), are currently exploring cooperation with other researchers in attempt to help with the ongoing crisis. As preparation for this and to get our feet wet, we decided it would be useful to do a replication study of the [Imperial Report 13](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/). We figured it might be useful for the public, in particular other researchers working on the same or similar models, to see the results of this analysis, and thus decided to make it available here.

We want to emphasize that you should look to the original paper rather than this post for developments and analysis of the model. As of right now, we're not going to be making any claims about the validity of the model nor the implications of the results. This post's purpose is only to add tiny bit of validation to the *inference* performed in the paper by obtaining the same results using a different probabilistic programming language (PPL) and to explore whether or not `Turing.jl` can be useful for researchers working on these problems.

The full code for the `Covid19.jl` package you can find [here](https://github.com/TuringLang/Covid19).


# Setup

This is all assuming that you're in the project directory of [`Covid19.jl`](https://github.com/TuringLang/Covid19), a small package where we gather most of our ongoing work.

In the project we use [`DrWatson.jl`](https://github.com/JuliaDynamics/DrWatson.jl) which provides a lot of convenient functionality for, well, working with a project. The below code will activate the `Covid19.jl` project to ensure that we're using correct versions of all the dependencies, i.e. code is reproducible. It's basically just doing `] activate` but with a few bells and whistles that we'll use later.

{% highlight julia %}
using DrWatson
quickactivate(@__DIR__)
{% endhighlight %}

With the project activate, we can import the `Covid19.jl` package:

{% highlight julia %}
# Loading the project (https://github.com/TuringLang/Covid19)
using Covid19
{% endhighlight %}

{% highlight julia %}
# Some other packages we'll need
using Random, Dates, Turing, Bijectors
{% endhighlight %}

And we'll be using the new [multithreading functionality in Julia](https://julialang.org/blog/2019/07/multithreading/) to speed things up, so we need the `Base.Threads` package.

{% highlight julia %}
using Base.Threads
nthreads()
{% endhighlight %}

    4

Here's a summary of the setup:

{% highlight julia %}
using Pkg
Pkg.status()
{% endhighlight %}

    Project Covid19 v0.1.0
    Status `~/Projects/mine/Covid19/Project.toml`
      [dce04be8] ArgCheck v2.0.0
      [c7e460c6] ArgParse v1.1.0
      [131c737c] ArviZ v0.4.1
      [76274a88] Bijectors v0.6.7 #tor/using-bijectors-in-link-and-invlink (https://github.com/TuringLang/Bijectors.jl.git)
      [336ed68f] CSV v0.6.1
      [a93c6f00] DataFrames v0.20.2
      [31c24e10] Distributions v0.23.2
      [ced4e74d] DistributionsAD v0.4.10
      [634d3b9d] DrWatson v1.10.2
      [1a297f60] FillArrays v0.8.7
      [47be7bcc] ORCA v0.3.1
      [f0f68f2c] PlotlyJS v0.13.1
      [91a5bcdd] Plots v1.1.2
      [438e738f] PyCall v1.91.4
      [d330b81b] PyPlot v2.9.0
      [df47a6cb] RData v0.7.1
      [2913bbd2] StatsBase v0.33.0
      [f3b207a7] StatsPlots v0.14.5
      [fce5fe82] Turing v0.11.0 #tor/modelling-temporary (https://github.com/TuringLang/Turing.jl.git)
      [9a3f8284] Random 
      [10745b16] Statistics 

In the Github project you will find a `Manifest.toml`. This means that if you're working directory is the project directory, and you do `julia --project` followed by `] instantiate` you will have *exactly* the same enviroment as we had when performing this analysis.


# Data

As mentioned `DrWatson.jl` provides a lot of convenience functions for working in a project, and one of them is `projectdir(args...)` which will resolve join the `args` to the absolute path of the project directory. In similar fashion it provides a `datadir` method which defaults to `projectdir("data")`. But to remove the possibility of making a type later on when writing `datadir("imperial-report13")`, we'll overload this method for this notebook:

{% highlight julia %}
import DrWatson: datadir

datadir() = projectdir("data", "imperial-report13")
datadir(s...) = projectdir("data", "imperial-report13", s...)
{% endhighlight %}

    datadir (generic function with 2 methods)

To ensure consistency with the original model from the paper (and to stay up-to-date with the changes made), we preprocessed the data using the `base.r` script from the [original repository (#6ee3010)](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096) and store the processed data in a `processed.rds` file. To load this RDS file obtained from the R script, we make use of the RData.jl package which allows us to load the RDS file into a Julia `Dict`.

{% highlight julia %}
using RData
{% endhighlight %}

{% highlight julia %}
rdata_full = load(datadir("processed_new.rds"))
rdata = rdata_full["stan_data"];
{% endhighlight %}

{% highlight julia %}
keys(rdata_full)
{% endhighlight %}

    Base.KeySet for a Dict{String,Int64} with 4 entries. Keys:
      "reported_cases"
      "stan_data"
      "deaths_by_country"
      "dates"

{% highlight julia %}
country_to_dates = d = Dict([(k, rdata_full["dates"][k]) for k in keys(rdata_full["dates"])])
{% endhighlight %}

    Dict{String,Array{Date,1}} with 14 entries:
      "Sweden"         => Date[2020-02-18, 2020-02-19, 2020-02-20, 2020-02-21, 2020…
      "Belgium"        => Date[2020-02-18, 2020-02-19, 2020-02-20, 2020-02-21, 2020…
      "Greece"         => Date[2020-02-19, 2020-02-20, 2020-02-21, 2020-02-22, 2020…
      "Switzerland"    => Date[2020-02-14, 2020-02-15, 2020-02-16, 2020-02-17, 2020…
      "Germany"        => Date[2020-02-15, 2020-02-16, 2020-02-17, 2020-02-18, 2020…
      "United_Kingdom" => Date[2020-02-12, 2020-02-13, 2020-02-14, 2020-02-15, 2020…
      "Denmark"        => Date[2020-02-21, 2020-02-22, 2020-02-23, 2020-02-24, 2020…
      "Norway"         => Date[2020-02-24, 2020-02-25, 2020-02-26, 2020-02-27, 2020…
      "France"         => Date[2020-02-07, 2020-02-08, 2020-02-09, 2020-02-10, 2020…
      "Portugal"       => Date[2020-02-21, 2020-02-22, 2020-02-23, 2020-02-24, 2020…
      "Spain"          => Date[2020-02-09, 2020-02-10, 2020-02-11, 2020-02-12, 2020…
      "Netherlands"    => Date[2020-02-14, 2020-02-15, 2020-02-16, 2020-02-17, 2020…
      "Italy"          => Date[2020-01-27, 2020-01-28, 2020-01-29, 2020-01-30, 2020…
      "Austria"        => Date[2020-02-22, 2020-02-23, 2020-02-24, 2020-02-25, 2020…

Since the data-format is not native Julia there might be some discrepancies in the *types* of the some data fields, and so we need to do some type-conversion of the loaded `rdata`. We also rename a lot of the fields to make it more understandable and for an easier mapping to our implementation of the model. Feel free to skip this snippet.

<details><summary>Data wrangling</summary>

{% highlight julia %}
# Convert some misparsed fields
rdata["N2"] = Int(rdata["N2"]);
rdata["N0"] = Int(rdata["N0"]);

rdata["EpidemicStart"] = Int.(rdata["EpidemicStart"]);

rdata["cases"] = Int.(rdata["cases"]);
rdata["deaths"] = Int.(rdata["deaths"]);

# Stan will fail if these are `nothing` so we make them empty arrays
rdata["x"] = []
rdata["features"] = []

countries = (
  "Denmark",
  "Italy",
  "Germany",
  "Spain",
  "United_Kingdom",
  "France",
  "Norway",
  "Belgium",
  "Austria", 
  "Sweden",
  "Switzerland",
  "Greece",
  "Portugal",
  "Netherlands"
)
num_countries = length(countries)

names_covariates = ("schools_universities", "self_isolating_if_ill", "public_events", "any", "lockdown", "social_distancing_encouraged")
lockdown_index = findfirst(==("lockdown"), names_covariates)


function rename!(d, names::Pair...)
    # check that keys are not yet present before updating `d`
    for k_new in values.(names)
        @assert k_new ∉ keys(d) "$(k_new) already in dictionary"
    end

    for (k_old, k_new) in names
        d[k_new] = pop!(d, k_old)
    end
    return d
end

# `rdata` is a `DictOfVector` so we convert to a simple `Dict` for simplicity
d = Dict([(k, rdata[k]) for k in keys(rdata)]) # `values(df)` and `keys(df)` have different ordering so DON'T do `Dict(keys(df), values(df))`

# Rename some columns
rename!(
    d,
    "f" => "π", "SI" => "serial_intervals", "pop" => "population",
    "M" => "num_countries", "N0" => "num_impute", "N" => "num_obs_countries",
    "N2" => "num_total_days", "EpidemicStart" => "epidemic_start",
    "X" => "covariates", "P" => "num_covariates"
)

# Add some type-information to arrays and replace `-1` with `missing` (as `-1` is supposed to represent, well, missing data)
d["deaths"] = Int.(d["deaths"])
# d["deaths"] = replace(d["deaths"], -1 => missing)
d["deaths"] = collect(eachcol(d["deaths"])) # convert into Array of arrays instead of matrix

d["cases"] = Int.(d["cases"])
# d["cases"] = replace(d["cases"], -1 => missing)
d["cases"] = collect(eachcol(d["cases"])) # convert into Array of arrays instead of matrix

d["num_covariates"] = Int(d["num_covariates"])
d["num_countries"] = Int(d["num_countries"])
d["num_total_days"] = Int(d["num_total_days"])
d["num_impute"] = Int(d["num_impute"])
d["num_obs_countries"] = Int.(d["num_obs_countries"])
d["epidemic_start"] = Int.(d["epidemic_start"])
d["population"] = Int.(d["population"])

d["π"] = collect(eachcol(d["π"])) # convert into Array of arrays instead of matrix

# Convert 3D array into Array{Matrix}
covariates = [rdata["X"][m, :, :] for m = 1:num_countries]

data = (; (k => d[String(k)] for k in [:num_countries, :num_impute, :num_obs_countries, :num_total_days, :cases, :deaths, :π, :epidemic_start, :population, :serial_intervals])...)
data = merge(data, (covariates = covariates, ));

# Can deal with ragged arrays, so we can shave off unobserved data (future) which are just filled with -1
data = merge(
    data,
    (cases = [data.cases[m][1:data.num_obs_countries[m]] for m = 1:data.num_countries],
     deaths = [data.deaths[m][1:data.num_obs_countries[m]] for m = 1:data.num_countries])
);
{% endhighlight %}

</details>

{% highlight julia %}
data.num_countries
{% endhighlight %}

    14

Because it's a bit much to visualize 14 countries at each step, we're going to use UK as an example throughout.

{% highlight julia %}
uk_index = findfirst(==("United_Kingdom"), countries)
{% endhighlight %}

    5

It's worth noting that the data user here is not quite up-to-date for UK because on <span class="timestamp-wrapper"><span class="timestamp">&lt;2020-04-30 to.&gt; </span></span> they updated their *past* numbers by including deaths from care*nursing homes (data source: [[https:/*www.ecdc.europa.eu/en][ECDC]]). Thus if you compare the prediction of the model to real numbers, it's likely that the real numbers will be a bit higher than what the model predicts.


# Model

For a thorough description of the model and the assumptions that have gone into it, we recommend looking at the [original paper](https://www.imperial.ac.uk/mrc-global-infectious-disease-analysis/covid-19/report-13-europe-npi-impact/) or their very nice [techical report from the repository](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096/Technical_description_of_Imperial_COVID_19_Model.pdf). The model described here is the one corresponding to the technical report linked. The link points to the correct commit ID and so should be consistent with this post despite potential changes to the "official" model made in the future.

For the sake of exposition, we present a compact version of the model here:
$$
\begin{align}
  \tau & \sim \mathrm{Exponential}(1 / 0.03) \\
  y_m & \sim \mathrm{Exponential}(\tau) \quad & \text{for} \quad m = 1, \dots, M \\
  \kappa & \sim \mathcal{N}^{ + }(0, 0.5) \\
  \mu_m & \sim \mathcal{N}^{ + }(3.28, \kappa) \quad & \text{for} \quad m = 1, \dots, M \\
  \gamma & \sim \mathcal{N}^{ + }(0, 0.2) \\
  \beta_m & \sim \mathcal{N}(0, \gamma) \quad & \text{for} \quad m = 1, \dots, M \\
  \tilde{\alpha}_k &\sim \mathrm{Gamma}(0.1667, 1) \quad & \text{for} \quad k = 1, \dots, K \\
  \alpha_k &= \tilde{\alpha}_k - \frac{\log(1.05)}{6} \quad & \text{for} \quad  k = 1, \dots, K \\
  R_{t, m} &= \mu_m \exp(- \beta_m x_{k_{\text{ld}}} - \sum_{k=1}^{K} \alpha_k x_k) \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T  \\
  c_{t, m} &= R_{t, m} \bigg(1 - \frac{c_{t - 1, m}}{p_m} \bigg) \sum_{\tau = 1}^{t - 1} c_{\tau, m} s_{t - \tau} \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T \\
  \varepsilon_m^{\text{ifr}} &\sim \mathcal{N}(1, 0.1)^{ + } \quad & \text{for} \quad m = 1, \dots, M \\
  \mathrm{ifr}_m^{ * } &\sim \mathrm{ifr}_m \cdot \varepsilon_m^{\text{ifr}} \quad & \text{for} \quad m = 1, \dots, M \\
  d_{t, m} &= \mathrm{ifr}_m^{ * } \sum_{\tau=1}^{t - 1} c_{\tau, m} \pi_{t - \tau} \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T \\
  \phi  & \sim \mathcal{N}^{ + }(0, 5) \\
  D_{t, m} &\sim \mathrm{NegativeBinomial}(d_{t, m}, \phi) \quad & \text{for} \quad m = 1, \dots, M, \ t = 1, \dots, T 
\end{align}
$$

where

-   \\(\\alpha_k\\) denotes the weights for the k-th intervention/covariate
-   \\(\\beta_m\\) denotes the weight for the `lockdown` intervention (whose index we denote by \\(k_{\\text{ld}}\\))
    -   Note that there is also a \\(\\alpha_{k_{\\text{ld}}}\\) which is shared between all the \\(M\\) countries
    -   In contrast, the \\(\\beta_m\\) weight is local to the country with index \\(m\\)
    -   This is a sort of  way to try and deal with the fact that `lockdown` means different things in different countries, e.g. `lockdown` in UK is much more severe than "lockdown" in Norway.
-   \\(\\mu_m\\) represents the \\(R_0\\) value for country \\(m\\) (i.e. \\(R_t\\) without any interventions)
-   \\(R_{t, m}\\) denotes the **reproduction number** at time \\(t\\) for country \\(m\\)
-   \\(p_{m}\\) denotes the **total/initial population** for country \\(m\\)
-   \\(\\mathrm{ifr}_m\\) denotes the **infection-fatality ratio** for country \\(m\\), and \\(\\mathrm{ifr}_m^{ * }\\) the *adjusted* infection-fatality ratio (see paper)
-   \\(\\varepsilon_m^{\\text{ifr}}\\) denotes the noise for the multiplicative noise for the \\(\\mathrm{ifr}_m^{ * }\\)
-   \\(\\pi\\) denotes the **time from infection to death** and is assumed to be a sum of two independent random times: the incubation period (*infection-to-onset*) and time between onset of symptoms and death (*onset-to-death*):

    $$
    \begin{equation*}
    \pi \sim \mathrm{Gamma}(5.1, 0.86) + \mathrm{Gamma}(18.8, 0.45)
    \end{equation*}
    $$

    where in this case the \\(\\mathrm{Gamma}\\) is parameterized by its mean and coefficient of variation. In the model, this is *precomputed* quantity and note something to that is to be inferred.
-   \\(\\pi_t\\) then denotes a discretized version of the PDF for \\(\\pi\\). The reasoning behind the discretization is that if we assume \\(d_m(t)\\) to be a continuous random variable denoting the death-rate at any time \\(t\\), then it would be given by
    
    $$
    \begin{equation*}
    d_m(t) = \mathrm{ifr}_m^{ * } \int_0^t c_m(\tau) \pi(t - \tau) dt
    \end{equation*}
    $$

    i.e. the convolution of the number of cases observed at time time \\(\\tau\\), \\(c_m(\\tau)\\), and the *probability* of death at prior to time \\(t\\) for the new cases observed at time \\(\\tau\\), \\(\\pi(t - \\tau)\\) (assuming stationarity of \\(\\pi(t)\\)). Thus, \\(c_m(\\tau) \\pi(t - \\tau)\\) can be interpreted as the portion people who got the virus at time \\(\\tau\\) have died at time \\(t\\) (or rather, have died after having the virus for \\(t - \\tau\\) time, with \\(t > \\tau\\)). Discretizing then results in the above model.
-   \\(s_t\\) denotes the **serial intervals**, i.e. the time between successive cases in a chain of transmission, also a precomputed quantity
-   \\(c_{t, m}\\) denotes the **expected daily cases** at time \\(t\\) for country \\(m\\)
-   \\(d_{t, m}\\) denotes the **expected daily deaths** at time \\(t\\) for country \\(m\\)
-   \\(D_{t, m}\\) denotes the **daily deaths** at time \\(t\\) for country \\(m\\) (in our case, this is the **likelihood**)

To see the reasoning for the choices of distributions and parameters for the priors, see the either the paper or the [techical report from the repository](https://github.com/ImperialCollegeLondon/covid19model/tree/6ee3010a58a57cc14a16545ae897ca668b7c9096/Technical_description_of_Imperial_COVID_19_Model.pdf).


## Code

In `Turing.jl`, a "sample"-statement is defined by `x ~ Distribution`. Therefore, if we convert the priors from the model into a block of code that will go into the `Turing.jl` `Model`, we get:

{% highlight julia %}
τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
y ~ filldist(Exponential(τ), num_countries)
ϕ ~ truncated(Normal(0, 5), 0, Inf)
κ ~ truncated(Normal(0, 0.5), 0, Inf)
μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_countries)

α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
α = α_hier .- log(1.05) / 6.

ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_countries)

# lockdown-related
γ ~ truncated(Normal(0, 0.2), 0, Inf)
lockdown ~ filldist(Normal(0, γ), num_countries)
{% endhighlight %}

The only anamoly in the above snippet is the use of `filldist` which is just a way to construct an efficient `MultivariateDistribution` from which we can draw i.i.d. samples from a `UnivariateDistribution` in a vectorized manner.

And the full model is defined:

{% highlight julia %}
@model function model_v2(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    lockdown_index,    # [Int] the index for the `lockdown` covariate in `covariates`
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64}
) where {TV}
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)
    num_countries = length(cases)
    num_obs_countries = length.(cases)

    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # Latent variables
    τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
    y ~ filldist(Exponential(τ), num_countries)
    ϕ ~ truncated(Normal(0, 5), 0, Inf)
    κ ~ truncated(Normal(0, 0.5), 0, Inf)
    μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_countries)

    α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
    α = α_hier .- log(1.05) / 6.

    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_countries)

    # lockdown-related
    γ ~ truncated(Normal(0, 0.2), 0, Inf)
    lockdown ~ filldist(Normal(0, γ), num_countries)

    # Initialization of some quantities
    expected_daily_cases = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]

    # Loops over countries and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_countries
        # country-specific parameters
        π_m = π[m]
        pop_m = population[m]
        expected_daily_cases_m = expected_daily_cases[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rt_m = Rt[m]
        Rt_adj_m = Rt_adj[m]

        last_time_step = last_time_steps[m]

        # Imputation of `num_impute` days
        expected_daily_cases_m[1:num_impute] .= y[m]
        cases_pred_m[1] = zero(cases_pred_m[1])
        cases_pred_m[2:num_impute] .= cumsum(expected_daily_cases_m[1:num_impute - 1])

        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rt_m .= μ[m] * exp.(xs * (-α) + (- lockdown[m]) * xs[:, lockdown_index])

        # adjusts for portion of pop that are susceptible
        Rt_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rt_m[1:num_impute]

        for t = (num_impute + 1):last_time_step
            cases_pred_m[t] = cases_pred_m[t - 1] + expected_daily_cases_m[t - 1]

            Rt_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rt_m[t] # adjusts for portion of pop that are susceptible
            expected_daily_cases_m[t] = Rt_adj_m[t] * sum(expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
        end

        expected_daily_deaths_m[1] = 1e-15 * expected_daily_cases_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = sum(expected_daily_cases_m[τ] * π_m[t - τ] * ifr_noise[m] for τ = 1:(t - 1))
        end
    end

    # Observe
    for m = 1:num_countries
        # Extract the estimated expected daily deaths for country `m`
        expected_daily_deaths_m = expected_daily_deaths[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_countries[m]
        # Observe!
        deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
    end

    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rt = Rt,
        Rt_adjusted = Rt_adj
    )
end;
{% endhighlight %}

Two things worth noting is the use of this `TV` variable to instantiate some internal variables and the use of `@threads`. 

As you can see in the arguments for the model, `TV` refers to a *type* and will be recognized as such by the `@model` macro when transforming the model code. This is used to ensure *type-stability* of the model.

<details><summary>More detailed explanation of <code>TV</code></summary>

A default execution of the model will then use `TV` as `Vector{Float64}`, thus making statements like `TV(undef, n)` result in a `Vector{Float64}` with `undef` (uninitialized) values and length `n`. But in the case where we want to take use a sampler using automatic differentiation (AD), e.g. `HMC`, this `TV` will be replaced with the AD-type corresponding to a `Vector`, e.g. `TrackedVector` in the case of [`Tracker.jl`](https://github.com/FluxML/Tracker.jl).

</details>

The use of `@threads` means that inside each execution of the `Model`, this loop will be performed in parallel, where the number of threads are specified by the enviroment variable `JULIA_NUM_THREADS`. This is thanks to the really nice multithreading functionality [introduced in Julia 1.3](https://julialang.org/blog/2019/07/multithreading/) (and so this also requires Julia 1.3 or higher to run the code). Note that the inside the loop is independent of each other and each `m` will be seen by only one thread, hence it's threadsafe.

This model is basically identitical to the one defined in [stan-models/base.stan (#6ee3010)](https://github.com/ImperialCollegeLondon/covid19model/blob/6ee3010a58a57cc14a16545ae897ca668b7c9096/stan-models/base.stan) with the exception of two points:

-   in this model we use `TruncatedNormal` for normally distributed variables which are positively constrained
-   we've added the use of `max(pop_m - cases_pred_m[t], 0)` in computing the *adjusted* \\(R_t\\), `Rt_adj`, to ensure that in the case where the entire populations has died there, the adjusted \\(R_t\\) is set to 0, i.e. if everyone in the country passed away then there is no spread (this does not affect "correctness" of inference) <sup><a id="fnr.1" class="footref" href="#fn.1">1</a></sup>
-   the `cases` and `deaths` arguments are arrays of arrays instead of 3D arrays, therefore we don't need to fill the future days with `-1` as is done in the original model


### Multithreaded observe

We can also make the `observe` statements parallel, but because the `~` is not (yet) threadsafe we unfortunately have to touch some of the internals of `Turing.jl`. But for observations it's very straight-forward: instead of observing by the following piece of code

{% highlight julia %}
for m = 1:num_countries
    # Extract the estimated expected daily deaths for country `m`
    expected_daily_deaths_m = expected_daily_deaths[m]
    # Extract time-steps for which we have observations
    ts = epidemic_start[m]:num_obs_countries[m]
    # Observe!
    deaths[m][ts] ~ arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ))
end
{% endhighlight %}

we can use the following

{% highlight julia %}
# Doing observations in parallel provides a small speedup
logps = TV(undef, num_countries)
@threads for m = 1:num_countries
    # Extract the estimated expected daily deaths for country `m`
    expected_daily_deaths_m = expected_daily_deaths[m]
    # Extract time-steps for which we have observations
    ts = epidemic_start[m]:num_obs_countries[m]
    # Observe!
    logps[m] = logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts])
end
Turing.acclogp!(_varinfo, sum(logps))
{% endhighlight %}

<details><summary>Explanation of what we just did</summary>

It might be worth explaining a bit about what's going on here. First we should explain what the deal is with `_varinfo`. `_varinfo` is basically the object used internally in Turing to track the sampled variables and the log-pdf *for a particular evaluation* of the model, and so `acclogp!(_varinfo, lp)` will increment the log-pdf stored in `_varinfo` by `lp`. With that we can explain what happens to `~` inside the `@macro`. Using the old observe-snippet as an example, the `@model` macro replaces `~` with

{% highlight julia %}
acclogp!(_varinfo., logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts]))
{% endhighlight %}

But we're iterating through `m`, so this would not be thread-safe since you might be two threads attempting to mutate `_varinfo` simultaneously.<sup><a id="fnr.2" class="footref" href="#fn.2">2</a></sup> Therefore, since no threads sees the same `m`, delaying the accumulation to after having computed all the log-pdf in parallel leaves us with equivalent code that is threadsafe.

You can read more about the `@macro` and its internals [here](https://turing.ml/dev/docs/for-developers/compiler#model-macro-and-modelgen).

</details>


### Final model

This results in the following model definition

{% highlight julia %}
@model function model_v2(
    num_impute,        # [Int] num. of days for which to impute infections
    num_total_days,    # [Int] days of observed data + num. of days to forecast
    cases,             # [AbstractVector{<:AbstractVector{<:Int}}] reported cases
    deaths,            # [AbstractVector{<:AbstractVector{<:Int}}] reported deaths; rows indexed by i > N contain -1 and should be ignored
    π,                 # [AbstractVector{<:AbstractVector{<:Real}}] h * s
    covariates,        # [Vector{<:AbstractMatrix}]
    epidemic_start,    # [AbstractVector{<:Int}]
    population,        # [AbstractVector{<:Real}]
    serial_intervals,  # [AbstractVector{<:Real}] fixed pre-calculated serial interval (SI) using empirical data from Neil
    lockdown_index,    # [Int] the index for the `lockdown` covariate in `covariates`
    predict=false,     # [Bool] if `false`, will only compute what's needed to `observe` but not more
    ::Type{TV} = Vector{Float64}
) where {TV}
    # `covariates` should be of length `num_countries` and each entry correspond to a matrix of size `(num_total_days, num_covariates)`
    num_covariates = size(covariates[1], 2)
    num_countries = length(cases)
    num_obs_countries = length.(cases)

    # If we don't want to predict the future, we only need to compute up-to time-step `num_obs_countries[m]`
    last_time_steps = predict ? fill(num_total_days, num_countries) : num_obs_countries

    # Latent variables
    τ ~ Exponential(1 / 0.03) # `Exponential` has inverse parameterization of the one in Stan
    y ~ filldist(Exponential(τ), num_countries)
    ϕ ~ truncated(Normal(0, 5), 0, Inf)
    κ ~ truncated(Normal(0, 0.5), 0, Inf)
    μ ~ filldist(truncated(Normal(3.28, κ), 0, Inf), num_countries)

    α_hier ~ filldist(Gamma(.1667, 1), num_covariates)
    α = α_hier .- log(1.05) / 6.

    ifr_noise ~ filldist(truncated(Normal(1., 0.1), 0, Inf), num_countries)

    # lockdown-related
    γ ~ truncated(Normal(0, 0.2), 0, Inf)
    lockdown ~ filldist(Normal(0, γ), num_countries)

    # Initialization of some quantities
    expected_daily_cases = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    cases_pred = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    expected_daily_deaths = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]
    Rt_adj = TV[TV(undef, last_time_steps[m]) for m in 1:num_countries]

    # Loops over countries and perform independent computations for each country
    # since this model does not include any notion of migration across borders.
    # => might has well wrap it in a `@threads` to perform the computation in parallel.
    @threads for m = 1:num_countries
        # country-specific parameters
        π_m = π[m]
        pop_m = population[m]
        expected_daily_cases_m = expected_daily_cases[m]
        cases_pred_m = cases_pred[m]
        expected_daily_deaths_m = expected_daily_deaths[m]
        Rt_m = Rt[m]
        Rt_adj_m = Rt_adj[m]

        last_time_step = last_time_steps[m]

        # Imputation of `num_impute` days
        expected_daily_cases_m[1:num_impute] .= y[m]
        cases_pred_m[1] = zero(cases_pred_m[1])
        cases_pred_m[2:num_impute] .= cumsum(expected_daily_cases_m[1:num_impute - 1])

        xs = covariates[m][1:last_time_step, :] # extract covariates for the wanted time-steps and country `m`
        Rt_m .= μ[m] * exp.(xs * (-α) + (- lockdown[m]) * xs[:, lockdown_index])

        # adjusts for portion of pop that are susceptible
        Rt_adj_m[1:num_impute] .= (max.(pop_m .- cases_pred_m[1:num_impute], zero(cases_pred_m[1])) ./ pop_m) .* Rt_m[1:num_impute]

        for t = (num_impute + 1):last_time_step
            cases_pred_m[t] = cases_pred_m[t - 1] + expected_daily_cases_m[t - 1]

            Rt_adj_m[t] = (max(pop_m - cases_pred_m[t], zero(cases_pred_m[t])) / pop_m) * Rt_m[t] # adjusts for portion of pop that are susceptible
            expected_daily_cases_m[t] = Rt_adj_m[t] * sum(expected_daily_cases_m[τ] * serial_intervals[t - τ] for τ = 1:(t - 1))
        end

        expected_daily_deaths_m[1] = 1e-15 * expected_daily_cases_m[1]
        for t = 2:last_time_step
            expected_daily_deaths_m[t] = sum(expected_daily_cases_m[τ] * π_m[t - τ] * ifr_noise[m] for τ = 1:(t - 1))
        end
    end

    # Observe
    # Doing observations in parallel provides a small speedup
    logps = TV(undef, num_countries)
    @threads for m = 1:num_countries
        # Extract the estimated expected daily deaths for country `m`
        expected_daily_deaths_m = expected_daily_deaths[m]
        # Extract time-steps for which we have observations
        ts = epidemic_start[m]:num_obs_countries[m]
        # Observe!
        logps[m] = logpdf(arraydist(NegativeBinomial2.(expected_daily_deaths_m[ts], ϕ)), deaths[m][ts])
    end
    Turing.acclogp!(_varinfo, sum(logps))

    return (
        expected_daily_cases = expected_daily_cases,
        expected_daily_deaths = expected_daily_deaths,
        Rt = Rt,
        Rt_adjusted = Rt_adj
    )
end;
{% endhighlight %}

    ┌ Warning: you are using the internal variable `_varinfo`
    └ @ DynamicPPL /homes/tef30/.julia/packages/DynamicPPL/3jy49/src/compiler.jl:175

We define an alias `model_def` so that if we want to try out a different model, there's only one point in the notebook which we need to change.

{% highlight julia %}
model_def = model_v2;
{% endhighlight %}

The input data have up to 30-40 days of unobserved future data which we might want to predict on. But during sampling we don't want to waste computation on sampling for the future for which we do not have any observations. Therefore we have an argument `predict::Bool` in the model which allows us to specify whether or not to generate future quantities.

{% highlight julia %}
# Model instantance used to for inference
m_no_pred = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    data.covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    false # <= DON'T predict
);
{% endhighlight %}

{% highlight julia %}
# Model instance used for prediction
m = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    data.covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= predict
);
{% endhighlight %}

Just to make sure everything is working, we can "evaluate" the model to obtain a sample from the prior:

{% highlight julia %}
res = m();
res.expected_daily_cases[uk_index]
{% endhighlight %}

    100-element Array{Float64,1}:
        166.5611423862265
        166.5611423862265
        166.5611423862265
        166.5611423862265
        166.5611423862265
        166.5611423862265
        309.141902695016
        374.0433560255902
        458.3253551103017
        567.9400022100784
        706.8887433414757
        880.8324318305234
       1097.7226599235064
          ⋮
     860793.7037040329
     860371.4970219615
     857377.5507668774
     851840.3636037583
     843819.1795993944
     833402.6417041641
     820706.6951623543
     805871.8261698874
     789059.7490498921
     770449.6760629454
     750234.3165334907
     728615.7558031189


# Visualization utilities

For visualisation we of course use [Plots.jl](https://github.com/JuliaPlots/Plots.jl), and in this case we're going to use the `pyplot` backend which uses Python's matplotlib under the hood.

{% highlight julia %}
using Plots, StatsPlots
pyplot()
{% endhighlight %}

    Plots.PyPlotBackend()

<details><summary>Method definition for plotting the predictive distribution</summary>

{% highlight julia %}
# Ehh, this can be made nicer...
function country_prediction_plot(country_idx, predictions_country::AbstractMatrix, e_deaths_country::AbstractMatrix, Rt_country::AbstractMatrix; normalize_pop::Bool = false)
    pop = data.population[country_idx]
    num_total_days = data.num_total_days
    num_observed_days = length(data.cases[country_idx])

    country_name = countries[country_idx]
    start_date = first(country_to_dates[country_name])
    dates = cumsum(fill(Day(1), data.num_total_days)) + (start_date - Day(1))
    date_strings = Dates.format.(dates, "Y-mm-dd")

    # A tiny bit of preprocessing of the data
    preproc(x) = normalize_pop ? x ./ pop : x

    daily_deaths = data.deaths[country_idx]
    daily_cases = data.cases[country_idx]

    p1 = plot(; xaxis = false, legend = :topleft)
    bar!(preproc(daily_deaths), label="$(country_name)")
    title!("Observed daily deaths")
    vline!([data.epidemic_start[country_idx]], label="epidemic start", linewidth=2)
    vline!([num_observed_days], label="end of observations", linewidth=2)
    xlims!(0, num_total_days)

    p2 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p2, preproc(e_deaths_country); label = "$(country_name)")
    title!("Expected daily deaths (pred)")
    bar!(preproc(daily_deaths), label="$(country_name) (observed)", alpha=0.5)

    p3 = plot(; legend = :bottomleft, xaxis=false)
    plot_confidence_timeseries!(p3, Rt_country; no_label = true)
    for (c_idx, c_time) in enumerate(findfirst.(==(1), eachcol(data.covariates[country_idx])))
        if c_time !== nothing
            # c_name = names(covariates)[2:end][c_idx]
            c_name = names_covariates[c_idx]
            if (c_name != "any")
                # Don't add the "any intervention" stuff
                vline!([c_time - 1], label=c_name)
            end
        end
    end
    title!("Rt")
    qs = [quantile(v, [0.025, 0.975]) for v in eachrow(Rt_country)]
    lq, hq = (eachrow(hcat(qs...))..., )
    ylims!(0, maximum(hq) + 0.1)

    # p3 = bar(replace(data.cases[country_idx], missing => -1.), label="$(country_name)")
    # title!("Daily cases")

    p4 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p4, preproc(predictions_country); label = "$(country_name)")
    title!("Expected daily cases (pred)")
    bar!(preproc(daily_cases), label="$(country_name) (observed)", alpha=0.5)

    vals = preproc(cumsum(e_deaths_country; dims = 1))
    p5 = plot(; legend = :topleft, xaxis=false)
    plot_confidence_timeseries!(p5, vals; label = "$(country_name)")
    plot!(preproc(cumsum(daily_deaths)), label="observed/recorded", color=:red)
    title!("Expected deaths (pred)")

    vals = preproc(cumsum(predictions_country; dims = 1))
    p6 = plot(; legend = :topleft)
    plot_confidence_timeseries!(p6, vals; label = "$(country_name)")
    plot!(preproc(daily_cases), label="observed/recorded", color=:red)
    title!("Expected cases (pred)")

    p = plot(p1, p3, p2, p4, p5, p6, layout=(6, 1), size=(900, 1200), sharex=true)
    xticks!(1:3:num_total_days, date_strings[1:3:end], xrotation=45)

    return p
end

function country_prediction_plot(country_idx, cases, e_deaths, Rt; kwargs...)
    n = length(cases)
    e_deaths_country = hcat([e_deaths[t][country_idx] for t = 1:n]...)
    Rt_country = hcat([Rt[t][country_idx] for t = 1:n]...)
    predictions_country = hcat([cases[t][country_idx] for t = 1:n]...)

    return country_prediction_plot(country_idx, predictions_country, e_deaths_country, Rt_country; kwargs...)
end
{% endhighlight %}

    country_prediction_plot (generic function with 2 methods)

</details>


# Prior

Before we do any inference it can be useful to inspect the *prior* distribution, in particular if you are working with a hierarchical model where the dependencies in the prior might lead to some unexpected behavior. In Turing.jl you can sample a chain from the prior using `sample`, much in the same way as you would sample from the posterior.

{% highlight julia %}
chain_prior = sample(m, Turing.Inference.PriorSampler(), 1_000);
{% endhighlight %}

{% highlight julia %}
plot(chain_prior[[:ϕ, :τ, :κ]]; α = .5, linewidth=1.5)
{% endhighlight %}

![img](../assets/figures/uk-prior-kappa-phi-tau-sample-plot.png)

For the same reasons it can be very useful to inspect the *predictive prior*.

{% highlight julia %}
# Compute the "generated quantities" for the PRIOR
generated_prior = vectup2tupvec(generated_quantities(m, chain_prior));
daily_cases_prior, daily_deaths_prior, Rt_prior, Rt_adj_prior = generated_prior;
{% endhighlight %}

{% highlight julia %}
country_prediction_plot(uk_index, daily_cases_prior, daily_deaths_prior, Rt_prior)
{% endhighlight %}

![img](../assets/figures/uk-predictive-prior-Rt.png)

And with the Rt *adjusted for remaining population*:

{% highlight julia %}
country_prediction_plot(uk_index, daily_cases_prior, daily_deaths_prior, Rt_adj_prior)
{% endhighlight %}

![img](../assets/figures/uk-predictive-prior-Rt-adjusted.png)

At this point it might be useful to remind ourselves of the total population of UK is:

{% highlight julia %}
data.population[uk_index]
{% endhighlight %}

    67886004

As we can see from the figures, the prior allows scenarios such as

-   *all* of the UK being infected
-   effects of interventions, e.g. `lockdown`, having a *negative* effect on `Rt` (in the sense that it can actually *increase* the spread)

But at the same time, it's clear that a very sudden increase jump from 0% to 100% of the population being infected is almost impossible under the prior. All in all, the model prior seems a reasonable choice: it allows for extreme situations without putting too much probabilty "mass" on those, while still encoding some structure in the model.


# Posterior inference

{% highlight julia %}
parameters = (
    warmup = 1000,
    steps = 3000
);
{% endhighlight %}


## Inference


### Run

To perform inference for the model we would simply run the code below:

{% highlight julia %}
chains_posterior = sample(m_no_pred, NUTS(parameters.warmup, 0.95, 10), parameters.steps + parameters.warmup)
{% endhighlight %}

*But* unfortunately it takes quite a while to run. Performing inference using `NUTS` with `1000` steps for adaptation/warmup and `3000` sample steps takes ~2hrs on a 6-core computer with `JULIA_NUM_THREADS = 6`. And so we're instead just going to load in the chains needed.

In contrast, `Stan` only takes roughly 1hr *on a single thread* using the base model from the repository. On a single thread `Turing.jl` is ~4-5X slower for this model, which is quite signficant.

This generally means that if you have a clear model in mind (or you're already very familiar with `Stan`), you probably want to use `Stan` for these kind of models. On the other hand, if you're in the process of heavily tweaking your model and need to be able to do `m = model(data...); m()` to check if it works, or you want more flexibility in what you can do with your model, `Turing.jl` might be a good option.

And this is an additional reason why we wanted to perform this replication study: we want `Turing.jl` to be *useful* and the way to check this is by applying `Turing.jl` to real-world problem rather than *just* benchmarks (though those are important too).

And regarding the performance difference, it really comes down to the difference in implementation of automatic differentation (AD). `Turing.jl` allows you to choose from the goto AD packages in Julia, e.g. in our runs we used [ForwardDiff.jl](https://github.com/JuliaDiff/ForwardDiff.jl), while `Stan` as a very, very fast AD implementation written exclusively for `Stan`. This difference becomes very clear in models such as this one where we have a lot of for-loops and recursive relationships (because this means that we can't easily vectorize). For-loops in Julia are generally blazingly fast, but with AD there's a bit of overhead. But that also means that in `Turing.jl` you have the ability to choose between different approaches to AD, i.e. forward-mode or reverse-mode, each with their different tradeoffs, and thus will benefit from potentially interesting future work, e.g. source-to-source AD using [Zygote.jl](https://github.com/FluxML/Zygote.jl).<sup><a id="fnr.3" class="footref" href="#fn.3">3</a></sup>

And one interesting tidbit is that you can very easily use `pystan` within `PyCall.jl` to sample from a `Stan` model, and then convert the results into a [MCMCChains.jl](https://github.com/TuringLang/MCMCChains.jl). This has some nice implications:

-   we can use all the convenient posterior analysis tools available in MCMCChains.jl to analyze chains from `Stan`
-   we can use the `generated_quantities` method in this notebook to execute the `Turing.jl` `Model` on the samples obtain using `Stan`

This was quite useful for us to be able validate the results from `Turing.jl` against those from `Stan`, and made it very easy to check that indeed `Turing.jl` and `Stan` produce the same results. You can find examples of this in the [notebooks in our repository](https://github.com/TuringLang/Covid19).


### Load

Unfortunately the resulting chains, each with 3000 steps, take up a fair bit of space and are thus too large to include in the Github repository. As a temporary hack around this, you can find download the chains from [this link](https://drive.google.com/open?id=16PomGVnjPI1Q4KLdA9gRloVolfRmrhPP). Then you simply navigate to the project-directory and unpack.

With that, we can load the chains from disk:

{% highlight julia %}
filenames = [
    relpath(projectdir("out", s)) for s in readdir(projectdir("out"))
    if occursin(savename(parameters), s) && occursin("seed", s)
]
filenames = [
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-4-threads_seed=2_steps=3000_warmup=1000_with_lockdown=true.jls",
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads_seed=1_steps=3000_warmup=1000_with_lockdown=true.jls",
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads_seed=3_steps=3000_warmup=1000_with_lockdown=true.jls",
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads_seed=4_steps=3000_warmup=1000_with_lockdown=true.jls",
]
filenames = [
     "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads-updated-full-truncation_seed=1_steps=3000_warmup=1000_with_lockdown=true.jls",
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads-updated-full-truncation_seed=2_steps=3000_warmup=1000_with_lockdown=true.jls",
    "out/chains_model=imperial-report13-v2-vectorized-non-predict-6-threads-updated-full-truncation_seed=4_steps=3000_warmup=1000_with_lockdown=true.jls"
]
length(filenames)
{% endhighlight %}

    3

{% highlight julia %}
chains_posterior_vec = [read(fname, Chains) for fname in filenames]; # read the different chains
chains_posterior = chainscat(chains_posterior_vec...); # concatenate them
chains_posterior = chains_posterior[1:3:end] # <= thin so we're left with 1000 samples

# rename some variables to make the chain compatible with new model where we've changed a variable name from μ₀ → μ
# chains_posterior = set_names(chains_posterior, Dict{String, String}(["μ₀[$i]" => "μ[$i]" for i = 1:length(names(chains_posterior[:μ₀]))]))
{% endhighlight %}

    Object of type Chains, with data of type 1000×78×3 Array{Float64,3}
    
    Iterations        = 1:2998
    Thinning interval = 3
    Chains            = 1, 2, 3
    Samples per chain = 1000
    internals         = acceptance_rate, hamiltonian_energy, hamiltonian_energy_error, is_accept, log_density, lp, max_hamiltonian_energy_error, n_steps, nom_step_size, numerical_error, step_size, tree_depth
    parameters        = ifr_noise[1], ifr_noise[2], ifr_noise[3], ifr_noise[4], ifr_noise[5], ifr_noise[6], ifr_noise[7], ifr_noise[8], ifr_noise[9], ifr_noise[10], ifr_noise[11], ifr_noise[12], ifr_noise[13], ifr_noise[14], lockdown[1], lockdown[2], lockdown[3], lockdown[4], lockdown[5], lockdown[6], lockdown[7], lockdown[8], lockdown[9], lockdown[10], lockdown[11], lockdown[12], lockdown[13], lockdown[14], y[1], y[2], y[3], y[4], y[5], y[6], y[7], y[8], y[9], y[10], y[11], y[12], y[13], y[14], α_hier[1], α_hier[2], α_hier[3], α_hier[4], α_hier[5], α_hier[6], γ, κ, μ[1], μ[2], μ[3], μ[4], μ[5], μ[6], μ[7], μ[8], μ[9], μ[10], μ[11], μ[12], μ[13], μ[14], τ, ϕ
    
    2-element Array{ChainDataFrame,1}
    
    Summary Statistics
         parameters      mean      std  naive_se    mcse        ess   r_hat
      ─────────────  ────────  ───────  ────────  ──────  ─────────  ──────
       ifr_noise[1]    1.0011   0.1021    0.0019  0.0015  2932.0567  1.0000
       ifr_noise[2]    1.0003   0.0984    0.0018  0.0020  3005.2198  0.9996
       ifr_noise[3]    0.9948   0.0999    0.0018  0.0019  3053.2148  0.9996
       ifr_noise[4]    1.0029   0.1001    0.0018  0.0023  2930.0994  1.0007
       ifr_noise[5]    0.9974   0.1012    0.0018  0.0020  2852.4820  1.0005
       ifr_noise[6]    0.9916   0.0956    0.0017  0.0015  3046.4634  0.9993
       ifr_noise[7]    0.9944   0.1001    0.0018  0.0020  2735.6888  1.0021
       ifr_noise[8]    0.9912   0.1016    0.0019  0.0021  3067.4748  1.0005
       ifr_noise[9]    1.0020   0.0980    0.0018  0.0020  2423.4016  1.0002
      ifr_noise[10]    1.0147   0.0982    0.0018  0.0017  2799.5798  0.9999
      ifr_noise[11]    0.9955   0.0987    0.0018  0.0019  2568.5121  1.0013
      ifr_noise[12]    1.0030   0.1008    0.0018  0.0017  3086.5435  0.9994
      ifr_noise[13]    1.0009   0.0990    0.0018  0.0021  2790.3241  1.0004
      ifr_noise[14]    1.0068   0.0962    0.0018  0.0018  2552.6097  0.9996
        lockdown[1]   -0.0021   0.1069    0.0020  0.0021  2706.9213  0.9993
        lockdown[2]   -0.0189   0.0852    0.0016  0.0019  1829.4690  1.0012
        lockdown[3]   -0.0195   0.1022    0.0019  0.0020  2825.7455  0.9993
        lockdown[4]    0.1267   0.1212    0.0022  0.0042   831.8401  1.0002
        lockdown[5]    0.0128   0.1085    0.0020  0.0026  1321.8935  1.0008
        lockdown[6]   -0.0283   0.0935    0.0017  0.0019  1418.7909  1.0011
        lockdown[7]   -0.0201   0.1177    0.0021  0.0024  2625.7223  1.0000
        lockdown[8]   -0.0756   0.1140    0.0021  0.0022  1642.8079  0.9997
        lockdown[9]   -0.0012   0.1034    0.0019  0.0016  2734.9363  0.9996
       lockdown[10]   -0.0026   0.1334    0.0024  0.0024  3005.9277  0.9994
       lockdown[11]    0.0087   0.1042    0.0019  0.0016  2822.7506  0.9992
       lockdown[12]    0.0045   0.1216    0.0022  0.0027  2501.5179  1.0001
       lockdown[13]   -0.0026   0.1100    0.0020  0.0025  2545.6748  1.0012
       lockdown[14]    0.0478   0.1162    0.0021  0.0027  1706.4820  1.0010
               y[1]   46.4383  23.1858    0.4233  0.4868  2386.0344  0.9999
               y[2]   38.5317  13.1005    0.2392  0.2874  1958.6332  0.9999
               y[3]   19.4335   9.5841    0.1750  0.2473  1287.9233  0.9999
               y[4]   72.6812  33.5631    0.6128  0.9727  1091.9024  1.0027
               y[5]   42.7182  17.2201    0.3144  0.3649  1716.9929  1.0005
               y[6]    9.1140   4.2425    0.0775  0.0797  2024.8046  1.0000
               y[7]   29.0969  17.3875    0.3175  0.2716  2410.8193  0.9992
               y[8]   16.6010  10.8060    0.1973  0.2234  1541.1963  1.0003
               y[9]   72.6486  33.9504    0.6198  0.7721  1815.8773  0.9993
              y[10]  140.4307  51.5581    0.9413  1.2643  1288.0214  1.0021
              y[11]   25.9010  13.3297    0.2434  0.2614  1924.6912  0.9994
              y[12]   82.1905  43.0910    0.7867  1.0058  1994.1915  1.0007
              y[13]   53.7862  25.7194    0.4696  0.5148  2112.6082  0.9994
              y[14]   81.2253  35.3834    0.6460  0.8644  1630.5038  1.0005
          α_hier[1]    0.0247   0.0506    0.0009  0.0010  2594.6821  1.0007
          α_hier[2]    0.0296   0.0548    0.0010  0.0012  2358.9473  1.0016
          α_hier[3]    0.4892   0.1663    0.0030  0.0049  1036.1321  1.0012
          α_hier[4]    0.0168   0.0359    0.0007  0.0008  2085.6784  1.0008
          α_hier[5]    1.1229   0.1455    0.0027  0.0028  1754.4547  0.9997
          α_hier[6]    0.0717   0.1082    0.0020  0.0037   589.3802  1.0025
                  γ    0.1101   0.0657    0.0012  0.0023   543.4732  1.0037
                  κ    1.1471   0.2347    0.0043  0.0053  1533.3092  1.0007
               μ[1]    3.9813   0.4716    0.0086  0.0109  1650.5822  0.9996
               μ[2]    3.6135   0.1689    0.0031  0.0043  1389.8969  1.0006
               μ[3]    4.2813   0.4265    0.0078  0.0164   592.0907  1.0031
               μ[4]    4.5631   0.3910    0.0071  0.0125   697.3221  1.0048
               μ[5]    3.9381   0.2982    0.0054  0.0091   840.5478  1.0033
               μ[6]    4.9616   0.3268    0.0060  0.0067  1695.0212  1.0001
               μ[7]    3.8714   0.5370    0.0098  0.0084  1800.9081  0.9991
               μ[8]    6.3770   0.7334    0.0134  0.0169  1199.8237  1.0004
               μ[9]    4.3805   0.5275    0.0096  0.0131  1550.2812  0.9997
              μ[10]    2.4311   0.2328    0.0043  0.0079   647.6823  1.0052
              μ[11]    3.8239   0.3794    0.0069  0.0100  1215.1098  1.0010
              μ[12]    2.1017   0.3097    0.0057  0.0099   988.5009  1.0028
              μ[13]    3.9721   0.4624    0.0084  0.0119  1316.8830  0.9998
              μ[14]    3.7075   0.3512    0.0064  0.0102  1115.3457  1.0008
                  τ   52.8203  17.2966    0.3158  0.4136  1355.0613  1.0004
                  ϕ    6.8790   0.5519    0.0101  0.0101  2812.6168  1.0003
    
    Quantiles
         parameters     2.5%     25.0%     50.0%     75.0%     97.5%
      ─────────────  ───────  ────────  ────────  ────────  ────────
       ifr_noise[1]   0.8019    0.9323    1.0019    1.0724    1.1935
       ifr_noise[2]   0.8114    0.9329    1.0008    1.0657    1.1920
       ifr_noise[3]   0.7999    0.9283    0.9954    1.0623    1.1882
       ifr_noise[4]   0.8059    0.9345    1.0038    1.0719    1.1987
       ifr_noise[5]   0.7988    0.9290    0.9978    1.0661    1.1959
       ifr_noise[6]   0.8105    0.9284    0.9895    1.0572    1.1825
       ifr_noise[7]   0.8024    0.9274    0.9927    1.0600    1.1936
       ifr_noise[8]   0.7930    0.9243    0.9908    1.0559    1.1955
       ifr_noise[9]   0.8102    0.9336    1.0023    1.0673    1.1868
      ifr_noise[10]   0.8285    0.9488    1.0164    1.0796    1.2193
      ifr_noise[11]   0.7926    0.9290    0.9983    1.0641    1.1840
      ifr_noise[12]   0.8072    0.9333    1.0015    1.0712    1.2006
      ifr_noise[13]   0.8048    0.9361    1.0002    1.0690    1.1946
      ifr_noise[14]   0.8179    0.9418    1.0066    1.0721    1.1968
        lockdown[1]  -0.2240   -0.0571   -0.0036    0.0521    0.2288
        lockdown[2]  -0.2093   -0.0651   -0.0116    0.0282    0.1430
        lockdown[3]  -0.2469   -0.0715   -0.0116    0.0329    0.1908
        lockdown[4]  -0.0408    0.0316    0.1036    0.1987    0.4167
        lockdown[5]  -0.2055   -0.0430    0.0064    0.0633    0.2568
        lockdown[6]  -0.2521   -0.0754   -0.0182    0.0236    0.1447
        lockdown[7]  -0.2979   -0.0715   -0.0105    0.0397    0.2054
        lockdown[8]  -0.3504   -0.1342   -0.0533   -0.0014    0.1084
        lockdown[9]  -0.2275   -0.0499    0.0005    0.0514    0.2181
       lockdown[10]  -0.2953   -0.0610   -0.0009    0.0549    0.2830
       lockdown[11]  -0.2068   -0.0434    0.0041    0.0595    0.2352
       lockdown[12]  -0.2525   -0.0529    0.0014    0.0611    0.2611
       lockdown[13]  -0.2215   -0.0597   -0.0022    0.0519    0.2328
       lockdown[14]  -0.1596   -0.0182    0.0263    0.1014    0.3462
               y[1]  14.8306   29.9829   41.9288   58.3169  100.5237
               y[2]  19.7647   28.9377   36.5774   45.5141   69.5499
               y[3]   6.3529   12.5614   17.4990   24.3877   43.2661
               y[4]  23.4652   48.6544   67.7566   91.0501  152.5388
               y[5]  15.9470   30.3710   40.3238   52.8219   82.7343
               y[6]   3.4192    6.1705    8.2256   11.0534   19.4929
               y[7]   7.8154   16.9869   24.9424   36.3621   74.9021
               y[8]   4.2702    9.3061   13.9125   20.6639   46.1895
               y[9]  26.0228   48.1463   66.7242   89.2108  157.8612
              y[10]  59.5464  103.2204  134.1816  170.9258  258.1249
              y[11]   8.6770   16.2873   23.2587   32.4366   59.1115
              y[12]  22.7250   51.6193   74.4397  103.6036  187.7852
              y[13]  18.1430   35.3752   49.6002   66.5192  116.7060
              y[14]  30.8264   55.3309   75.5827  100.5668  163.6102
          α_hier[1]   0.0000    0.0000    0.0019    0.0248    0.1765
          α_hier[2]   0.0000    0.0000    0.0031    0.0315    0.2044
          α_hier[3]   0.1427    0.3828    0.4924    0.6015    0.7993
          α_hier[4]   0.0000    0.0000    0.0012    0.0146    0.1277
          α_hier[5]   0.8209    1.0280    1.1250    1.2191    1.3991
          α_hier[6]   0.0000    0.0003    0.0139    0.1088    0.3612
                  γ   0.0127    0.0593    0.1022    0.1495    0.2608
                  κ   0.7538    0.9799    1.1264    1.2909    1.6678
               μ[1]   3.1502    3.6546    3.9470    4.2642    5.0133
               μ[2]   3.2833    3.4988    3.6068    3.7261    3.9530
               μ[3]   3.5862    3.9885    4.2325    4.5304    5.2493
               μ[4]   3.9237    4.2856    4.5150    4.7955    5.4220
               μ[5]   3.4762    3.7300    3.8897    4.1093    4.6526
               μ[6]   4.3534    4.7338    4.9547    5.1783    5.6308
               μ[7]   2.9072    3.5066    3.8426    4.2107    5.0483
               μ[8]   4.9969    5.8755    6.3652    6.8593    7.8987
               μ[9]   3.4066    4.0227    4.3383    4.7113    5.5326
              μ[10]   2.0718    2.2672    2.3952    2.5638    2.9626
              μ[11]   3.1531    3.5595    3.7960    4.0641    4.6259
              μ[12]   1.5835    1.8859    2.0736    2.2889    2.7778
              μ[13]   3.1743    3.6594    3.9305    4.2536    5.0145
              μ[14]   3.0966    3.4590    3.6761    3.9298    4.4287
                  τ  26.9661   40.4253   50.5220   62.0736   95.3553
                  ϕ   5.8478    6.5015    6.8573    7.2352    8.0540

{% highlight julia %}
plot(chains_posterior[[:κ, :ϕ, :τ]]; α = .5, linewidth=1.5)
{% endhighlight %}

![img](../assets/figures/uk-posterior-kappa-phi-tau-sample-plot.png)

{% highlight julia %}
# Compute generated quantities for the chains pooled together
pooled_chains = MCMCChains.pool_chain(chains_posterior)
generated_posterior = vectup2tupvec(generated_quantities(m, pooled_chains));

daily_cases_posterior, daily_deaths_posterior, Rt_posterior, Rt_adj_posterior = generated_posterior;
{% endhighlight %}

The predictive posterior:

{% highlight julia %}
country_prediction_plot(uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_posterior)
{% endhighlight %}

![img](../assets/figures/uk-predictive-posterior-Rt.png)

and with the adjusted \\(R_t\\):

{% highlight julia %}
country_prediction_plot(uk_index, daily_cases_posterior, daily_deaths_posterior, Rt_adj_posterior)
{% endhighlight %}

![img](../assets/figures/uk-predictive-posterior-Rt-adjusted.png)


## All countries: prior vs. posterior predictive

For the sake of completeness, here are the predictive priors and posteriors for all the 14 countries in a side-by-side comparison.

<div class="two-by-two">

![img](../assets/figures/country-prior-predictive-01.png)

![img](../assets/figures/country-posterior-predictive-01.png)

![img](../assets/figures/country-prior-predictive-02.png)

![img](../assets/figures/country-posterior-predictive-02.png)

![img](../assets/figures/country-prior-predictive-03.png)

![img](../assets/figures/country-posterior-predictive-03.png)

![img](../assets/figures/country-prior-predictive-04.png)

![img](../assets/figures/country-posterior-predictive-04.png)

![img](../assets/figures/country-prior-predictive-05.png)

![img](../assets/figures/country-posterior-predictive-05.png)

![img](../assets/figures/country-prior-predictive-06.png)

![img](../assets/figures/country-posterior-predictive-06.png)

![img](../assets/figures/country-prior-predictive-07.png)

![img](../assets/figures/country-posterior-predictive-07.png)

![img](../assets/figures/country-prior-predictive-08.png)

![img](../assets/figures/country-posterior-predictive-08.png)

![img](../assets/figures/country-prior-predictive-09.png)

![img](../assets/figures/country-posterior-predictive-09.png)

![img](../assets/figures/country-prior-predictive-10.png)

![img](../assets/figures/country-posterior-predictive-10.png)

![img](../assets/figures/country-prior-predictive-11.png)

![img](../assets/figures/country-posterior-predictive-11.png)

![img](../assets/figures/country-prior-predictive-12.png)

![img](../assets/figures/country-posterior-predictive-12.png)

![img](../assets/figures/country-prior-predictive-13.png)

![img](../assets/figures/country-posterior-predictive-13.png)

![img](../assets/figures/country-prior-predictive-14.png)

![img](../assets/figures/country-posterior-predictive-14.png)

</div>


## What if we didn't do any/certain interventions?

One interesting thing one can do after obtaining estimates for the effect of each of the interventions is to run the model but now *without* all or a subset of the interventions performed. Thus allowing us to get a sense of what the outcome would have been without those interventions, and also whether or not the interventions have the wanted effect.

`data.covariates[m]` is a binary matrix for each `m` (country index), with `data.covariate[m][:, k]` then being a binary vector representing the time-series for the k-th covariate: `0` means the intervention has is not implemented, `1` means that the intervention is implemented. As an example, if schools and universites were closed after the 45th day for country `m`, then `data.covariate[m][1:45, k]` are all zeros and `data.covariate[m][45:end, k]` are all ones.

{% highlight julia %}
# Get the index of schools and univerities closing
schools_universities_closed_index = findfirst(==("schools_universities"), names_covariates)
# Time-series for UK
data.covariates[uk_index][:, schools_universities_closed_index]
{% endhighlight %}

    100-element Array{Float64,1}:
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     0.0
     ⋮
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0
     1.0

Notice that the above assumes that not only are schools and universities closed *at some point*, but rather that they also stay closed in the future (at the least the future that we are considering).

Therefore we can for example simulate "what happens if we never closed schools and universities?" by instead setting this entire vector to `0` and re-run the model on the infererred parameters, similar to what we did before to compute the "generated quantities", e.g. \\(R_t\\).

<details><summary>Convenience function for zeroing out subsets of the interventions</summary>

{% highlight julia %}
"""
    zero_covariates(xs::AbstractMatrix{<:Real}; remove=[], keep=[])

Allows you to zero out covariates if the name of the covariate is in `remove` or NOT zero out those in `keep`.
Note that only `remove` xor `keep` can be non-empty.

Useful when instantiating counter-factual models, as it allows one to remove/keep a subset of the covariates.
"""
zero_covariates(xs::AbstractMatrix{<:Real}; kwargs...) = zero_covariates(xs, names_covariates; kwargs...)
function zero_covariates(xs::AbstractMatrix{<:Real}, names_covariates; remove=[], keep=[])
    @assert (isempty(remove) || isempty(keep)) "only `remove` or `keep` can be non-empty"

    if isempty(keep)
        return hcat([
            (names_covariates[i] ∈ remove ? zeros(eltype(c), length(c)) : c) 
            for (i, c) in enumerate(eachcol(xs))
        ]...)
    else
        return hcat([
        (names_covariates[i] ∈ keep ? c : zeros(eltype(c), length(c))) 
        for (i, c) in enumerate(eachcol(xs))
        ]...)
    end
end
{% endhighlight %}

    zero_covariates (generic function with 2 methods)

</details>

Now we can consider simulation under the posterior with *no* intervention, and we're going to visualize the respective *portions* of the population by rescaling by total population:

{% highlight julia %}
# What happens if we don't do anything?
m_counterfactual = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    [zeros(size(c)) for c in data.covariates], # <= remove ALL covariates
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(5, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)
{% endhighlight %}

![img](../assets/figures/counterfactual-remove-all.png)

We can also consider the cases where we only do *some* of the interventions, e.g. we never do a full lockdown (`lockdown`) or close schools and universities (`schools_universities`):

{% highlight julia %}
# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    [zero_covariates(c; remove = ["lockdown", "schools_universities"]) for c in data.covariates], # <= remove covariates
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)
{% endhighlight %}

![img](../assets/figures/counterfactual-remove-lockdown-and-schools.png)

As mentioned, this assumes that we will stay in lockdown and schools and universities will be closed in the future. We can also consider, say, removing the lockdown, i.e. opening up, at some future point in time:

{% highlight julia %}
lift_lockdown_time = 75

new_covariates = [copy(c) for c in data.covariates] # <= going to do inplace manipulations so we copy
for covariates_m ∈ new_covariates
    covariates_m[lift_lockdown_time:end, lockdown_index] .= 0
end

# What happens if we never close schools nor do a lockdown?
m_counterfactual = model_def(
    data.num_impute,
    data.num_total_days,
    data.cases,
    data.deaths,
    data.π,
    new_covariates,
    data.epidemic_start,
    data.population,
    data.serial_intervals,
    lockdown_index,
    true # <= use full model
);

# Compute the "generated quantities" for the "counter-factual" model
generated_counterfactual = vectup2tupvec(generated_quantities(m_counterfactual, pooled_chains));
daily_cases_counterfactual, daily_deaths_counterfactual, Rt_counterfactual, Rt_adj_counterfactual = generated_counterfactual;
country_prediction_plot(uk_index, daily_cases_counterfactual, daily_deaths_counterfactual, Rt_adj_counterfactual; normalize_pop = true)
{% endhighlight %}

![img](../assets/figures/counterfactual-remove-lockdown-after-a-while.png)


# Conclusion

Well, there isn't one. As stated before, drawing conclusions is not the purpose of this document. With that being said, we *are* working on exploring this and other models further, e.g. relaxing certain assumptions, model validation & comparison, but this will hopefully be available in a more technical and formal report sometime in the near future after proper validation and analysis. But since time is of the essence in these situations, we thought it was important to make the above and related code available to the public immediately. At the very least it should be comforting to know that two different PPLs both produce the same inference results when the model might be used to inform policy decisions on a national level.

If you have any questions or comments, feel free to reach out either on the [Github repo](https://github.com/TuringLang/Covid19) or to any of us personally.

# Footnotes

<sup><a id="fn.1" href="#fnr.1">1</a></sup> It's important to point out that in Stan these samples will be correctly rejected because it leads to `-Inf` joint probability later in model. Hence inference will be correct even if `max` is not used.

<sup><a id="fn.2" href="#fnr.2">2</a></sup> You *could* use something like [Atomic in Julia](https://docs.julialang.org/en/v1/base/multi-threading/#Base.Threads.Atomic), but it comes at a unnecessary performance overhead in this case.

<sup><a id="fn.3" href="#fnr.3">3</a></sup> Recently one of our team-members joined as a maintainer of [ReverseDiff.jl](https://github.com/JuliaDiff/ReverseDiff.jl) to make sure that `Turing.jl` also has fast and reliable reverse-mode differentiation. ReverseDiff.jl is already compatible with `Turing.jl`, but we hope that this will help make if much, much faster.
