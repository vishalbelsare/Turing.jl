module Variational

using ..Core, ..Utilities
using DocStringExtensions: TYPEDEF, TYPEDFIELDS
using Distributions, Bijectors, DynamicPPL
using LinearAlgebra
using ..Turing: PROGRESS, Turing
using DynamicPPL: Model, SampleFromPrior, SampleFromUniform
using Random: AbstractRNG

using ForwardDiff
using Tracker

using AdvancedVI

import ..Core: getchunksize, getADbackend

import AbstractMCMC
import ProgressLogging

using Requires
function __init__()
    @require Flux="587475ba-b771-5e3f-ad9e-33799f191a9c" begin
        apply!(o, x, Δ) = Flux.Optimise.apply!(o, x, Δ)
        Flux.Optimise.apply!(o::TruncatedADAGrad, x, Δ) = apply!(o, x, Δ)
        Flux.Optimise.apply!(o::DecayedADAGrad, x, Δ) = apply!(o, x, Δ)
    end
    @require Zygote = "e88e6eb3-aa80-5325-afca-941959d7151f" begin
        function AdvancedVI.grad!(
            vo,
            alg::VariationalInference{<:Turing.ZygoteAD},
            q,
            model,
            θ::AbstractVector{<:Real},
            out::DiffResults.MutableDiffResult,
            args...
        )
            f(θ) = if (q isa Distribution)
                - vo(alg, update(q, θ), model, args...)
            else
                - vo(alg, q(θ), model, args...)
            end
            y, back = Zygote.pullback(f, θ)
            dy = back(1.0)
            DiffResults.value!(out, y)
            DiffResults.gradient!(out, dy)
            return out
        end
    end
    @require ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267" begin
        function AdvancedVI.grad!(
            vo,
            alg::VariationalInference{<:Turing.ReverseDiffAD{false}},
            q,
            model,
            θ::AbstractVector{<:Real},
            out::DiffResults.MutableDiffResult,
            args...
        )
            f(θ) = if (q isa Distribution)
                - vo(alg, update(q, θ), model, args...)
            else
                - vo(alg, q(θ), model, args...)
            end
            tp = Turing.Core.tape(f, θ)
            ReverseDiff.gradient!(out, tp, θ)
            return out
        end
        @require Memoization = "6fafb56a-5788-4b4e-91ca-c0cea6611c73" begin
            function AdvancedVI.grad!(
                vo,
                alg::VariationalInference{<:Turing.ReverseDiffAD{true}},
                q,
                model,
                θ::AbstractVector{<:Real},
                out::DiffResults.MutableDiffResult,
                args...
            )
                f(θ) = if (q isa Distribution)
                    - vo(alg, update(q, θ), model, args...)
                else
                    - vo(alg, q(θ), model, args...)
                end
                ctp = Turing.Core.memoized_tape(f, θ)
                ReverseDiff.gradient!(out, ctp, θ)
                return out
            end
        end
    end
end

export
    vi,
    ADVI,
    ELBO,
    elbo,
    TruncatedADAGrad,
    DecayedADAGrad


"""
    make_logjoint(model::Model; weight = 1.0)

Constructs the logjoint as a function of latent variables, i.e. the map z → p(x ∣ z) p(z).

The weight used to scale the likelihood, e.g. when doing stochastic gradient descent one needs to
use `DynamicPPL.MiniBatch` context to run the `Model` with a weight `num_total_obs / batch_size`.

## Notes
- For sake of efficiency, the returned function is closes over an instance of `VarInfo`. This means that you *might* run into some weird behaviour if you call this method sequentially using different types; if that's the case, just generate a new one for each type using `make_logjoint`.
"""
function make_logjoint(model::Model; weight = 1.0)
    # setup
    ctx = DynamicPPL.MiniBatchContext(
        DynamicPPL.DefaultContext(),
        weight
    )
    varinfo_init = Turing.VarInfo(model, ctx)

    function logπ(z)
        varinfo = VarInfo(varinfo_init, SampleFromUniform(), z)
        model(varinfo)

        return getlogp(varinfo)
    end

    return logπ
end

function logjoint(model::Model, varinfo, z)
    varinfo = VarInfo(varinfo, SampleFromUniform(), z)
    model(varinfo)

    return getlogp(varinfo)
end


# objectives
function (elbo::ELBO)(
    rng::AbstractRNG,
    alg::VariationalInference,
    q,
    model::Model,
    num_samples;
    weight = 1.0,
    kwargs...
)
    return elbo(rng, alg, q, make_logjoint(model; weight = weight), num_samples; kwargs...)
end

# VI algorithms
include("advi.jl")

end
