using Bucephalus, Distributions, Statistics

m = Model()
# Parameters
@parameter m.α = 0.5
@parameter m.δ = 0.1
@parameter m.ν = 0.5
@parameter m.ρ = 0.5
# Agent Classes
@agent(m.hh, dynamic=true, n=10_000)

## Variables
# State
@statevar m.A = 1.0
@statevar hh.l = LogNormal()
@statevar hh.k = LogNormal()
@statedep m.L(l) = mean(l)
@statedep m.K(k) = mean(k)
@statedep m.w(α,A,K,L) = (1-α)*A*(K/L)^α
@statedep m.r(α,A,K,L) = α*A*(K/L)^(1-α)
@statedep hh.wealth(w,l,δ,r,k) = w*l + (1-δ+r)*k
# Choice
@choicevar(hh.c, Float64, bounds=(0.0,wealth))
addfactor!.(hh, [l, k, A, L, K, w, r, wealth])
# Payoffs, Shocks, Transitions
@utility hh.u(c) = log(c)
@shock hh.ε = Normal(0, 0.01)
@shock m.ζ = Uniform(0, 1)
@transition l(ν,l,ρ,ε) = exp(ν*log(l) + ρ + ε)
@transition k(wealth, c) = wealth - c
@transition A(ζ, A) = ζ > 0.9 ? 2.5 - A : A

## Train Agents
train(m)
