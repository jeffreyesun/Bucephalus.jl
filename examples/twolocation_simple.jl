using Bucephalus, Distributions, Statistics

m = Model()
# Parameters
@parameter m.ρ2 = 0.2
@parameter m.H1 = 1.0
# Agent Classes and Locations
n_hh = 10_000
@agent(m.hh, dynamic=true, n=n_hh)
@segmentation(m.loc, n=2)

## Variables
# State
@statevar hh.l = LogNormal()
@statevar loc.ki = [1.0, 2.0]
@statevar loc.A  = [1.5, 1.0]
@matchvar hh.residence::Match{:loc} = rand(1:2, n_hh)
@statedep hh.income(l,By(A,residence)) = l*A
# Equilibrium
@equilvar m.ρ1 = 1.0
@equildep loc.ρ(ρ1,ρ2,ki) = ki == 1 ? ρ1 : ρ2
# Choice Variables
@choicevar(hh.c, Float64, bounds=(0.0,income))
@choicevar!(hh.residence)
addfactor!.(hh, [l, By(ki,residence), income, By(ρ,residence)])
@choicedep hh.h(c,income,By(ρ,residence)) = (income-c)/ρ
@choicedep loc.H(ByMembers(h,residence)) = sum(h) / n_hh
# Equilibrium Conditions
@equilibriumcondition m.eq1(H,H1) = H[1] - H1
# Payoffs, Shocks, Transitions
@utility hh.u(c,h) = sqrt(max(c,0)*max(h,0))
@shock hh.ε = Normal(0,0.05)
@shock m.A1 = Uniform(.2,.5)
@transition l(l,ε) = exp(0.5*log(l) + 0.5 + ε)
@transition A(ki, A1) = ki == 1 ? 1.0 + A1 : 1.0

## Train Agents
train(m)
