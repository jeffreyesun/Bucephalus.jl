

#### Setup
m = Model()

# Parameters
@parameter m.α = 0.5
@parameter m.δ = 0.1
@parameter m.A = 1.0
@parameter m.ν = 0.5
@parameter m.ρ = 0.5

# Agents
hh   = m <| Agent("hh"; n=10000, atomistic=true)
firm = m <| Agent("firm"; n=1, static=true, atomistic=true)
#@agent (m.hh, dynamic=true, n=10_000)
#@agent (m.firm, dynamic=false, n=10_000)

#### Stage Game

# State
#@statevar hh.l = LogNormal()
#@statevar hh.k = LogNormal()
hh <| StateVariable(:l, init=LogNormal())
hh <| StateVariable(:k, init=LogNormal())

# State-Dependent
@statedep m.Ls(l) = mean(l)
@statedep m.Ks(k) = mean(k)

# "Free" Equilibrium Variables
model <| EquilibriumQuantity(:p, 1.0; bounds=(0,Inf))
model <| EquilibriumQuantity(:w, 1.0; bounds=(0,Inf))
model <| EquilibriumQuantity(:r, 0.1; bounds=(0,Inf))
#@equilvar(m.p = 1.0; bounds=(0,Inf))
#@equilvar(m.w = 1.0; bounds=(0,Inf))
#@equilvar(m.r = 0.1; bounds=(0,Inf))

# Equilibrium-Dependent Variables
@equildep hh.wealth = w*l + (1-δ+r)*k

# Choice
@choicevar(hh.c; bounds=(0,wealth))
@choicevar(firm.Ld; bounds=(0,Inf))
@choicevar(firm.Kd; bounds=(0,Inf))

# Choice-Dependent Variables
@choicedep firm.Y(A,Kd,α,Ld) = A*Kd^α*Ld^(1-α)
@choicedep firm.Π(p,Y,w,Ld,r,Kd) = p*Y - w*Ld - r*Kd

# Equilibrium Conditions
@equilibriumcondition m.eq1(Ld, Ls) = Ld - Ls
@equilibriumcondition m.eq2(Kd, Ks) =  Kd - Ks
@equilibriumcondition m.eq3(Y, c) = Y - mean(c)

# Payoff
@utility!(hh.u(c,p) = log(c/p))
@utility!(firm.Π)

# Shock
@shock hh.ε = Normal(0, 0.01)

# Transition
@transition l(ν,l,ρ,ε) = exp(ν*log(l) + ρ + ε)
@transition k(wealth, c) = wealth - c


## Simulate

modeldata = compile(model)
iterate!(modeldata)

## Solve
solve(model, simplesolver())
