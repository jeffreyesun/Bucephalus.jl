

#### Setup
model = Model(horizon=Inf)

# Parameters
model <| Parameter(:α, 0.5)
model <| Parameter(:δ, 0.1)
model <| Parameter(:ν, 0.5)
model <| Parameter(:ρ, 0.5)
model <| Parameter(:μ, 0.5)
model <| Parameter(:η, 0.5)

# Agents
hh   = model <| Agent("hh", n=10000, dynamic=true, atomistic=true)
firm = model <| Agent("firm", n=1, atomistic=true)

#### Stage Game

# State
hh <| StateVariable(:l, init_dist=LogNormal())
hh <| StateVariable(:k, init_dist=LogNormal())
model <| StateVariable(:A, init_dist=LogNormal())

# "Free" Equilibrium
model <| EquilibriumQuantity(:p, 1.0)
model <| EquilibriumQuantity(:w, 1.0)
model <| EquilibriumQuantity(:r, 0.1)

# Dependent Equilibrium
hh <| @equilibriumstate wealth = w*l + (1-δ+r)*k

# Factor
hh <| @factor [wealth,l]

# Choice
hh <| ChoiceVariable(:c, bounds=(0,wealth))
firm <| ChoiceVariable(:L)
firm <| ChoiceVariable(:K)

# Consequence
firm <| @consequence Y = A*K^α*L^(1-α)
firm <| @consequence Π = p*Y - w*L - r*K

# Equilibrium Conditions
model <| @equilibriumcondition L == sum(l) # w
model <| @equilibriumcondition K == sum(k) # r
model <| @equilibriumcondition Π # (== 0)

# Payoff
hh <| @utility log(c)
firm <| @utility Π

# Shock
hh <| Shock(:ε, Normal())
model <| Shock(:χ, Normal())

# Transition
hh <| @transition l = l + exp(ν*log(l) + ρ + ε)
hh <| @transition k = wealth - p*c
model <| @transition A = A + exp(μ*log(A) + η + χ)

#### Compile

gmd = compile(m)

#### Solve

policy!(gmd)