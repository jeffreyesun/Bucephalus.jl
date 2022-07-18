

#### Setup
model = Model()

# Parameters
model <| Parameter(:α, 0.5)
model <| Parameter(:δ, 0.1)
model <| Parameter(:ν, 0.5)
model <| Parameter(:ρ, 0.5)

# Agents
nfirms = 1000
hh   = model <| Agent("hh", n=10000, dynamic=true, atomistic=true)
firm = model <| Agent("firm", n=nfirms, atomistic=true)

#### Stage Game

# State
hh <| StateVariable(:l, init_dist=LogNormal())
hh <| StateVariable(:k, init_dist=LogNormal())
hh <| StateVariable(:employed, Bool, init_dist=Bernoulli(0.95))
hh <| StateVariable(:employer, Int, bounds=(1,nfirms), init_dist=Categorical(fill(1/nfirms,nfirms)))
model <| StateVariable(:A, init_dist=LogNormal())

# Equilibrium
model <| EquilibriumQuantity(:p, 1.0)
model <| EquilibriumQuantity(:w, 1.0)
model <| EquilibriumQuantity(:r, 0.1)

# Choice
hh <| ChoiceVariable(:c, bounds=(0,w*l + (1+r)*k))
hh <| ChoiceVariable(:search, Bool)
firm <| ChoiceVariable(:L)
firm <| ChoiceVariable(:K)

# Consequence
firm <| @consequence Y = production(params,Ref(employer),L,K)
firm <| @consequence Π = p*Y - w*L - r*K

# Equilibrium Conditions
model <| @equilibriumcondition L == sum(l) # w
model <| @equilibriumcondition K == sum(k) # r
model <| @equilibriumcondition Π # (== 0)

# Payoff
hh <| @utility log(c)
firm <| @utility Π

# Shock
hh <| Shock(:ε, Normal(), common=false)

# Transition
hh <| @transition l = l + exp(ν*log(l) + ρ + ε)
hh <| @transition k = w*l + (1-δ+r)*k - p*c
hh <| @transition employed = matching(params,employed,search,etc)
