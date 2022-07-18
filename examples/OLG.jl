
using Distributions
using Parameters

# OLG Via Reincarnation

#### Setup
model = Model(horizon=Inf)

# Parameters
model <| Parameter(:α, 0.5)
model <| Parameter(:δ, 0.1)
model <| Parameter(:A, 1.0)
model <| Parameter(:ν, 0.5)
model <| Parameter(:ρ, 0.5)

# Agents
hh   = model <| Agent("hh", n=10000, dynamic=true, atomistic=true)
firm = model <| Agent("firm", n=1, atomistic=true)

#### Stage Game

# State
hh <| StateVariable(:l, init_dist=LogNormal())
hh <| StateVariable(:k, init_dist=LogNormal())
hh <| StateVariable(:alive, Bool, init_dist=Bernoulli(0.95))

# Equilibrium
model <| EquilibriumQuantity(:p, 1.0)
model <| EquilibriumQuantity(:w, 1.0)
model <| EquilibriumQuantity(:r, 0.1)

# Choice
hh <| ChoiceVariable(:c, bounds=(0,w*l + (1+r)*k))
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
hh <| Shock(:death, Bernoulli(0.95))

# Transition
function update_hh(params, l, k, r, w, p, c, ε, death)
    @unpack params =  δ, ν, ρ
    if death
        return 0.0, 0.0, false
    else
        l′ = l + exp(v*log(l) + ρ + ε)
        k′ = w*l + (1-δ+r)*k - p*c
        alive = true
        return l′, k′, alive
    end
end
hh <| @transition l, k, alive = update_hh(params,l,k,r,w,p,c,ε,death)

# Value function override
hh <| @valuefunctionoverride v = alive ? v : 0.0