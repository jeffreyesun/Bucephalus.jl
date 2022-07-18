
m = Model()

# Parameters
α = addParameter!(m, :α, 0.5)
δ = addParameter!(m, :δ, 0.1)
ν = addParameter!(m, :ν, 0.5)
ρ = addParameter!(m, :ρ, 0.5)
μ = addParameter!(m, :μ, 0.5)
η = addParameter!(m, :η, 0.5)

# Agents
n_hh = 10_000
hh = addAgent!(m, :hh; dynamic=true, n=n_hh)
firm = addAgent!(m, :firm; dynamic=false, n=1)

# Stage Game:
@enum Default DE_true DE_false
es = EnumSampler(Default, [0.9,0.1])

# State
l = addStateVariable!(hh, Float64, :l; init=rand(LogNormal(), n_hh))
k = addStateVariable!(hh, Float64, :k; init=rand(LogNormal(), n_hh))
A = addStateVariable!(m, Float64, :A; init=rand(LogNormal()))
defaulted = addStateVariable!(hh, Default, :defaulted; init=es)

# Equilibrium Variables
pos_bounds = Bounds(lower=0.0)
w = addEquilibriumVariable!(m, Float64, :w; init=1.0, bounds=pos_bounds)
r = addEquilibriumVariable!(m, Float64, :r; init=1.0, bounds=pos_bounds)
p = addEquilibriumVariable!(m, Float64, :p; init=1.0, bounds=pos_bounds)

wealth = addDependentVariable!(hh, Float64, :wealth, stage=st_equildep)
f_wealth(w,l,r,k,δ) = w*l + (1-δ+r)*k
addRule!(hh, inputs=[w,l,r,k,δ], outputs=wealth, f=f_wealth)

# Choice
c_bounds = Bounds(lower=0.0, upper=wealth)
c = addChoiceVariable!(hh, Float64, :c; init=0.1, bounds=c_bounds)
default = addChoiceVariable!(hh, Default, :default; init=DE_false)
L = addChoiceVariable!(firm, Float64, :L; init=1.6, bounds=pos_bounds)
K = addChoiceVariable!(firm, Float64, :K; init=1.6, bounds=pos_bounds)

# Factors
addfactor!.(hh, [k, l, w, r, p, A])

# Choicedeps
Y = addDependentVariable!(firm, Float64, :Y; stage=st_choicedep)
Π = addDependentVariable!(firm, Float64, :Π; stage=st_choicedep)
function f_output(p,w,L,r,K,A,α)
    Y = A*K^α*L^(1-α)
    Π = p*Y - w*L - r*K
    return Y, Π
end
addRule!(firm, inputs=[p,w,L,r,K,A,α], outputs=[Y,Π], f=f_output)

# Equilibrium Conditions
eq1 = addDependentVariable!(m, Float64, :eq1; stage=st_choicedep)
eq2 = addDependentVariable!(m, Float64, :eq2; stage=st_choicedep)
eq3 = addDependentVariable!(m, Float64, :eq3; stage=st_choicedep)
FOC1 = addDependentVariable!(m, Float64, :FOC1; stage=st_choicedep)
FOC2 = addDependentVariable!(m, Float64, :FOC2; stage=st_choicedep)
addEquilbriumConditions!(m, [eq1,eq2,eq3, FOC1, FOC2])

f_eq1(L,l) = L - mean(l)
f_eq2(K,k) = K - mean(k)
f_eq3(Y,c,p) = Y - mean(c./p)
f_FOC1(p, A, K, L, w, α) = p*A*α*K^(α-1)*L^(1-α) - w
f_FOC2(p, A, K, L, r, α) = p*A*(1-α)*K^α*L^(-α) - r
# Y = mean(c)?
addRule!(m, inputs=[L,l], outputs=eq1, f=f_eq1)
addRule!(m, inputs=[K,k], outputs=eq2, f=f_eq2)
addRule!(m, inputs=[Y,c,p], outputs=eq3, f=f_eq3)
addRule!(m, inputs=[p,A,K,L,w,α], outputs=FOC1, f=f_FOC1)
addRule!(m, inputs=[p,A,K,L,r,α], outputs=FOC2, f=f_FOC2)
#addRule!(m, inputs=Π, outputs=eq3, f=f_eq3) # Walras

# Payoff
u_hh(c, p) = log(c/p)
addPayoffRule!(hh, inputs=[c, p], f=u_hh)
u_firm(Π) = Π
addPayoffRule!(firm, inputs=Π, f=u_firm)

# Shock
ε = addShockVariable!(hh, Float64, :ε; dist=Normal(0, 0.01))
χ = addShockVariable!(m, Float64, :χ; dist=Normal(0, 0.01))

# Transition
f_l(l,ε,ρ,ν) = exp(ν*log(l) + ρ + ε)
addRule!(hh, inputs=[l,ε,ρ,ν], outputs=l, f=f_l)
f_k(wealth,c) = wealth - c
addRule!(hh, inputs=[wealth,c], outputs=k, f=f_k)
f_defaulted(default) = default
addRule!(hh, inputs=default, outputs=defaulted, f=f_defaulted)
f_A(A,χ,μ,η) = exp(μ*log(A) + η + χ)
addRule!(m, inputs=[A,χ,μ,η], outputs=A, f=f_A)
