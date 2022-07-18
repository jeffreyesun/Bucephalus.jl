
#### Describe

m = Model()

# Parameters
α = addParameter!(m, :α, 0.5)
δ = addParameter!(m, :δ, 0.1)
A = addParameter!(m, :A, 1.0)
ν = addParameter!(m, :ν, 0.5)
ρ = addParameter!(m, :ρ, 0.5)

# Agents
n_hh = 10_000
hh = addAgent!(m, :hh; dynamic=true, n=n_hh)
firm = addAgent!(m, :firm; dynamic=false, n=1)

# Stage Game:

# State
l = addStateVariable!(hh, Float64, :l; init=rand(LogNormal(), n_hh))
#init_dist=LogNormal()
k = addStateVariable!(hh, Float64, :k; init=rand(LogNormal(), n_hh))
#init_dist=LogNormal()

# Statedeps
Ls = addDependentVariable!(m, Float64, :Ls, stage=st_statedep)
addRule!(m, inputs=l, outputs=Ls, f=mean) #, broadcast_inputs=[false])
Ks = addDependentVariable!(m, Float64, :Ks, stage=st_statedep)
addRule!(m, inputs=k, outputs=Ks, f=mean) #, broadcast_inputs=[false])

# Equilibrium Variables
pos_bounds = Bounds(lower=0.0)
w = addEquilibriumVariable!(m, Float64, :w; init=1.0, bounds=pos_bounds)
r = addEquilibriumVariable!(m, Float64, :r; init=1.0, bounds=pos_bounds)

wealth = addDependentVariable!(hh, Float64, :wealth, stage=st_equildep)
f_wealth(w,l,r,k,δ) = w*l + (1-δ+r)*k
addRule!(hh, inputs=[w,l,r,k,δ], outputs=wealth, f=f_wealth)

# Choice
c_bounds = Bounds(lower=0.0, upper=wealth)
c = addChoiceVariable!(hh, Float64, :c; init=0.1, bounds=c_bounds)
L = addChoiceVariable!(firm, Float64, :L; init=1.6, bounds=pos_bounds)
K = addChoiceVariable!(firm, Float64, :K; init=1.6, bounds=pos_bounds)

# Factors
addfactor!.(hh, [k, l, r, Ks])

Y = addDependentVariable!(firm, Float64, :Y; stage=st_choicedep)
Π = addDependentVariable!(firm, Float64, :Π; stage=st_choicedep)
function f_output(w,L,r,K,A,α)
    Y = A*K^α*L^(1-α)
    Π = Y - w*L - r*K
    return Y, Π
end
addRule!(firm, inputs=[w,L,r,K,A,α], outputs=[Y,Π], f=f_output)

eq1 = addDependentVariable!(m, Float64, :eq1; stage=st_choicedep)
eq2 = addDependentVariable!(m, Float64, :eq2; stage=st_choicedep)
addEquilbriumConditions!(m, [eq1,eq2])

f_eq1(L,l) = L - mean(l)
f_eq2(K,k) = K - mean(k)
#f_eq3(Π) = Π
addRule!(m, inputs=[L,l], outputs=eq1, f=f_eq1)
addRule!(m, inputs=[K,k], outputs=eq2, f=f_eq2)
#addRule!(m, inputs=Π, outputs=eq3, f=f_eq3) # Walras

# Payoff

u_hh(c) = log(c)
addPayoffRule!(hh, inputs=c, f=u_hh)
u_firm(Π) = Π
addPayoffRule!(firm, inputs=Π, f=u_firm)

# Shock
ε = addShockVariable!(hh, Float64, :ε; dist=Normal(0, 0.01))

# Transition
f_l(l,ε,ρ,ν) = exp(ν*log(l) + ρ + ε)
addRule!(hh, inputs=[l,ε,ρ,ν], outputs=l, f=f_l)
f_k(wealth,c) = wealth - c
addRule!(hh, inputs=[wealth,c], outputs=k, f=f_k)
