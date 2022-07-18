
m = Model()

# Parameters
α = addParameter!(m, :α, 0.5)
δ = addParameter!(m, :δ, 0.1)
ν = addParameter!(m, :ν, 0.5)
ρ = addParameter!(m, :ρ, 0.5)
μ = addParameter!(m, :μ, 0.5)
η = addParameter!(m, :η, 0.5)
A = addParameter!(m, :A, 1.0)

# Agents
n_hh = 10_000
hh = addAgent!(m, :hh; dynamic=true, n=n_hh)

# State
l = addStateVariable!(hh, Float64, :l; init=rand(LogNormal(), n_hh))
k = addStateVariable!(hh, Float64, :k; init=rand(LogNormal(), n_hh))

# StateDep
w = addDependentVariable!(m, Float64, :w; stage=st_statedep)
r = addDependentVariable!(m, Float64, :r; stage=st_statedep)
f_w(A,α,k,l) = A*(mean(k)/mean(l))^α
f_r(A,α,k,l) = A*(mean(k)/mean(l))^(α-1)
addRule!(m, inputs=[A,α,k,l], outputs=w, f=f_w)
addRule!(m, inputs=[A,α,k,l], outputs=r, f=f_r)

wealth = addDependentVariable!(hh, Float64, :wealth, stage=st_statedep)
f_wealth(w,l,r,k,δ) = w*l + (1-δ+r)*k
addRule!(hh, inputs=[w,l,r,k,δ], outputs=wealth, f=f_wealth)

# Choice
pos_bounds = Bounds(lower=0.0)
c_bounds = Bounds(lower=0.0, upper=wealth)
c = addChoiceVariable!(hh, Float64, :c; init=0.1, bounds=c_bounds)

# Factors
addfactor!.(hh, [k, l, w, r, A])

# Payoff
u_hh(c) = log(c)
addPayoffRule!(hh, inputs=c, f=u_hh)

# Shock
ε = addShockVariable!(hh, Float64, :ε; dist=Normal(0, 0.1))

# Transition
f_l(l,ε,ρ,ν) = exp(ν*log(l) + ρ + ε)
addRule!(hh, inputs=[l,ε,ρ,ν], outputs=l, f=f_l)
f_k(wealth,c) = wealth - c
addRule!(hh, inputs=[wealth,c], outputs=k, f=f_k)
