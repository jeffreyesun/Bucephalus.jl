
m = Model()

# Parameters
@parameter m.α = 0.5
@parameter m.δ = 0.1
@parameter m.ν = 0.5
@parameter m.ρ = 0.5
@parameter m.μ = 0.5
@parameter m.η = 0.5

# Agents
n_hh = 10_000
hh = addAgent!(m, :hh; dynamic=true, n=n_hh)
firm = addAgent!(m, :firm; dynamic=false, n=1)

# Markets
n_loc = 5
loc = addMarket!(m, :loc; n=n_loc)

# Stage Game:
@enum Default DE_true DE_false
es = EnumSampler(Default, [0.9,0.1])

# State
@statevar hh.l = rand(LogNormal(), n_hh)
@statevar hh.k = rand(LogNormal(), n_hh)
@statevar m.A = rand(LogNormal())
@statevar loc.elevation = rand(LogNormal(), n_loc)
@statevar hh.defaulted::Default = es
@matchvar hh.residence::Match{:loc} = rand(1:n_loc, n_hh)

# Statedep
@statedep m.meanelev(elevation) = mean(elevation)
@statedep loc.elev_diff(elevation, meanelev) = elevation-meanelev
@statedep hh.reselev(By(elevation, residence)) = elevation
@statedep loc.K_byres(ByMembers(k, residence)) = mean(k)

# Equilibrium Variables
pos_bounds = Bounds(lower=0.0)
w = addEquilibriumVariable!(m, Float64, :w; init=1.0, bounds=pos_bounds)
r = addEquilibriumVariable!(m, Float64, :r; init=1.0, bounds=pos_bounds)
p = addEquilibriumVariable!(m, Float64, :p; init=1.0, bounds=pos_bounds)

@equildep hh.wealth(w,l,r,k,δ) = w*l + (1-δ+r)*k

# Choice
c_bounds = Bounds(lower=0.0, upper=wealth)
c = addChoiceVariable!(hh, Float64, :c; init=0.1, bounds=c_bounds)
makeChoiceVariable!(hh, defaulted)
L = addChoiceVariable!(firm, Float64, :L; init=1.6, bounds=pos_bounds)
K = addChoiceVariable!(firm, Float64, :K; init=1.6, bounds=pos_bounds)

# Factors
addfactor!.(hh, [k, l, w, r, p, A, By(elev_diff, residence)])

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
@choicedep m.eq1(L,l) = L - mean(l)
@choicedep m.eq2(K,k) = K - mean(k)
@choicedep m.eq3(Y,c,p) = Y - mean(c./p)
@choicedep m.FOC1(p, A, K, L, w, α) = p*A*α*K^(α-1)*L^(1-α) - w
@choicedep m.FOC2(p, A, K, L, r, α) = p*A*(1-α)*K^α*L^(-α) - r

addEquilbriumConditions!(m, [eq1,eq2,eq3, FOC1, FOC2])

# Y = mean(c)?
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
@transition l(l,ε,ρ,ν) = exp(ν*log(l) + ρ + ε)
@transition k(wealth,c) = wealth - c
@transition A(A,χ,μ,η) = exp(μ*log(A) + η + χ)
