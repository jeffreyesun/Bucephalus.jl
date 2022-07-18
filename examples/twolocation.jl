
using Distributions

m = Model()

# Parameters
@parameter m.α = 0.5
@parameter m.ϕ1 = 1.5
@parameter m.ϕ2 = 0.5
@parameter m.F  = 0.1
@parameter m.σ  = 1.5
@parameter m.r  = 0.1
@parameter m.δ  = 0.1
@parameter m.χ  = 1.0
@parameter m.ν = 0.5
@parameter m.η = 0.5
@parameter m.H1 = 1.0

# Agents
n_hh = 1_000
hh = addAgent!(m, :hh; dynamic=true, n=n_hh)

# Markets
n_loc = 2
loc = addMarket!(m, :loc; n=n_loc)

# Stage Game
@enum Sector SE_tradable = 1 SE_nontradable = 2
es = EnumSampler(Sector, [0.5,0.5])

# State
@statevar hh.l = rand(LogNormal(), n_hh)
@statevar hh.s::Sector = es
@statevar loc.ki = [1.0, 2.0]
@statevar loc.A  = [1.0, 1.0]
@matchvar hh.residence::Match{:loc} = rand(1:n_loc, n_hh)

# Statedep

@statedep hh.lϕ(l,s,ϕ1,ϕ2) = s==1 ? l^ϕ1 : l^ϕ2
@statedep loc.w(A) = A
@statedep hh.income(lϕ,By(w,residence)) = lϕ*w
@statedep loc.L_T(ByMembers(lϕ,residence),ByMembers(s,residence)) = sum(lϕ .*(s.==SE_tradable)) / n_hh
@statedep loc.L_N(ByMembers(lϕ,residence),ByMembers(s,residence)) = sum(lϕ .*(s.==SE_nontradable)) / n_hh
@statedep loc.J(L_N,σ,F)  = L_N/(σ*F)
@statedep loc.Y_N(J,σ,F)  = J^(σ/(σ-1))*(σ-1)*F
@statedep loc.Y_T(A,L_T)  = A*L_T
@statedep loc.P(J,σ,F,w)  = σ/(σ-1) * J^(1/(1-σ)) * (w+1)
@statedep m.ρ2(r,χ,δ)     = r*χ + δ

# Equil
ρ1 = addEquilibriumVariable!(m, Float64, :ρ1; init=1.0)

# Equildep
@equildep hh.wealth(income) = income
@equildep loc.ρ(ρ1,ρ2,ki) = ki == 1 ? ρ1 : ρ2

# Choice
posbounds = Bounds(0.0, Inf)
c_share = addChoiceVariable!(hh, Float64, :c_share; init=0.5, bounds=posbounds)
h_share = addChoiceVariable!(hh, Float64, :h_share; init=0.5, bounds=posbounds)
makeChoiceVariable!(hh, residence)
makeChoiceVariable!(hh, s)

# Choicedep
@choicedep hh.c(c_share,h_share,wealth,By(P,residence)) = c_share/(c_share+h_share)*wealth/P
@choicedep hh.h(h_share,c_share,wealth,By(ρ,residence)) = h_share/(c_share+h_share)*wealth/ρ
@choicedep loc.H(ByMembers(h,residence)) = sum(h) / n_hh

# Eqconds
@choicedep m.eq1(H,H1) = H[1] - H1
addEquilbriumConditions!(m, [eq1])

# Factors
addfactor!.(hh, [l, s, By(ki, residence)])

# Payoff
#f_u(c, h)
addPayoffRule!(hh, inputs=c, f=identity)

# Shock
ε = addShockVariable!(hh, Float64, :ε; dist=Normal(0,0.05))

# Transition
@macroexpand @transition l(l,ε,η,ν) = exp(ν*log(l) + η + ε)
