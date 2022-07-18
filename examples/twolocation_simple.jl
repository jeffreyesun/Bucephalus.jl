
m = Model()

# Parameters
@parameter m.r  = 0.1
@parameter m.ρ2 = 0.2
@parameter m.ν = 0.5
@parameter m.η = 0.5
@parameter m.H1 = 1.0

hh = addAgent!(m, :hh; dynamic=true, n=1_000)
#@agent(m.hh; dynamic=true, n=10_000)

# Locations
loc = addMarket!(m, :loc; n=2)
# @segment(m.loc; n=2)

# State
@statevar hh.l = rand(LogNormal(), n_hh)
#@statevar hh.l = LogNormal()
@statevar loc.ki = [1.0, 2.0]
@statevar loc.A  = [1.0, 1.0]
@matchvar hh.residence::Match{:loc} = rand(1:n_loc, n_hh)

# Statedep
@statedep loc.w(A) = A
@statedep hh.income(l,By(w,residence)) = l*w
@statedep loc.L(ByMembers(l,residence)) = sum(l) / n_hh
@statedep loc.Y(A,L) = A*L

# Equil
ρ1 = addEquilibriumVariable!(m, Float64, :ρ1; init=1.0)
#@equilvar m.ρ1 = 1.0

# Equildep
@equildep loc.ρ(ρ1,ρ2,ki) = ki == 1 ? ρ1 : ρ2

# Choice
Ibounds = Bounds(0.0, 1.0)
c_share = addChoiceVariable!(hh, Float64, :c_share; init=0.5, bounds=posbounds)
makeChoiceVariable!(hh, residence)
# @choicevar(hh.c_share = 0.5, bounds=(0.0,1.0))
# @choicevar!(hh.residence)

# Choicedep
@choicedep hh.c(c_share,income,By(P,residence)) = c_share*wealth/P
@choicedep hh.h(c_share,income,By(ρ,residence)) = (1-c_share)*wealth/ρ
@choicedep loc.H(ByMembers(h,residence)) = sum(h) / n_hh

# Eqconds
@choicedep m.eq1(H,H1) = H[1] - H1
addEquilbriumConditions!(m, [eq1])
#@equilibriumcondition m.eq1(H,H1) = H[1] - H1

# Factors
addfactor!.(hh, [l, s, By(ki, residence)])

# Payoff
addPayoffRule!(hh, inputs=c, f=identity)
#@utility!(hh.c)

# Shock
ε = addShockVariable!(hh, Float64, :ε; dist=Normal(0,0.05))
#@shock hh.ε = Normal(0,0.05)

# Transition
@transition l(l,ε,η,ν) = exp(ν*log(l) + η + ε)
