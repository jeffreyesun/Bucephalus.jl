
m = Model()

@parameter m.r = 1/0.95 - 1

n_hh = 10_000
hh = addAgent!(m, :hh; dynamic=true, n=n_hh)

@statevar hh.b = rand(LogNormal(), n_hh)

@statedep hh.wealth(b) = b + 1

cbounds = Bounds(0.0, b)
c = addChoiceVariable!(hh, Float64, :c; init=0.5, bounds=cbounds)

addPayoffRule!(hh, inputs=c, f=identity)

@transition b(b, c, r) = (b-c)*(1+r)




md = compile(m)

iterate!(md)


