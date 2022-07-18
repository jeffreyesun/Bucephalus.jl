include("../src/main.jl")
include("../examples/HAGE aggshocks.jl")

validate(m)
md = compile(m)
initialize!(md)
dsB = dualschema(md, :B)
advancestate!(md)
computedeps!(md, st_statedep)

firmd = hostdata(md, firm)
eqvars = allvars(model(md), st_equil)
scvars = localvars(firm, st_choice)
iter_eq!, gxinit = get_iterate_equilibrium(md)
eqvals = gxinit[1:length(eqvars)]
scvals = gxinit[length(eqvars)+1:end]
eqvals = from_R.(md, eqvars, eqvals)
scvals = from_R.(md, scvars, scvals)

f = zero(gxinit)
J = zeros(length(gxinit), length(gxinit))
h = 1e-12
e1 = zero(eqvals)
e1[1] += h
evp1 = eqvals .+ e1

iter_eq!(f, J, eqvals, scvals)
eq3val0 = float(getval(md, eq3; ds=dsB))
cval0 = float(getval(md, c; ds=dsB)[1])
f0 = copy(f)
#@show f0
iter_eq!(f, nothing, evp1, scvals)
eq3val = float(getval(md, eq3; ds=dsB))
cval = float(getval(md, c; ds=dsB)[1])
df = (f .- f0)./h
deq3 = (eq3val - eq3val0)/h
dc = (cval - cval0)/h
#@show f
#@show df
@assert all(df .≈ J[1:5, 1])
@assert deq3 ≈ derivative(md, dsB, eq3, w)
c1val = getval(md, c; ds=dsB)[1]
@assert dc ≈ derivative(dsB, c1val, w)


setval!(md, w, 1.0; ds=dsB)
choose!(md, hh, dsB)
cval0 = getval(md, c; ds=dsB)[2]
layer0 = layers(P_nn(brain(hostdata(md, hh))))[1].res[:B][1]
dc = derivative(dsB, cval0, w)
h = 1e-13
setval!(md, w, getval(md, w; ds=dsB) + h; ds=dsB)
choose!(md, hh, dsB)
cval = getval(md, c; ds=dsB)[2]
layer = layers(P_nn(brain(hostdata(md, hh))))[1].res[:B][1]
dc = float((cval - cval0))/h
dlayer = float((layer - layer0))/h
@assert dc ≈ derivative(dsB, cval0, w)
@assert dlayer ≈ derivative(dsB, layer0, w)