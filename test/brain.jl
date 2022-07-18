# Test ∂loss∂params

hhd = agentdata(md, hh)
_P_nn = P_nn(brain(hhd))
_V_nn = V_nn(brain(hhd))
dsA = dualschema(md, :A)
dsB = dualschema(md, :B)
u = payoffvar(hh)

loss1 = deepcopy(loss_P(hhd; ds=dsA))
loss1mean = mean(loss1)

∂loss∂params = ∇loss_P(hhd; ds=dsA)
∇loss1 = Knet.value(∂loss∂params)

params1 = first(params(_P_nn))
param1 = params1[1]
gradparam1 = grad(∂loss∂params, params1)[1]

params1[1] += 0.001

iteratechoice!(md, dsB)
computedeps!(md, st_choicedep, dsB)
dedual!(md, dsB)
setargvarsduals!(md; ds_out=dsA)
computedeps!(md, st_choicedep, dsA)
#drawshocks!(md)
setval!.(md, allvars(model(md), st_shock); ds_out=dsA)
computedeps!(md, st_nextstate, dsA)
dedual!(md, dsA)

loss2 = deepcopy(loss_P(hhd; ds=dsA))
loss2mean = mean(loss2)

loss1mean
loss1mean + 0.001*gradparam1

loss1mean + 0.001*gradparam1 - loss2mean
loss1mean - loss2mean

∂loss∂params2 = ∇loss_P(hhd; ds=dsA)
∇loss2 = Knet.value(∂loss∂params2)

∇loss1 + 0.0001*gradparam1