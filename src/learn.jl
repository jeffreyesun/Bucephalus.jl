
import ForwardDiff

#using Flux

# Large agents take into account the GE effects of their actions, i.e.
#   the effects of their actions on free variables. Small agents do not.
# Watch out for too-similar batches. Agents shouldn't learn to cooperate.
# Factors are factors for V, *not* for u. i.e. Can use autodiff to figure out u'.

#getrulesconnecting(in::AbstractVariable, out::AbstractVariable) = getrulesconnecting(AbstractVariable[in], out)
#function getcrulesconnecting(md::ModelData, ins::Vector{AbstractVariable}, out::AbstractVariable)
#end

############################
# Pretrain Neural Networks #
############################

#=
"Train policy network to produce given initial choice values given initial state."
function pretrain(md::ModelData, a::Agent, P_nn::PNet)
    initmat = hcat[init(var) for var in outputs(nn)]
    #TODO Need to make sure init values vary w/r/t inputs
    for i in 1:100
        error = @diff mean(run_NN(md, a, nn; usefuturestate=true) .- initmat)
        @show error
        for param in params(P_nn)
            ∇param = grad(error, param)
            param .-= lr*∇param
        end
    end
end
=#

#################
# Compute Error #
#################

#TODO (md::ModelData, a::Agent)

"Compute the one-period V error: u_t + βV_{t+1} - V_t. Returns an n*1 matrix."
function loss_V(md::ModelData, a::Agent; ds::Schema=nothing)
    b = brain(md, a)
    _V_nn = V_nn(b)
    u = getpayoffval(agentdata(md, a); ds)
    return abs.(u .+ β(b)*run_NN(md, a, _V_nn; ds, usefuturestate=true, prealloc=false) .- run_NN(md, a, _V_nn; ds, prealloc=false))
    #TODO Should I update based on future V dependent on params, or future V fixed?
    # return run_NN(md, V_nn; ds) + loss_P(md, P_nn; ds)
end

∇loss_V(md::ModelData, a::Agent) = @diff mean(loss_V(md, a))

#function ∇loss_V(ad::AgentData; ds::Schema=nothing)
#    b = brain(ad)
#    _V_nn = V_nn(b)
#    u = getpayoffval(ad; ds)
#    V_supposed = u .+ β(b)*run_NN(ad, _V_nn; ds, usefuturestate=true)
#    return @diff mean(abs.(V_supposed .- run_NN(ad, _V_nn; ds)))
#end

"Compute the one-period P error: -u - βV_{t+1}"
function loss_P(md::ModelData, a::Agent; ds::Schema)
    ad = agentdata(md, a)
    b = brain(ad)
    _V_nn = V_nn(b)
    _P_nn = P_nn(b)
    u = getpayoffval(ad; ds)
    return - u .- β(b).*run_NN(md, a, _V_nn; ds, usefuturestate=true)
end

lowerboundlong(md, var) = (val = boundvalue(md, lowerbound(var)) ; isa(val, Vector) ? val : fill(val, n(var)))
upperboundlong(md, var) = (val = boundvalue(md, upperbound(var)) ; isa(val, Vector) ? val : fill(val, n(var)))

"Compute the derivative of loss_P with respect to the parameters of P_nn, via continuous choice variables."
function ∇loss_P(md::ModelData, a::Agent; ds::DualSchema)
    b = brain(md, a)
    ad = agentdata(md, a)
    _P_nn = P_nn(b)
    loss = loss_P(md, a; ds) #[n,k]
    _choicevars = outvars(_P_nn) #[k]
    # Multiply by float.(loss) to capture quadratic loss
    ∂loss∂choice = jacobian(vec(loss), _choicevars, ds)#.*float.(loss) #[n,k]
    nn_out = run_NN_pretransform(md, a, _P_nn) #[n,k]
    # Compute the derivative of choices with respect to nn_out
    lowermat = reduce(hcat, lowerboundlong.(md, _choicevars)) #[n,k]
    uppermat = reduce(hcat, upperboundlong.(md, _choicevars)) #[n,k]
    ∂choice∂nnout = from_R′.(nn_out, lowermat, uppermat) #[n,k]
    loss_float = float.(loss) #TODO Preallocate
    ∂loss∂P_params = @diff loss_fordiff(md, a, _P_nn, ∂loss∂choice, ∂choice∂nnout, loss_float)
    return ∂loss∂P_params
end


selectioninds(md, a, choicevar) = Int.(getval(md, choicevar))

function get_nonchoiceprob_inds(md, a, P_nn::PNet)
    _outvars = filter(!ischoiceprob, outvars(P_nn))
    nonchoiceprob_inds = [findfirst(==(outvar), _outvars) for outvar in _outvars]
    return nonchoiceprob_inds
end

#TODO Cache/memoize this
function get_choiceprob_inds(md, a, choicevar, P_nn::PNet)
    _choiceprobs = choiceprobs(choicevar)
    _outvars = outvars(P_nn)
    choiceprob_inds = [findfirst(==(probvar), _outvars) for probvar in components(_choiceprobs)]
end

"
Get p(x) for each x that actually got chosen.
Formally, this is the likelihood of the discrete choice realization.
"
function likelihood_p(md, a, choicevar, nnout, P_nn::PNet)
    discrete_inds = get_choiceprob_inds(md, a, choicevar, P_nn)
    p = nnout[:,discrete_inds]
    p = RtoR⁺_nocheck.(p)
    p = p./sum(p, dims=2)
    indices = [CartesianIndex(i, si) for (i,si) in enumerate(selectioninds(md, a, choicevar))]
    likelihood_p = p[indices]
    #NOTE Due to a bug in Knet, p MUST be indexed like this.
    return likelihood_p
end

"
Return a function of nnout tangent to the loss function, incorporating both continuous and discrete
choice variables as described below:

∇_θ𝔼[f(x,a(θ))] = ∇_θ∑_x p(x|θ)f(x,a(θ))
= ∑_x ∇_θp(x|θ)f(x, a(θ)) + p(x|θ)∇_θf(x,a(θ))
= ∑_x p(x|θ)∇_θlogp(x|θ)f(x,a(θ)) + p(x|θ)∇_θf(x,a(θ))
= 𝔼[∇_θlogp(x|θ)f(x,a(θ)) + ∇_θf(x,a(θ))]
"
function loss_fordiff(md::ModelData, a::Agent, P_nn::PNet, ∂loss∂choice::Matrix{Float64}, ∂choice∂nnout::Matrix{Float64}, loss::Matrix{Float64})
    nnout = run_NN_pretransform(md, a, P_nn; prealloc=false)
    
    loss_fordiff = ∂loss∂choice.*∂choice∂nnout.*nnout
    loss_fordiff = sum(loss_fordiff, dims=2)

    for choicevar in localdiscretechoicevars(a)
        loss_fordiff += loss .* log.(likelihood_p(md, a, choicevar, nnout, P_nn)) #XXXX
    end
    
    # Sum over the contributions of individual variables, average over agents
    return mean(loss_fordiff)
end

##########################
# Update Neural Networks #
##########################

# One Period, One Draw #
#----------------------#

"Update V based on one period"
function sgdupdate_V!(md::ModelData, a::Agent, _V_nn::VNet=V_nn(md, a); lr=.1)
    diffV_loss = ∇loss_V(md, a)
    for param in params(_V_nn)
        ∇param = grad(diffV_loss, param)
        param .-= lr*∇param
    end
    for d in layers(_V_nn)
        d.w_bare .= d.w
        d.b_bare .= d.b
    end
    #TODO Split difference if loss worsens
    return mean(loss_V(md, a))
    #TODO return mean(value(diffV_loss))
end

"Update P based on one period"
function sgdupdate_P!(md::ModelData, a::Agent, _P_nn::PNet=P_nn(md, a); lr=0.1)
    dsA = dualschema(md, :A)
    diffP_loss = ∇loss_P(md, a; ds=dsA)
    sumgrad = 0.0
    for param in params(_P_nn)
        ∇param = grad(diffP_loss, param)
        param .-= lr*∇param
        sumgrad += sum(abs.(∇param))
    end
    for d in layers(_P_nn)
        d.w_bare .= d.w
        d.b_bare .= d.b
    end
    return sumgrad #XXXX
    #return mean(loss_P(md, a; ds=nothing))
end

"Update V for all dynamic agents"
function sgdupdate_V!(md::ModelData)
    losses = Float64[]
    for a in agents(model(md)) if isdynamic(a)
        loss = sgdupdate_V!(md, a)
        push!(losses, loss)
    end end
    return only(losses)
end

# One Period, Many Draws, Optimization Style #
#--------------------------------------------#

"Draw shocks multiple times (stages st_shock, st_nextstate) to better estimate 𝔼[∇loss_V], 𝔼[∇loss_P].
Doesn't store all values of V. Instead stores all gradients. mean(gradient(loss)) = gradient(mean(loss)) by linearity.
Actually, I'm taking abs(error), and mean(gradient(abs(error))) != gradient(abs(mean(error))).
Can I set this up as rootfinding instead of optimization for V?"
function sgd_update_batch!(md::ModelData, a::Agent, batch_size::Int=10, _V_nn::VNet=V_nn(md, a), _P_nn=P_nn(md, a); lr=0.1)
    dsA = dualschema(md, :A)
    diffV_loss = ∇loss_V(md, a)
    diffP_loss = ∇loss_P(md, a; ds=dsA)
    diffV_loss_mats = [grad(diffV_loss, param) for param in params(_V_nn)]
    diffP_loss_mats = [grad(diffP_loss, param) for param in params(_P_nn)]
    for draw = 1:(batch_size-1)
        drawshocks!(md)
        setval!.(md, allvars(model(md), st_shock); ds_out=dsA)
        computedeps!(md, st_nextstate, dsA)
        dedual!(md, dsA)
        diffV_loss = ∇loss_V(md, a)
        diffP_loss = ∇loss_P(md, a; ds=dsA)
        for (i,param) in enumerate(params(_V_nn))
            diffV_loss_mats[i] .+= grad(diffV_loss, param)
        end
        for (i,param) in enumerate(params(_P_nn))
            #=
            if param == bend
                #println("MID")
                #@show diffP_loss_mats[i]
            end
            =#
            _grad = grad(diffP_loss, param)
            @assert !any(isnan.(_grad))
            #=
            if any(isnan.(_grad))
                @show param
                @show _grad
                error()
            end
            =#
            diffP_loss_mats[i] .+= _grad
        end
    end
    for (i,param) in enumerate(params(_V_nn))
        diffV_loss_mats[i] ./= batch_size
        @. param -= lr/0.1*diffV_loss_mats[i]
    end
    for (i,param) in enumerate(params(_P_nn))
        diffP_loss_mats[i] ./= batch_size
        @. param -= lr*diffP_loss_mats[i]
    end

    #dcdc = diffP_loss_mats[end][3]
    #dcdh = diffP_loss_mats[end][4]
    #@show dcdc
    #@show dcdh

    #mean_diffP = sum(sum.([abs.(m) for m in diffP_loss_mats]))
    #@show mean_diffP
    #return diffP_loss_mats
    return diffV_loss_mats
end

# One Period, Many Draws, Rootfinding Style #
#-------------------------------------------#

function sgd_update_batch_rf!(md::ModelData, a::Agent, batch_size::Int=10, _V_nn::VNet=V_nn(md, a), _P_nn=P_nn(md, a); lr=0.1)
    dsA = dualschema(md, :A)

    u = getpayoffval(agentdata(md, a))
    _β = β(brain(md,a))
    Vtestmean = u .+ _β.*run_NN(md, a, _V_nn; usefuturestate=true, prealloc=false)
    diffP_loss = ∇loss_P(md, a; ds=dsA)
    diffP_loss_mats = [grad(diffP_loss, param) for param in params(_P_nn)]
    for draw = 1:(batch_size-1)
        drawshocks!(md)
        setval!.(md, allvars(model(md), st_shock); ds_out=dsA)
        computedeps!(md, st_shockdep, dsA)
        computedeps!(md, st_nextstate, dsA)
        dedual!(md, dsA)
        diffV_loss = ∇loss_V(md, a)
        diffP_loss = ∇loss_P(md, a; ds=dsA)
        Vtestmean .+= u .+ _β.*run_NN(md, a, _V_nn; usefuturestate=true, prealloc=false)
        for (i,param) in enumerate(params(_P_nn))
            diffP_loss_mats[i] .+= grad(diffP_loss, param)
        end
    end
    Vtestmean ./= batch_size
    diffV_loss = @diff mean(abs.(Vtestmean .- run_NN(md, a, _V_nn; prealloc=false)))
    V_loss = mean(loss_V(md, a))
    @show V_loss
    for h=0:5
        h == 5 && error("All adjustment worsen V_loss.")
        for param in params(_V_nn)
            ∇param = grad(diffV_loss, param)
            param .-= lr/2^h*∇param
        end
        if (V_loss_new = mean(loss_V(md, a))) > V_loss
            println("V_loss got worse: $V_loss_new. Retrying with smaller lr.")
            for param in params(_V_nn)
                ∇param = grad(diffV_loss, param)
                param .+= lr/2^h*∇param
                #TODO Don't undo the subtraction here
            end
        else
            break
        end
    end

    for (i,param) in enumerate(params(_P_nn))
        diffP_loss_mats[i] ./= batch_size
        @. param -= lr*diffP_loss_mats[i]
    end
end

# Many Periods, One Draw #
#------------------------#

function sgd_update_simulate!(md::ModelData, a::Agent, sample_length::Int=10, _V_nn::VNet=V_nn(md, a); lr=0.1)
    ad = hostdata(md, a)
    _β = β(brain(ad))
    u = getpayoffval(ad)
    #TODO Support multiple agents
    checkpointstate!(md)
    for t = 1:(sample_length-1)
        iterate!(md)
        u .+= _β^t .* getpayoffval(ad)
    end
    restorecheckpoint!(md)
    @inline LHS() = u .+ _β^sample_length*run_NN(md, a, _V_nn; usefuturestate=true, prealloc=false)
    @inline RHS() = run_NN(md, a, _V_nn; prealloc=false)
    diffV_loss = @diff mean(abs.(LHS() .- RHS()))
    for param in params(_V_nn)
        ∇param = grad(diffV_loss, param)
        param .-= lr*∇param
    end
end

# Many Periods, Many Draws, Rootfinding Style #
#---------------------------------------------#

function sgd_update_samplebatch!(
    md::ModelData, a::Agent, batch_size::Int=10, sample_length::Int=5, _V_nn=V_nn(md, a), _P_nn=P_nn(md, a); lr=0.1
)
    dsA = dualschema(md, :A)
    ad = hostdata(md, a)
    _β = β(brain(ad))
    βV1mean = zeros(n(a), 1)
    diffP_loss_mats = [zeros(size(param)) for param in params(_P_nn)]
    checkpointstate!(md)
    # Iterate over draws
    for draw = 1:batch_size
        # Record policy function gradients
        diffP_loss = ∇loss_P(md, a; ds=dsA)
        for (i,param) in enumerate(params(_P_nn))
            diffP_loss_mats[i] .+= grad(diffP_loss, param)
            #TODO Try doing this for each simulated period, not just the base period.
        end
        # Generate sample path
        for t = 1:(sample_length-1)
            iterate!(md)
            βV1mean .+= _β^t .* getpayoffval(ad)
        end
        # Add future V to sample value
        βV1mean .+= _β^sample_length.*run_NN(md, a, _V_nn; usefuturestate=true, prealloc=false)
        draw < batch_size && restorecheckpoint!(md)
        iterate!(md, false) #TODO Restore all vars for 1/sample_length speedup
    end
    # Update policy function parameters
    for (i, param) in enumerate(params(_P_nn))
        diffP_loss_mats ./= batch_size #*sample_length
        @. param -= lr*diffP_loss_mats[i]
    end
    # Generate simulated V
    βV1mean ./= batch_size
    u = getpayoffval(ad)
    diffV_loss = @diff mean(abs.(u .+ βV1mean .- run_NN(md, a, _V_nn; prealloc=false)))

    # Update value function parameters
    V_loss = mean(loss_V(md, a))
    P_loss = mean(loss_P(md, a; ds=nothing))
    @show V_loss
    @show P_loss
    #=
    for h=0:5
        h == 5 && error("All adjustments worsen V_loss.")
        for param in params(_V_nn)
            ∇param = grad(diffV_loss, param)
            param .-= lr/2^h*∇param
        end
        if (V_loss_new = mean(loss_V(md, a))) > V_loss
            println("V_loss got worse: $V_loss_new. Retrying with smaller lr.")
            for param in params(_V_nn)
                ∇param = grad(diffV_loss, param)
                param .+= lr/2^h*∇param
                #TODO Don't undo the subtraction here
            end
        else
            break
        end
    end
    =#
end

##################
# Run Simulation #
##################

"Get to equilibrium given initial V_nn and P_nn"
function burn_in(md::ModelData; T_burn=1000)
    # Burn-in for k
    means = Float64[]

    println("Burning in...")
    for t=1:T_burn
        print("\rt=$t")
        iterate!(md)
    end
end

"Train value function, given initial policy."
function train_V(md::ModelData; T_trainV=1000)
    means = Float64[]
    V_losses = Float64[]

    println("Training V...")
    for t=1:T_trainV
        local Vloss
        for h=1:20
            iterate!(md)
            Vloss = sgdupdate_V!(md)
            push!(V_losses, Vloss)
            print("\rt=$t, Vloss=$Vloss")
        end
        println("\rt=$t, Vloss=$Vloss")
        iterate!(md)
    end

    return means, V_losses
end

"Train policy function while updating value function and simulating forward."
function train_PV(md::ModelData; T_trainPV=Inf)
    dynamicagent = only(filter(isdynamic, agents(model(md))))

    println("Training P and V...")
    t = 0
    while t < T_trainPV
        sgd_update_samplebatch!(md, dynamicagent)
        t += 1
    end
end

"Run model, learning continuously."
function train(md::ModelData; T_burn=1000, T_trainV=1000, T_trainPV=Inf)
    burn_in(md; T_burn)
    train_V(md; T_trainV)
    train_PV(md; T_trainPV)
    
    #=
    P_losses = Float64[]
    dsA = dualschema(md, :A)
    dsB = dualschema(md, :B)
    
    Ploss = 0.0
    for episode = 1:100
        local Vloss
        for i=1:5 iterate!(md) end
        for ii=1:10
            for j=1:250
                Vloss = sgdupdate_V!(md, :hh; lr=.01)
                push!(means, mean(getval(md, :k)))
                push!(V_losses, Vloss)
                rval = getval(md, :r)
                print("\repisode=$episode, ii=$ii, Vloss=$Vloss, Ploss=$Ploss, r=$rval")
            end
            for i=1:50
                solve_equilibrium!(md)
                dedual!(md, dsB)
                setargvarsduals!(md; ds_out=dsA)
                computedeps!(md, st_choicedep, dsA)
                #drawshocks!(md)
                setval!.(md, allvars(model(md), st_shock); ds_out=dsA)
                computedeps!(md, st_nextstate, dsA)
                dedual!(md, dsA)
                Ploss = sgdupdate_P!(md, :hh; lr=.1)
                push!(P_losses, Ploss)
                rval = float.(getval(md, :r; ds=dsB))
                print("\repisode=$episode, ii=$ii, Vloss=$Vloss, Ploss=$Ploss, r=$rval")
            end
            println("\repisode=$episode, ii=$ii, Vloss=$Vloss, Ploss=$Ploss, r=$rval")
        end
        rval = getval(md, :r)
    end
    return P_losses
    =#
end

function train(m::Model; T_burn=1000, T_trainV=1000, T_trainPV=Inf)
    md = compile(m)
    train(md; T_burn, T_trainV, T_trainPV)
end