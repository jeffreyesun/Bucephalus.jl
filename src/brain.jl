######################
# Accessor Functions #
######################

# NeuralNet
invars(nn::NeuralNet) = nn.invars
chain(nn::NeuralNet) = nn.chain
layers(c::Chain) = c.layers
outvars(P_nn::PNet) = P_nn.outvars
# Brain
agent(b::Brain) = b.agent
β(b::Brain) = b.β
V_nn(b::Brain) = b.V_nn
P_nn(b::Brain) = b.P_nn

#####################
# Utility Functions #
#####################

# Inherited/Computed Properties #
#-------------------------------#

layers(nn::NeuralNet) = layers(chain(nn))
V_nn(ad::AgentData) = V_nn(brain(ad))
P_nn(ad::AgentData) = P_nn(brain(ad))
nns(b::Brain) = (V_nn(b), P_nn(b))
hasbrain(ad::AgentData) = !isnothing(brain(ad))
agentdata(md::ModelData, b::Brain) = agentdata(md, agent(b))
agentdata(md::ModelData, nn::NeuralNet) = agentdata(md, agent(nn))

###############
# Constructor #
###############

leaky_relu(x) = relu(x) + 0.1*min(0,x)

function Dense(i::Int, o::Int, f=leaky_relu)
    w = param(o,i; atype=Array{Float64})
    b = param0(o; atype=Array{Float64})
    w_bare = zeros(o,i) .= w
    b_bare = zeros(o) .= b
    res = Dict{Symbol, Matrix}()
    return Dense(w,b,f, w_bare, b_bare, res)
end

"Preallocate the result matrices of duals for layer d."
function preallocate_res!(d::Dense, n::Int, ds::DualSchema{S}) where S
    d.res[S] = fill(templatedual(ds), length(d.b_bare), n)
    return nothing
end

"Preallocate the result matrices of values for layer d."
function preallocate_res!(d::Dense, n::Int, ds::Nothing=nothing)
    o = length(d.b_bare)
    d.res[:value] = fill(zero(Float64), o, n)
    return nothing
end


############
# Defaults #
############

function V_nn_default(a::Agent)
    invars = localvars(a, st_state) #TODO valuefactors?
    #TODO Add discrete invars as one-hot
    n_in = length(invars)
    chain = Chain(
        Dense(n_in,  4n_in),
        Dense(4n_in, 4n_in),
        Dense(4n_in, 4n_in),
        Dense(4n_in, 1, identity)
    )
    return VNet(invars, chain)
end

function P_nn_default(a::Agent)
    invars = choicefactors(a)
    #TODO generalized moments
    outvars = localvars(a, st_choice)
    n_in = length(invars)
    n_out = length(outvars)
    chain = Chain(
        Dense(n_in,   4n_in),
        Dense(4n_in,  4n_in),
        Dense(4n_in,  4n_out),
        Dense(4n_out, 4n_out),
        Dense(4n_out, n_out, identity)#2sigm(x))
    )
    return PNet(invars, outvars, chain)
end

function basicbrain(a::Agent, β=0.95)
    V_nn = V_nn_default(a)
    P_nn = P_nn_default(a)
    return Brain(a, β, V_nn, P_nn)
end

function basicbrains(m)
    brainsdict = Dict{Symbol,Union{Brain,Nothing}}()
    for a in agents(m)
        brainsdict[symbol(a)] = isdynamic(a) ? basicbrain(a) : nothing
    end
    return brainsdict
end

##############
# Activation #
##############

"Execute a single neural layer `d`, storing the result in the entry of d.res corresponding to `ds`.
Note: mat(x) here is an o*n matrix, where n is the number of input vectors being evaluated simultaneously."
function rundense!(d::Dense, x, ds::DualSchema{S}, prealloc::Bool=true) where S
    #@assert prealloc=true
    #TODO Only run this on first-order Duals, and replace mul! with an optimized version.
    res = d.res[S]
    mul!(res, d.w_bare, mat(x))
    res .+= d.b_bare
    res .= d.f.(res)
    return res
end
"Execute a single neural layer `d`. If `prealloc==false``, don't use preallocated arrays for the result.
For reversediff to work, `prealloc` must be `false`."
function rundense!(d::Dense, x, ds::Nothing, prealloc::Bool=true)
    if prealloc
        res = d.res[:value]
        res .= d.w*mat(x)
        res .+= d.b
        res .= d.f.(res)
        return res
    else
        return d.f.(d.w*mat(x) .+ d.b)
    end
end

(c::Chain)(x, ds::Schema, prealloc::Bool=true) = (for l in c.layers x = rundense!(l, x, ds, prealloc) end; x)

function run_NN_pretransform(nn::NeuralNet, x, ds::Schema, prealloc::Bool=true)
    return chain(nn)(x, ds, prealloc)' # [n,k]
    #'
end

function get_NN_inputs(md::ModelData, a::Agent, nn::NeuralNet;
    ds::Schema=nothing, usefuturestate::Bool=false, prealloc::Bool=true
)
    vars = invars(nn)
    dualtype = isnothing(ds) ? Float64 : typeof(dual(ds,0,NaN))
    varsdata = zeros(dualtype, length(vars), n(a))
    #TODO Preallocate this matrix
    for (k,var) in enumerate(vars)
        vardata = getval(md, var; ds=Try(ds), usefuturestate)
        if eltype(vardata) <: Enum
            varsdata[k,:] .= Float64.(Int.(vardata))
            # TODO Do this as a one-hot
        elseif eltype(vardata) <: Match
            varsdata[k,:] .= Float64.(Int.(vardata))
        elseif !(eltype(vardata) <: Dual) && !isnothing(ds)
            varsdata[k,:] .= dual.(ds, 0, vardata)
        elseif eltype(vardata) <: Dual && isnothing(ds)
            varsdata[k,:] .= value.(vardata)
        else
            varsdata[k,:] .= vardata
        end
        # If agent(var) != a, do something different
    end
    return varsdata
end


"Run the neural network `nn`, without applying the final transformation.
Note: If `ds` is a DualSchema, the AutoGrad machinery will not be engaged."
function run_NN_pretransform(md::ModelData, a::Agent, nn::NeuralNet;
    ds::Schema=nothing, usefuturestate::Bool=false, prealloc::Bool=true
)
    varsdata = get_NN_inputs(md, a, nn; ds, usefuturestate, prealloc)
    return run_NN_pretransform(nn, varsdata, ds, prealloc)
end

function run_NN(md::ModelData, a::Agent, nn::NeuralNet; ds::Schema=nothing, usefuturestate::Bool=false, prealloc::Bool=true)
    nn_out = run_NN_pretransform(md, a, nn; ds, usefuturestate, prealloc) #[n,k]

    return nn isa PNet ? reduce(hcat, from_R.(md, outvars(nn), eachcol(nn_out); ds)) : nn_out
end

#######
# API #
#######

function get_choice(md::ModelData, a::Agent; ds::Schema=nothing)
    _P_nn = P_nn(md, a)
    return Dict(symbol.(outvars(_P_nn)) .=> eachcol(run_NN(md, a, _P_nn; ds)))
end
