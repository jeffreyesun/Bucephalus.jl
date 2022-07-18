######################
# Accessor Functions #
######################

# HostData
@inline description(hd::HostData) = hd.description
@inline modeldata(hd::HostData) = hd.modeldata
@inline symbol(hd::HostData) = symbol(description(hd))
@inline getvar(hd::HostData, s::Symbol) = hd.description.vars[s]
    # VarData Dicts
@inline vardatadict(hd::HostData)::Dict = hd.vardatacache.vardata
@inline nextstatedatadict(hd::HostData)::Dict = hd.vardatacache.nextstatedata
@inline vardatadualdict(hd::HostData)::Dict = hd.vardatacache.vardatadual
@inline vardatadualdict(hd::HostData, level::Symbol) = hd.vardatacache.vardatadual[level]
@inline nextstatedatadualdict(hd::HostData)::Dict = hd.vardatacache.nextstatedatadual
@inline nextstatedatadualdict(hd::HostData, level::Symbol) = hd.vardatacache.nextstatedatadual[level]
@inline statedatacheckpointdict(hd::HostData)::Dict = hd.vardatacache.statedatacheckpoint
    # Local VarData
@inline _localvardata(hd::HostData, var::AbstractVariable, ds::Nothing, nextstate::Val{false}) =
    vardatadict(hd)[symbol(var)]
@inline _localvardata(hd::HostData, var::StateVariable, ds::Nothing, nextstate::Val{true}) =
    nextstatedatadict(hd)[symbol(var)]
@inline _localvardata(hd::HostData, var::AbstractVariable, ds::DualSchema{S}, nextstate::Val{false}) where S =
    vardatadualdict(hd)[S][symbol(var)]
@inline _localvardata(hd::HostData, var::StateVariable, ds::DualSchema{S}, nextstate::Val{true}) where S =
    nextstatedatadualdict(hd)[S][symbol(var)]
@inline model(hd::HostData)::Model = hd.description.model

# ModelData
@inline model(md::ModelData) = md.description
@inline modeldata(md::ModelData) = md
@inline agentdatadict(md::ModelData) = md.agentdata
@inline agentdata(md::ModelData, a::Agent) = md.agentdata[symbol(a)]
@inline agentdata(md::ModelData) = collect(values(md.agentdata))
@inline agentdata(md::ModelData, var::AbstractVariable) = agentdata(md, agent(var))
@inline marketdatadict(md::ModelData) = md.marketdata
@inline marketdata(md::ModelData, mr::Market) = md.marketdata[symbol(mr)]
@inline marketdata(md::ModelData) = collect(values(md.marketdata))
@inline marketdata(md::ModelData, var::AbstractVariable) = marketdata(md, host(var))
@inline sortedrules(md::ModelData) = md.sortedrules
@inline sortedrules(md::ModelData, stage::Stage) = md.sortedrules[Int(stage)]
@inline dualschemasdict(md::ModelData) = md.dualschemas
@inline dualschemas(md::ModelData) = collect(values(md.dualschemas)) #TODO do lazily?
@inline schemas(md::ModelData) = [nothing, dualschemas(md)...]
@inline dualschema(md::ModelData, level::Symbol) = md.dualschemas[level]

# AgentData
@inline agent(ad::AgentData)::Agent = ad.description
@inline brain(ad::AgentData)::Union{Brain,Nothing} = ad.brain

# MarketData
@inline market(mrd::MarketData)::Market = mrd.description
@inline membersdatadict(mrd::MarketData) = mrd.membersdata
@inline membersdata(mrd::MarketData, matchvar::AbstractVariable) = mrd.membersdata[symbol(matchvar)]
@inline membersdata(md::ModelData, bym::ByMembers) = membersdata(marketdata(md, host(bym)), matchvar(bym))

# CompiledRule
@inline description(cr::CompiledRule) = cr.description
@inline schemasymbol(::Nothing) = :value
@inline schemasymbol(::DualSchema{S}) where {S} = S
@inline outvardata(cr::CompiledRule, ds::Schema) = cr.outvardata[schemasymbol(ds)]
@inline invardata(cr::CompiledRule, ds::Schema) = cr.invardata[schemasymbol(ds)]
@inline zippedouts(cr::CompiledRule, ds::Schema) = cr.zippedouts[schemasymbol(ds)]

#####################
# Utility Functions #
#####################

# Aggregation #
#-------------#

allhostdata(md::ModelData) = [agentdata(md)..., marketdata(md)..., md]
#allhostdata(md::ModelData) = reduce(vcat, agentdata(md), init=[md])

# Inherited/Computed Properties #
#-------------------------------#

n(hd::HostData) = n(description(hd))
hostdata(md::ModelData, ::Model) = md
hostdata(md::ModelData, a::Agent) = agentdata(md, a)
hostdata(md::ModelData, mr::Market) = marketdata(md, mr)
hostdata(md::ModelData, var::AbstractVariable) = hostdata(md, host(var))
function hostdata(hd::HostData, var::AbstractVariable)
    haskey(vardict(description(hd)), symbol(var)) || error("HostData `$hd` has no variable `$var`.")
    return hd
end
brain(md::ModelData, a::Agent) = brain(agentdata(md, a))
V_nn(md::ModelData, a::Agent) = V_nn(agentdata(md, a))
P_nn(md::ModelData, a::Agent) = P_nn(agentdata(md, a))
partnerhostdata(md::ModelData, matchvar::AbstractVariable) = hostdata(md, partnerhost(model(md), matchvar))

#####################
# Utility Functions #
#####################

# Resolve datastage/ds from usefuturestate/tryds #
# `datastage` refers to the stage that the data is taken from. While a variable can have stage st_state,
# we might want to access data from stage `st_nextstate`
#--------------------------------------------------#

@inline datastage(stage::Stage, usefuturestate::Bool) = stage==st_state ? usefuturestate : stage
_ds(::AbstractVariable, ds::DualSchema, ::Bool) = ds
_ds(::AbstractVariable, ds::Nothing, ::Bool) = nothing
# Note: At some point, a DualSchema will keep track of a set of variables rather than (coarser) stages.
# This function will then have to be rewritten.
"Return `ds(tds)` if `ds(tds)` covers `stage`, otherwise return `nothing`."
function _ds(var::AbstractVariable, tds::Try, usefuturestate::Bool)
    return (iscontinuous(var) && datastage(stage(var), usefuturestate) in stages(ds(tds))) ? ds(tds) : nothing
end

###################
# Data Read/Write #
###################

# Private getters and setters given VarData #
#-------------------------------------------#

@inline _getval(data::Ref) = data[]
@inline _getval(data::AbstractArray) = data #TODO Return a ReadOnlyArray
@inline _setval!(data::Ref, val) = data[] = val
@inline _setval!(data::Ref, val::AbstractArray) = data[] = only(val)
@inline _setval!(data::AbstractArray, arr) = data .= arr
@inline _setval!(data::Ref, ref::Ref) = data[] = ref[]
_setval!(data::SubArray, ::Any) = error("Cannot set value of a view. Are you trying to set a `By`?")

# Retrieve VarData for a given variable #
#---------------------------------------#

function localvardata(hd::HostData, var::AbstractVariable; ds::Schema=nothing, usefuturestate::Bool=false)
    ds = _ds(var, ds, usefuturestate)
    nextstate = usefuturestate && stage(var) == st_state
    _localvardata(hd, var, ds, Val{nextstate}())
end

function vardata(hd::HostData, var::AbstractVariable; ds::Schema=nothing, usefuturestate::Bool=false)
    hd::HostData = hostdata(hd, var)
    return localvardata(hd, var; ds, usefuturestate)
end

function vardata(md::ModelData, by::By; ds::Schema=nothing, usefuturestate=false)
    usefuturevalue = usefuturestate in (true, :valueonly)
    usefuturematch = usefuturestate in (true, :matchonly)
    return @view getval(md, partnervar(by); ds, usefuturestate=usefuturevalue)[
        Int.(getval(md, matchvar(by); usefuturestate=usefuturematch))
    ]
end

function vardata(md::ModelData, bym::ByMembers; ds::Schema=nothing, usefuturestate::Bool=false)
    _membersdata = membersdata(md, bym)
    return [
            @view getval(md, partnervar(bym); ds, usefuturestate)[Int.(_membersdata[i])]
        for i in 1:n(host(bym))
    ]
end

payoffdata(ad::AgentData; ds::Schema=nothing) = localvardata(ad, payoffvar(agent(ad)); ds)

# Read and write data for a given variable #
#------------------------------------------#

function getval(hd::HostData, var::Inputtable; ds::Schema=nothing, usefuturestate::Bool=false)
    _getval(vardata(hd, var; ds, usefuturestate))
end

function setval!(hd::HostData, var::AbstractVariable, val::FloatVal;
    ds::Schema=nothing, usefuturestate::Bool=false, ds_in::Nothing=nothing
)
    ds = _ds(var, ds, usefuturestate)
    _setval!(vardata(hd, var; ds, usefuturestate), toschema.(val, var, ds))
end

function setval!(hd::HostData, var::AbstractVariable, val::DualVal{S};
    ds::Schema=nothing, usefuturestate::Bool=false, ds_in::DualSchema{S}=ds
) where {S}
    ds = _ds(var, ds, usefuturestate)
    _setval!(vardata(hd, var; ds, usefuturestate), toschema.(val, var, ds, ds_in))
end

function setval!(hd::HostData, var::AbstractVariable, val::DiscreteVal; usefuturestate::Bool=false)
    _setval!(vardata(hd, var; usefuturestate), val)
end

@inline function setval!(hd::HostData, var::AbstractVariable;
    ds_out::Schema=nothing, usefuturestate::Bool=false, ds_in::Schema=nothing
)
    setval!(hd, var, getval(hd,var;ds=ds_in,usefuturestate); ds=ds_out, usefuturestate, ds_in)
end

getpayoffval(ad::AgentData; ds::Schema=nothing) = getval(ad, payoffvar(agent(ad)); ds)

##########################
# Move State Data around #
##########################

# Update Match Data #
#-------------------#

"Given a Match StateVariable on an agent, fill in the corresponding members vectors on the partner."
function updatemembers!(md::ModelData, var::StateVariable{<:Match})
    matchval = getval(md, var; usefuturestate=true)
    partner = partnerhost(model(m), var)

    # Build new membersdata
    membersdata_var = [Match{symbol(host(var))}[] for i=1:n(partner)]
    for i = 1:n(var)
        push!(membersdata_var[Int(matchval[i])], Match{symbol(host(var))}(i))
    end

    # Replace existing membersdata
    membersdatadict(partnerhostdata(md, var))[symbol(var)] = membersdata_var
    return nothing
end

function updatemembers!(md::ModelData)
    for var in allvars(model(m), st_state) if dtype(var) <: Match
        updatemembers!(md, var)
    end end
end

"Rewire `By` inputs of a rule."
function updatebyinputs!(md::ModelData, cr::CompiledRule)
    for ds in dualschemas(md)
        for (i, input) in enumerate(inputs(description(cr)))
            if input isa Union{By, ByMembers} && stage(input) in stages(ds)
                invardata(cr, ds)[i] = vardata(md, input; ds=ds)
            end
        end
    end
end

"Rewire rules that take `By` or `ByMembers` inputs."
function updatebyrules!(md::ModelData)
    var isa StateVariable && fillinmatches!(modeldata(hd), var, val; usefuturestate)
    for rulestage in sortedrules(md)
        for cr in rulestage
            updatebyinputs!(md, cr)
        end
    end
end

# Advance State #
#---------------#

"Move future state variable data to current state variable data for host `hd`."
function advancelocalstate!(hd::HostData)
    for (s, arr) in nextstatedatadict(hd)
        _setval!(localvardata(hd, getvar(hd,s)), arr)
    end
    return nothing
end

"Move future state variable data to current state variable data. i.e. advance one period."
function advancestate!(md::ModelData)
    advancelocalstate!.(allhostdata(md))
    updatemembers!(md)
    updatebyrules!(md)
end

# Checkpoint and Restore State #
#------------------------------#

"Stash local state to restore later"
function checkpointlocalstate!(hd::HostData)
    for (s, arr) in statedatacheckpointdict(hd)
        _setval!(arr, localvardata(hd, getvar(hd,s)))
    end
end

checkpointstate!(md::ModelData) = (checkpointlocalstate!.(allhostdata(md)); nothing)

"Restore local state from checkpoint"
function restorelocalcheckpoint!(hd::HostData)
    for (s, arr) in statedatacheckpointdict(hd)
        _setval!(localvardata(hd, getvar(hd,s)), arr)
    end
end

function restorecheckpoint!(md::ModelData)
    restorelocalcheckpoint!.(allhostdata(md))
    updatemembers!(md)
    updatebyrules!(md)
end

################
# Constructors #
################

function VarDataCache()
    vardata = Dict{Symbol,Union{Vector,Ref}}()
    nextstatedata = Dict{Symbol,Union{Vector,Ref}}()
    vardatadual = Dict{Symbol,Dict{Symbol,Union{Vector,Ref}}}()
    nextstatedatadual = Dict{Symbol,Dict{Symbol,Union{Vector,Ref}}}()
    statedatacheckpoint = Dict{Symbol,Union{Vector,Ref}}()
    return VarDataCache(vardata, nextstatedata, vardatadual, nextstatedatadual, statedatacheckpoint)
end

function AgentData(a::Agent, md::ModelData, brain::Union{Nothing,Brain})
    vardatacache = VarDataCache()
    return AgentData(a, md, vardatacache, brain)
end

function MarketData(mr::Market, md::ModelData)
    vardatacache = VarDataCache()
    membersdata = Dict{Symbol,Vector{Vector{Match}}}()
    return MarketData(mr, md, vardatacache, membersdata)
end

function ModelData(m::Model, brainsdict::Dict{Symbol,<:Union{Nothing,Brain}})
    agentdata = Dict{Symbol,AgentData}()
    marketdata = Dict{Symbol,MarketData}()
    vardata = Dict{Symbol,Union{Vector,Ref}}()
    nextstatedata = Dict{Symbol,Union{Vector,Ref}}()
    vardatadual = Dict{Symbol,Dict{Symbol,Union{Vector,Ref}}}()
    nextstatedatadual = Dict{Symbol,Dict{Symbol,Union{Vector,Ref}}}()
    statedatacheckpoint = Dict{Symbol,Union{Vector,Ref}}()
    vardatacache = VarDataCache(vardata, nextstatedata, vardatadual, nextstatedatadual, statedatacheckpoint)
    sortedrules = Vector{CompiledRule}[]
    dualschemas = Dict{Symbol,DualSchema}()
    md = ModelData(m, agentdata, marketdata, vardatacache, sortedrules, dualschemas)
    merge!(agentdata, Dict(s => AgentData(a, md, brainsdict[s]) for (s,a) in agentdict(m)))
    merge!(marketdata, Dict(s => MarketData(mr, md) for (s,mr) in marketdict(m)))
    return md
end

# When adding inplace stuff
# if broadcast_output
#   n = rule.host.n
#   outargs = [[RefArray(out, i) for i=1:n] for outvec in outputvecs]
# else
#   outargs = [isheterogeneous(out.host) ? outvec : outvec[] for outvec in outputvecs]
# end
# args = vcat(outargs, args)

function compilerule(md::ModelData, r::Rule)
    invardata = Dict{Symbol, Vector}()
    outvardata = Dict{Symbol, Vector}()
    zippedouts = Dict{Symbol, Vector}()
    for ds in schemas(md) if isnothing(ds) || stage(r) in stages(ds)
        sym = schemasymbol(ds)
        invardata[sym] = [
                vardata(md, in; ds=Try(ds), usefuturestate=(isa(in, By) ? :matchonly : false))
            for in in inputs(r)
        ]
        outvardata[sym] = [vardata(md, out; ds=iscontinuous(out) ? ds : nothing, usefuturestate=true) for out in outputs(r)]
        if length(outputs(r)) > 1 && broadcast_output(r)
            zippedouts[sym] = collect(zip(outvardata[sym]...))
        end
    end end

    return CompiledRule(r, invardata, outvardata, f_rule(r), zippedouts)
end

# Check if an object contains any nan values, recursively.
anynan(x::Union{AbstractArray,Tuple,Base.RefValue}) = any(anynan.(x))
anynan(::Union{Enum,Match}) = false
anynan(x::Number) = isnan(x)

# Rules only operate within schemas. So a rule will either operate values->values, A->A, etc.
# When rules take off-host variables to on-host variables, if the host is atomistic, all on-host duals are cleared.
function apply!(cr::CompiledRule, ds::Schema=nothing) #TODO Allow multiple outvars
    _outvardata = outvardata(cr, ds)
    if broadcast_output(description(cr))
        if length(_outvardata) == 1
            #TODO Why is Ref not working? e.g. @statedep loc.L_N(Ref(lϕ),Ref(s)) = sum(lϕ .*(s.==2)) / n_hh
            #=
            indata = invardata(cr, ds)
            anynan(indata) && error()
            only(_outvardata) .= cr.f.(indata...)
            if anynan(_outvardata)
                @show cr
                error()
            end
            =#
            only(_outvardata) .= cr.f.(invardata(cr, ds)...)
        else
            zippedouts(cr, ds) .= cr.f.(invardata(cr, ds)...)
            for (i,out) in enumerate(_outvardata)
                out .= getindex.(zippedouts, i)
            end
        end
    else
        #TODO IMPORTANT! If copying from non-idiosyncratic to idiosyncratic atomistic variable, strip partials!
        args = _getval.(invardata(cr, ds))
        for (outdata, val) in zip(_outvardata, cr.f(args...))
            _setval!(outdata, val)
        end
    end
    return outvardata
end

