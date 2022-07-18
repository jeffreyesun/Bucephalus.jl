######################
# Accessor Functions #
######################

# Rule
f_rule(r::Rule) = r.f
inputs(r::Rule) = r.inputs
outputs(r::Rule) = r.outputs
stage(::Rule{_stage}) where {_stage} = _stage
broadcast_inputs(r::Rule) = r.broadcast_inputs
broadcast_output(r::Rule) = r.broadcast_output

# Variable
host(var::AbstractVariable) = var.host
_host!(var::AbstractVariable, h::Host) = var.host = h
symbol(var::AbstractVariable) = var.symbol
dtype(::AbstractVariable{stage,T}) where {stage,T} = T
stage(::AbstractVariable{_stage,T}) where {_stage,T} = _stage
rule(var::Union{StateVariable,DependentVariable}) = var.rule
_rule!(var::Union{StateVariable,DependentVariable}, rule) = var.rule = rule
hasrule(var::Union{StateVariable,DependentVariable}) = !isnothing(var.rule)
init(var::Union{StateVariable,EquilibriumVariable,ChoiceVariable}) = var.init
dist(var::ShockVariable) = var.dist
value(var::Parameter) = var.value
bounds(var::Union{ChoiceVariable,EquilibriumVariable}) = var.bounds
iscontinuous(var::AbstractVariable{st, T}) where {st,T} = T == Float64
ischoiceprob(var::AbstractVariable) = false
ischoiceprob(var::ChoiceVariable) = var.ischoiceprob
ischoicevar(var::AbstractVariable) = false
ischoicevar(var::ChoiceVariable) = true
ischoicevar(var::Union{DependentVariable,StateVariable}) = var.ischoicevar
choiceprobs(var::Union{ChoiceVariable,StateVariable}) = var.choiceprobs
components(vecvar::VectorVariable) = vecvar.components

# EnumSampler
EnumSampler(EnumT, p) = EnumSampler(collect(instances(EnumT)), pweights(p))

# Host
model(h::Host) = h.model
symbol(h::Host) = h.symbol
n(h::Host) = h.n
vardict(h::Host) = h.vars
localvars(h::Host) = collect(values(h.vars))
varsymbols(h::Host) = collect(keys(h.vars))
getvar(h::Host, s::Symbol) = h.vars[s]
eqconds(h::Host) = h.eqconds
payoffvar(h::Host) = h.payoffvar
localrules(h::Host) = h.rules

# Model
model(m::Model) = m
n(::Model) = 1
isatomistic(::Model) = false
agentdict(m::Model) = m.agents
agents(m::Model) = collect(values(m.agents))
agentsymbols(m::Model) = collect(keys(m.agents))
marketdict(m::Model) = m.markets
markets(m::Model) = collect(values(m.markets))
marketsymbols(m::Model) = collecT(keys(m.markets))
getagent(m::Model, s::Symbol) = m.agents[s]
gethost(m::Model, s::Symbol) = get(m.agents, s, m.markets[s])
paramdict(m::Model) = m.params
allparams(m::Model) = collect(values(m.params))
paramsymbols(m::Model) = collect(keys(m.params))

# Agent
payoffvar(a::Agent) = a.payoffvar
matchvars(a::Agent) = a.matchvars
isdynamic(a::Agent) = a.dynamic
isatomistic(a::Agent) = a.atomistic
choicefactors(a::Agent) = a.choicefactors

# Market
members(mr::Market) = mr.members

# By
partnervar(by::By) = by.partnervar
matchvar(by::By) = by.matchvar
partnervar(bym::ByMembers) = bym.partnervar
matchvar(bym::ByMembers) = bym.matchvar

#####################
# Utility Functions #
#####################

# Inherited/Computed Properties #
#-------------------------------#

# Parent Model
model(var::AbstractVariable) = model(host(var))
agent(var::AbstractVariable) = isa(host(var), Agent) ? host(var) : error("`var` does not belong to an agent.")

# Heterogeneity
isheterogeneous(h::Host) = n(h) > 1
isheterogeneous(var::AbstractVariable) = isheterogeneous(host(var))
n(var::AbstractVariable) = n(host(var))

# Equilibrium
haslocalequilibrium(h::Host) = !isempty(localvars(h, st_equil))
hasequilibrium(m::Model) = any(haslocalequilibrium.(allhosts(m)))

# Bounds
hasbounds(::Union{ChoiceVariable,EquilibriumVariable}) = true
hasbounds(::AbstractVariable) = false

# Stage
# `StateVariable`s are unique in that they are read from one stage but written to another.
outstage(var::AbstractVariable) = stage(var)
outstage(::StateVariable) = st_nextstate
stage(by::By) = stage(partnervar(by))
stage(bym::ByMembers) = stage(partnervar(bym))

# Match
partnersymbol(_Match::DataType) = (@assert _Match <: Match; _Match.parameters[1])
partnersymbol(::AbstractVariable{S, Match{partnerS}}) where {S, partnerS} = partnerS
partnerhost(m::Model, ::AbstractVariable{S, Match{partnerS}}) where {S, partnerS} = gethost(m, partnerS)
partnerhost(var::AbstractVariable) = partnerhost(model(var), var)
partnerhost(m::Model, ::Match{partnerS}) where {partnerS} = gethost(m, partner)
partnerhost(m::Model, _Match::DataType) = gethost(m, partnersymbol(_Match))
zero(m::Match{S}) where {S} = Match{S}(0)
zero(::Type{Match{S}}) where {S} = Match{S}(0)
zeros(::Type{Match{S}}, n::Int) where {S} = fill(Match{S}(0), n)
Int(m::Match) = m.i

# By
host(by::By) = by.host
upstreamvars(by::By) = upstreamvars(partnervar(by))

# ByMembers
host(bym::ByMembers) = bym.host
upstreamvars(bym::ByMembers) = upstreamvars(partnervar(bym))

# Other
unbroadcastedtype(var) = isheterogeneous(var) ? Vector{dtype(var)} : dtype(var)
isatomistic(var::AbstractVariable) = isatomistic(host(var))


# Aggregation #
#-------------#

allhosts(m::Model) = reduce(vcat, vcat(agents(m), markets(m)), init=[m])
aggregatehosts(f::Function, m::Model; kwargs...) = mapreduce(f, vcat, allhosts(m); kwargs...)
allsymbols(m::Model) = [symbol(m); paramsymbols(m); agentsymbols(m); mapreduce(varsymbols, vcat, allhosts(m))]
staticagents(m::Model) = (a for a in agents(m) if !isdynamic(a))
filterstage(vars::Vector{<:AbstractVariable}, st::Stage) = filter(var->in(st,[stage(var),outstage(var)]), vars)

# Queries #
#---------#

#TODO Cache these instead of searching each time. Memoize.jl?

#   Local (Search only over variables directly assigned to host)
localvars(h::Host, stage::Stage) = filterstage(localvars(h), stage)
localdepvars(h::Host) = filter(var->isa(var, DependentVariable), localvars(h))
localnondepvars(h::Host) = filter(var->!isa(var, DependentVariable), localvars(h))
localcontinuousvars(h::Host, stage::Stage) = filter(iscontinuous, localvars(h, stage))
localdiscretevars(h::Host, stage::Stage) = filter(!iscontinuous, localvars(h, stage))
localdiscretechoicevars(h::Host) = filter(v->!iscontinuous(v) & ischoicevar(v), localvars(h))
localmatchvars(h::Host) = filter(x->dtype(x)<:Match, localvars(hh))

#   Global (Search over all variables in parent model)
allvars(m::Model) = aggregatehosts(localvars, m, init=allparams(m))
allvars(m::Model, stage::Stage) = aggregatehosts(h->localvars(h,stage), m)
allcontinuousvars(m::Model) = filter(iscontinuous, allvars(m))
allcontinuousvars(m::Model, stage::Stage) = filter(iscontinuous, allvars(m, stage))
alldiscretevars(m::Model) = filter(!iscontinuous, allvars(m))
alldiscretevars(m::Model, stage::Stage) = filter(!iscontinuous, allvars(m, stage))
alldepvars(m::Model) = aggregatehosts(localdepvars, m)
allnondepvars(m::Model) = aggregatehosts(localnondepvars, m)
alleqconds(m::Model) = aggregatehosts(eqconds, m)

const FORBIDDEN_VARSYMBOL_CHARS = ['>', '[', ']', '%']

"Verify that symbol is unique across the model."
function checksymbol(h::Host, symbol::Symbol)
    for char in FORBIDDEN_VARSYMBOL_CHARS
        char in string(symbol) && error("Character '"*char*"' not allowed in symbols.")
    end
    symbol in allsymbols(model(h)) && error("Symbol $symbol already in use.")
end

################
# Constructors #
################

function DependentVariable{stage,T}(host,symbol,rule) where {stage,T}
    return DependentVariable{stage,T}(host, symbol, rule, false, nothing)
end

function StateVariable{T}(host, symbol, rule, init) where {T}
    return StateVariable(host, symbol, rule, init, false, nothing)
end

function By(partnervar::AbstractVariable, matchvar::AbstractVariable{S,<:Match}) where {S}
    return By(host(matchvar), partnervar, matchvar)
end

function ByMembers(partnervar::AbstractVariable, matchvar::AbstractVariable{S,<:Match}) where {S}
    return ByMembers(partnerhost(matchvar), partnervar, matchvar)
end

function Agent(m::Model, symbol::Symbol;
    dynamic::Bool,
    atomistic::Bool,
    n::Int,
    u_symbol::Symbol
)
    payoffvar = DependentVariable{st_choicedep,Float64}(u_symbol, nothing)
    vars = Dict{Symbol, AbstractVariable}(u_symbol => payoffvar)
    eqconds = AbstractVariable[]
    matchvars = AbstractVariable[]
    rules = Rule[]
    choicefactors = Inputtable[]
    a = Agent(m, symbol, dynamic, atomistic, n, vars, eqconds, payoffvar, matchvars, rules, choicefactors)
    _host!(payoffvar, a)
    return a
end

function Market(m::Model, symbol::Symbol; n::Int)
    vars = Dict{Symbol, AbstractVariable}()
    eqconds = AbstractVariable[]
    members = Dict{Symbol,Agent}()
    rules = Rule[]
    return Market(m, symbol, n, vars, eqconds, members, rules)
end

function Model(;
    agents=Dict{Symbol, Agent}(),
    markets=Dict{Symbol, Market}(),
    bigT=nothing,
    params=Dict{Symbol, Parameter}(),
    vars=Dict{Symbol, AbstractVariable}(),
    eqconds=AbstractVariable[],
    rules=Rule[]
)
    return Model(:model, agents, markets, bigT, params, vars, eqconds, rules)
end

###################
# Causal Ordering #
###################

upstreamvars(var::DependentVariable) = AbstractVariable[var] ∪ upstreamvars(rule(var))
upstreamvars(var::AbstractVariable) = AbstractVariable[var]
upstreamvars(::Nothing) = AbstractVariable[]
upstreamvars(rule::Rule) = mapreduce(upstreamvars, union, inputs(rule), init=AbstractVariable[])

function sortvars(m::Model)
    vars = allvars(m)
    # Sort by stage
    sort!(vars; by=outstage)
    # Topological sort within stage
    visited = Dict(vars .=> false)
    sortedvars = AbstractVariable[]
    function include_upstream(var)
        if !visited[var]
            visited[var] = true
            if var isa Union{DependentVariable, StateVariable}
                for prior_var in inputs(rule(var))
                    isa(prior_var, Union{By,ByMembers}) && (prior_var = partnervar(prior_var))
                    if !isa(prior_var, StateVariable)
                        include_upstream(prior_var)
                    end
                end
            end
            push!(sortedvars, var)
        end
    end
    include_upstream.(vars)
    @assert issorted(sortedvars; by=outstage)
    return sortedvars
end

function sortrules(m::Model)
    sortedvars = sortvars(m)
    sortedrules = Rule[]
    for var in sortedvars if var isa Union{DependentVariable,StateVariable}
        rule(var) in sortedrules || push!(sortedrules, rule(var))
    end end
    return sortedrules
end

rand(es::EnumSampler) = sample(es.a, es.p)
rand(es::EnumSampler, n::Int) = sample(es.a, es.p, n)
rand!(es::EnumSampler, x::AbstractArray) = sample!(es.a, es.p, x)