#########
# Stage #
#########

@enum Stage begin
    st_param = -1
    st_state = 0
    st_statedep = 1
    st_equil = 2
    st_equildep = 3
    st_choice = 4
    st_choicedep = 5
    st_shock = 6
    st_shockdep = 7
    st_nextstate = 8
end
#TODO ST_PARAM, ST_STATE, etc.

const STAGETODEP = Dict(
    st_param => false,
    st_state => false,
    st_statedep => true,
    st_equil => false,
    st_equildep => true,
    st_choice => false,
    st_choicedep => true,
    st_shock => false,
    st_shockdep => true,
    st_nextstate => false
)

const ALLSTAGES = collect(instances(Stage))

##########
# Bounds #
##########

struct Bound{T, islower} # T <: Union{Float64, <:AbstractVariable}, islower::Bool
    boundvar::T
    inclusive::Bool
end

struct Bounds
    # Only exclusive bounds are currently supported.
    # Inclusive bounds will be handled via Lagrange.
    lower::Bound
    upper::Bound
end

Base.Broadcast.broadcastable(b::Bounds) = Ref(b)

#GuessableVariable = Union{ChoiceVariable, EquilibriumVariable}

######################
# Host and Variables #
######################

abstract type Host end
abstract type Inputtable end
abstract type AbstractVariable{stage, T} <: Inputtable end

Base.Broadcast.broadcastable(h::Host) = Ref(h)
Base.Broadcast.broadcastable(var::AbstractVariable) = Ref(var)

shortname(var::AbstractVariable) = symbol(var)
Base.show(io::IO, var::T) where {T<:AbstractVariable} = Base.print(io, "$T($(symbol(host(var))), :$(shortname(var)))")
Base.show(io::IO, h::T) where {T<:Host} = print(io, "$T(:$(symbol(h)))")

struct Rule{stage}
    inputs::Vector{Inputtable} #TODO Put `broadcasted` on AbstractVariable?
    outputs::Vector{AbstractVariable}
    f::Function
    broadcast_inputs::BitVector
    broadcast_output::Bool
end

Base.show(io::IO, r::Rule) = print(io, "Rule($(shortname.(r.inputs)) => $(shortname.(r.outputs)))")

struct Parameter{T} <: AbstractVariable{st_param,T}
    host::Host
    symbol::Symbol
    value::T
end

Base.show(io::IO, var::Parameter) = Base.print(io, "Parameter(model, :$(var.symbol), $(var.value))")

struct VectorVariable{stage, T}
    host::Host
    components::Vector{AbstractVariable{stage, T}}
end

mutable struct DependentVariable{stage,T} <: AbstractVariable{stage,T}
    host::Host
    symbol::Symbol
    rule::Union{Rule,Nothing}
    ischoicevar::Bool
    choiceprobs::Union{VectorVariable, Nothing} # The choice probabilities that determine the variable, if discrete
    DependentVariable{stage,T}(symbol,rule) where {stage,T} = (v = new{stage,T}(); v.symbol=symbol; v.rule=rule;v)
    DependentVariable{stage,T}(host,symbol,rule,ischoicevar,choiceprobs) where {stage,T} = new{stage,T}(host,symbol,rule,ischoicevar,choiceprobs)
end

mutable struct EquilibriumVariable{T} <: AbstractVariable{st_equil,T}
    host::Host
    symbol::Symbol
    bounds::Bounds
    init::T #TODO Change to rule-ish thing
end

mutable struct ChoiceVariable{T} <: AbstractVariable{st_choice,T}
    host::Host
    symbol::Symbol
    bounds::Bounds #TODO Change to Space
    init::Union{T,Vector{T}}
    ischoiceprob::Bool # The alternative being a continuous choice variable
end

mutable struct StateVariable{T} <: AbstractVariable{st_state,T}
    host::Host
    symbol::Symbol
    rule::Union{Rule,Nothing}
    init::Union{T,Vector{T}}
    ischoicevar::Bool
    choiceprobs::Union{VectorVariable, Nothing} # The choice probabilities that determine the variable, if discrete
    #ccp_placeholder::Union{ChoiceVariable, Nothing}
end

mutable struct ShockVariable{T} <: AbstractVariable{st_shock,T}
    host::Host
    symbol::Symbol
    dist::Sampleable
end

struct EnumSampler{EnumT} <: Sampleable{Univariate, Discrete}
    a::Vector{EnumT}
    p::AbstractWeights
end

struct Match{partner}
    i::Int
end

#TODO Handle all of this multiple inheritance with interfaces?
#TODO Agent{N} <: Host{N} ?

struct Agent <: Host
    model::Union{Host,Nothing}
    symbol::Symbol
    dynamic::Bool #discount_factor, V computation stuff, payoff_stage, etc?
    atomistic::Bool
    n::Int
    vars::Dict{Symbol, AbstractVariable}
    eqconds::Vector{AbstractVariable}
    payoffvar::AbstractVariable
    matchvars::Vector{AbstractVariable}
    rules::Vector{Rule} #TODO Centralize rules?
    choicefactors::Vector{Inputtable}
    #broadcast_factors::BitVector
end

struct Market <: Host
    model::Union{Host,Nothing}
    symbol::Symbol
    n::Int
    vars::Dict{Symbol, AbstractVariable}
    eqconds::Vector{AbstractVariable}
    members::Dict{Symbol, Agent}
    rules::Vector{Rule}
end

struct Model <: Host
    symbol::Symbol
    agents::Dict{Symbol, Agent}
    markets::Dict{Symbol, Market}
    bigT::Union{Int, Nothing}
    params::Dict{Symbol, Parameter}
    vars::Dict{Symbol, AbstractVariable}
    eqconds::Vector{AbstractVariable}
    rules::Vector{Rule}
end

Base.show(io::IO, ::Model) = Base.print(io, "Model()")

"
Represents a variable on an agent whose values are taken as the value of
the agent's match.
E.g. By(altitude, residence) is the altitude of an agent's residence.
"
struct By <: Inputtable
    host::Agent
    partnervar::AbstractVariable
    matchvar::AbstractVariable
end
shortname(by::By) = "By($(symbol(partnervar(by))),$(symbol(matchvar(by))))"

"
Represents a variable on a market whose values are taken as the values
of the market's members (on some particular matchvar).
E.g. ByMembers(wealth, residence) is the vector of wealth values of agents
whose residence is that market.
"
struct ByMembers <: Inputtable
    host::Market
    partnervar::AbstractVariable
    matchvar::AbstractVariable
end
shortname(bym::ByMembers) = "ByMembers($(symbol(partnervar(bym))),$(symbol(matchvar(bym))))"

##################
# Layout Utility #
##################

"Stores information about how a set of variables map onto a vector of values."
struct ArgLayout
    #TODO Either decide to do vector-value variables or remove unnecessary logic.
    nargs::Int
    nargspervar::Vector{Int}
    argoffsets::Vector{Int}
    argvars_long::Vector{AbstractVariable}
    argidict::Dict{Symbol,Int} #TODO Split off these last 5 as a separate struct at least
end

#########
# Duals #
#########

"Templates of Dual and Partials objects. Either zero or one entries are set to one, and all other
entries are set to zero. If SecondOrder, second-order Partials (all zero) are included."
struct DualTemplates{SecondOrder} # SecondOrder::Bool
    templatedual::Dual # A Dual object with value set to NaN and all derivatives set to zero
    ownpartials::Vector{Partials} # The derivative of each var with respect to itself
    zeropartials::Partials # A Partials object with all derivatives set to zero
end

"Stores information about how the set (possibly vector-valued) differentiands `argvars`
are mapped onto the (ordered) vector of partials."
struct DualLayout{SecondOrder} # SecondOrder::Bool
    arglayout::ArgLayout
    templates_ord1::DualTemplates{false}
    templates_ord2::Union{Nothing,DualTemplates{true}}
end

abstract type AbstractDualSchema{symbol, SecondOrder} end

struct DualSchema{symbol, SecondOrder} <: AbstractDualSchema{symbol, SecondOrder} # SecondOrder::Bool
    "The stages in which variables can have this dual form. Used for selecting where variables are sent by Try(ds). Sorted."
    stages::Vector{Stage} # TODO Replace this with a list of variables
    "The variables that derivatives are being taken with respect to."
    argvars::Vector{AbstractVariable}
    duallayout::DualLayout
end

struct Try{symbol, SecondOrder} <: AbstractDualSchema{symbol, SecondOrder}
    ds::DualSchema{symbol, SecondOrder}
end

Base.show(io::IO, ds::DualSchema{S}) where S = print(io, "DualSchema(:$S)")

Schema = Union{AbstractDualSchema, Nothing}
FloatVal = Union{Float64,AbstractVector{Float64}}
DualVal{S} = Union{Dual{S},AbstractVector{<:Dual{S}}}
FloatDualVal = Union{FloatVal,DualVal}
DiscreteValScalar = Union{Enum, Match}
DiscreteValVector = Vector{<:DiscreteValScalar}
DiscreteVal = Union{DiscreteValScalar, DiscreteValVector}
PartialsSeed = Dict{Symbol,Union{Float64,Dual}}
PartialsSeeds = Dict{Symbol,<:PartialsSeed}

Base.Broadcast.broadcastable(ds::AbstractDualSchema) = Ref(ds)

#######################
# Neural Nets, Brains #
#######################

abstract type Layer end

struct Dense <: Layer
    w::Param
    b::Param
    f
    w_bare::Matrix{Float64}
    b_bare::Vector{Float64}
    res::Dict{Symbol,Matrix}
end

struct Chain
    layers::NTuple{<:Any,Layer}
    Chain(args...) = new(args)
end

abstract type NeuralNet end

struct VNet <: NeuralNet
    invars::Vector{Inputtable}
    chain::Chain
end

struct PNet <: NeuralNet
    invars::Vector{Inputtable}
    outvars::Vector{AbstractVariable}
    chain::Chain
end

struct Brain
    agent::Agent
    β::Float64
    V_nn::VNet
    P_nn::PNet
end

##############
# Model Data #
##############

struct CompiledRule
    description::Rule
    invardata::Dict{Symbol,Vector}
    outvardata::Dict{Symbol,Vector}
    f::Function
    zippedouts::Dict{Symbol,Vector}
end

Base.show(io::IO, cr::CompiledRule) = print(io, "CompiledRule($(symbol.(cr.description.inputs)) => $(symbol.(cr.description.outputs)))")

abstract type HostData{Host} end

Base.Broadcast.broadcastable(hd::HostData) = Ref(hd)

struct VarDataCache
    vardata::Dict{Symbol, Union{Vector,Ref}}
    nextstatedata::Dict{Symbol, Union{Vector,Ref}}
    vardatadual::Dict{Symbol,Dict{Symbol, Union{Vector,Ref}}}
    nextstatedatadual::Dict{Symbol,Dict{Symbol, Union{Vector,Ref}}}
    statedatacheckpoint::Dict{Symbol, Union{Vector,Ref}}
    # TODO combine vardata and vardatadual (and nextstatedata[dual]).
    # Use :value as the symbol for the float dualschema and disallow :value as a DualSchema name.
    # Use this convention everywhere, instead of float =^= nothing?
end

struct AgentData <: HostData{Agent}
    description::Agent
    modeldata::HostData{Model}
    vardatacache::VarDataCache
    brain::Union{Brain,Nothing}
end

struct MarketData <: HostData{Market}
    description::Market
    modeldata::HostData{Model}
    vardatacache::VarDataCache
    membersdata::Dict{Symbol,Vector{Vector{Match}}} # Members in the current stage
end

struct ModelData <: HostData{Model}
    description::Model
    agentdata::Dict{Symbol, AgentData}
    marketdata::Dict{Symbol, MarketData}
    vardatacache::VarDataCache
    sortedrules::Vector{Vector{CompiledRule}} # Separated by stages
    dualschemas::Dict{Symbol, DualSchema}
end

Base.show(io::IO, hd::T) where {T<:HostData} = print(io, "$T(:$(symbol(hd)))")

VarData = Union{AbstractArray, Ref}
