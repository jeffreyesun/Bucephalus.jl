
#############################################
# Add Agents, Variables, and Rules to Model #
#############################################

##########
# Agents #
##########

function addAgent!(m::Model, symbol::Symbol;
    dynamic::Bool,
    atomistic::Bool=true,
    n::Int=1,
    u_symbol::Symbol=Symbol("u", symbol),
)
    checksymbol.(m, [symbol, u_symbol])
    agent = Agent(m, symbol; dynamic, atomistic, n, u_symbol)
    agentdict(m)[symbol] = agent
    return agent
end

###########
# Markets #
###########

function addMarket!(m::Model, symbol::Symbol; n::Int)
    checksymbol(m, symbol)
    mr = Market(m, symbol; n)
    marketdict(m)[symbol] = mr
    return mr
end

#############
# Variables #
#############

# Low-Level Wiring #
#------------------#

function _varsymbol(h::Host, symbol::Symbol)
    if symbol == Symbol("")
        symbol = Symbol(symbol(h), ">var", length(vars(h))+1)
    else
        checksymbol(h, symbol)
    end
    return symbol
end

function _addvariable!(h::Host, var::AbstractVariable)
    @assert h == host(var)
    vardict(h)[symbol(var)] = var
    return var
end

"Process Initial Value Supplied for Variable."
function processinit(h::Host, dtype::DataType, init)
    init isa Sampleable && (init = rand(init, n(h)))
    # TODO Keep init as a sampleable. Don't fill in until initialize.
    init isa Vector && n(h) == 1 && (init = only(init))
    if init isa dtype
        if n(h) == 1
            return init
        else
            return fill(init, n(h))
        end
    elseif init isa AbstractVector{dtype}
        length(init) == n(h) || error("`length(init)` must equal `n(host)`.")
        return init
    elseif init isa AbstractVector
        eltype(init) <: Int || error("`init` must match `dtype`.")
        return dtype.(init)
    end
    error("Invalid format for `init`")
end

# Parameters #
#------------#

function addParameter!(m::Model, symbol::Symbol, value)
    checksymbol(m, symbol)
    param = Parameter{typeof(value)}(m, symbol, value)
    paramdict(m)[symbol] = param
    return param
end

# Stage Game Variables #
#----------------------#

function addDependentVariable!(h::Host, dtype::DataType, symbol::Symbol=Symbol(""); stage::Stage)
    symbol = _varsymbol(h, symbol)
    var = DependentVariable{stage,dtype}(h, symbol, nothing)
    stage in [st_statedep, st_equildep, st_choicedep, st_shockdep] || error("Invalid stage for dependent variable.")
    return _addvariable!(h, var)
end

function addStateVariable!(h::Host, dtype::DataType, symbol::Symbol=Symbol(""); init)
    symbol = _varsymbol(h, symbol)
    init = processinit(h, dtype, init)
    var = StateVariable{dtype}(h, symbol, nothing, init)
    return _addvariable!(h, var)
end

function addEquilibriumVariable!(h::Host, dtype::DataType, symbol::Symbol=Symbol(""); init, bounds::Bounds=Bounds())
    symbol = _varsymbol(h, symbol)
    init = processinit(h, dtype, init)
    var = EquilibriumVariable{dtype}(h, symbol, bounds, init)
    return _addvariable!(h, var)
end

"Add a (continuous) choice variable to the model."
function addChoiceVariable!(h::Host, dtype::DataType, symbol::Symbol=Symbol(""); init, bounds::Bounds=Bounds())
    @assert dtype == Float64
    symbol = _varsymbol(h, symbol)
    init = processinit(h, dtype, init)
    var = ChoiceVariable{dtype}(h, symbol, bounds, init, false)
    return _addvariable!(h, var)
end

addPayoffVar!(args...; kwargs...) = error("Cannot add new payoff variables.")

function addEquilbriumConditions!(h::Host, conds::Vector{<:AbstractVariable})
    all(stage(cond) in [st_equil, st_equildep, st_choice, st_choicedep] for cond in conds) || error(
        "Equilibrium conditions must be in equilibrium or choice stages."
        )
    all(host(cond)==h for cond in conds) || error("Not all variables supplied belong to host \"$(symbol(h))\".")
        union!(eqconds(h), conds)
    return
end

function addShockVariable!(h::Host, dtype::DataType, symbol::Symbol=Symbol(""); dist::Sampleable, checksymbol::Bool=true)
    checksymbol && (symbol = _varsymbol(h, symbol))
    var = ShockVariable{dtype}(h, symbol, dist)
    return _addvariable!(h, var)
end

# Matching #
#----------#

function addVariable!(h::Host, dtype::DataType, stage::Stage, symbol::Symbol=Symbol(""); args...)
    if stage == st_state
        return addStateVariable!(h, dtype, symbol; args...)
    else
        error("Not yet implemented")
    end
end

"Add a match-type variable to the model, instantiating both the variable and its corresponding variable on the partner."
function addMatchVariable!(h::Host, _Match::DataType, stage::Stage, _symbol::Symbol=Symbol(""); init)
    _partnerhost = partnerhost(model(h), _Match)
    # Instantiate the variable itself and record it as one of the host's matchvars
    var = addVariable!(h, _Match, stage, _symbol; init)
    push!(matchvars(h), var)

    # Record the match on the partner as well
    members(_partnerhost)[symbol(var)] = h
    return var
end

# Rules #
#-------#

function validaterulebroadcasting(h::Host; inputs, outputs, f::Function, broadcast_output::Bool, _stage::Stage)
    
    h isa Model && broadcast_output && error("Cannot broadcast if host is a model.")
    !broadcast_output && any(.!isa.(inputs, Inputtable)) && error(
        "If not broadcasting outputs, `inputs` must be a Vector{<:Inputtable}."
    )
    all(outstage.(outputs) .== _stage) || error("All outputs must have the same stage.")

    # Validate outputs
    broadcast_output && !isheterogeneous(h) && error(
        "Broadcasting outputs is only defined for heterogeneous agents."
    )
    for out in outputs
        hasrule(out) && error("Output $(symbol(out)) already has a rule.")
        isa(h, Agent) && host(out) != h && error("Output $(symbol(out)) does not belong to the given agent.")
    end

    # Validate inputs
    for input in inputs
        bcast = broadcast_output & isa(input, Inputtable)
        input = !broadcast_output | bcast ? input : input[]
        input isa Inputtable || error("Inputs must all be Inputtables or Ref wrappers thereof.")
        stage(input) <= _stage || error("Stage of input $(symbol(input)) is after output stage.")
        
        if bcast && isheterogeneous(host(input)) && host(input) != h
            error(
                "Cannot broadcast over input variable $(symbol(input)).\\
                It belongs to a heterogeneous agent that does not match the output."
            )
        elseif broadcast_output && !bcast && !isheterogeneous(host(input))
            error("Input $(symbol(input)) does not belong to a heterogeneous agent. Remove `Ref` wrapper.")
        end
        _stage == st_nextstate || length(upstreamvars(input) ∩ outputs) == 0 || error(
            "Given input $(symbol(input)) circularly depends on outputs."
        )
    end
end

function _addrule!(h::Host, rule::Rule, _outputs)
    @assert _outputs == outputs(rule)
    
    _rule!.(_outputs, Ref(rule))
    push!(localrules(h), rule)
    return rule
end

function addRule!(h::Host; inputs, outputs, f::Function, broadcast_output::Bool=isheterogeneous(h))
    # Put inputs, output, and broadcast_inputs into a standard form.
    inputs isa Vector || (inputs = [inputs])
    outputs isa Vector || (outputs = [outputs])
    stage = outstage(first(outputs))

    validaterulebroadcasting(h; inputs, outputs, f, broadcast_output, _stage=stage)
    
    broadcast_inputs = broadcast_output .& isa.(inputs, Inputtable)
    inputs = [!broadcast_output | bcast ? input : input[] for (input,bcast)=zip(inputs,broadcast_inputs)]
    rule = Rule{stage}(inputs, outputs, f, broadcast_inputs, broadcast_output)

    return _addrule!(h, rule, outputs)
end

function addPayoffRule!(
    h::Host;
    inputs, f::Function, broadcast_inputs=nothing, broadcast_output::Bool=isheterogeneous(h)
)
    addRule!(h; inputs, outputs=payoffvar(h), f, broadcast_output)
    return payoffvar(h)
end

# Factors #
#---------#

function addfactor!(a::Agent, factor::Inputtable)
    stage(factor) < st_choice || error("Factor must precede choice stage.")
    in(factor, choicefactors(a)) && @warn("Factor already added to agent.")
    push!(choicefactors(a), factor)
end


#######################
# Clean Syntax Macros #
#######################


macro parameter(ex)
    hostname = ex.args[1].args[1]
    varquotsymb = ex.args[1].args[2]
    varname = eval(varquotsymb)
    val = ex.args[2]
    return esc(:($varname = addParameter!($hostname, $varquotsymb, $val)))
end

macro statevar(ex)
    if ex.args[1].head == :(::)
        dtype = ex.args[1].args[2]
        ex.args[1] = ex.args[1].args[1]
    else
        dtype = Float64
    end
    hostname = ex.args[1].args[1]
    varquotsymb = ex.args[1].args[2]
    varname = eval(varquotsymb)
    init = ex.args[2]
    return esc(:($varname = addStateVariable!($hostname, $dtype, $varquotsymb; init=$init)))
end

macro matchvar(ex)
    if ex.args[1].head == :(::)
        dtype = ex.args[1].args[2]
        ex.args[1] = ex.args[1].args[1]
    else
        error("Must specify Match type")
    end
    hostname = ex.args[1].args[1]
    varquotsymb = ex.args[1].args[2]
    varname = eval(varquotsymb)
    init = ex.args[2]
    return esc(:($varname = addMatchVariable!($hostname, $dtype, st_state, $varquotsymb; init=$init)))
end

function input2varname(input)
    input isa Symbol && return input
    input.args[1] in [:Ref,:By,:ByMembers] && return input.args[2]
    error("Inputs must be Inputtables")
end

function macro_addrule(hostname, ex, varname)
    #TODO Escape symbols individually for hygeine
    fname = Symbol("f_", varname)
    ex_f = copy(ex)
    ex_f.args[1].args[2:end] .= input2varname.(ex.args[1].args[2:end])
    ex_f.args[1].args[1] = fname
    ex_f = esc(ex_f)
    
    inputs = :([$(ex.args[1].args[2:end]...)])
    ex_addrule = esc(:(addRule!($hostname, inputs=$inputs, outputs=$varname, f=$fname)))

    return ex_f, ex_addrule, esc(varname)
end

function macro_depvar(ex, stage)
    hostname = ex.args[1].args[1].args[1]
    varname = eval(ex.args[1].args[1].args[2])
    ex.args[1].args[1] = varname
    
    ex_newvar = esc(:($varname = addDependentVariable!($hostname, Float64, $(Meta.quot(varname)); stage=$stage)))

    exs_addrule = macro_addrule(hostname, ex, varname)
    return Expr(:block, ex_newvar, exs_addrule...)
end

macro statedep(ex)
    return macro_depvar(ex, :st_statedep)
end

macro equildep(ex)
    return macro_depvar(ex, :st_equildep)
end

macro choicedep(ex)
    return macro_depvar(ex, :st_choicedep)
end

macro shockdep(ex)
    return macro_depvar(ex, :st_shockdep)
end

macro transition(ex)
    varname = ex.args[1].args[1]
    hostname = :(Bucephalus.host($varname))
    ex.args[1].args[1] = varname

    exs_addrule = macro_addrule(hostname, ex, varname)
    return Expr(:block, exs_addrule...)
end


##############
# Validation #
##############


#TODO Move to bottom
function validate(m::Model)
    fillintransitions!(m)
    symbs = allsymbols(m)
    @assert length(unique(symbs)) == length(symbs)
    vars = allvars(m)
    @assert length(unique(vars)) == length(vars)
    validatehost.(allhosts(m))
    validateagent.(agents(m))
    validatemarket.(markets(m))
    hasequilibrium(m) || @assert isempty(allvars(m, st_equildep))
    return nothing
end

function fillintransitions!(m::Model)
    vars = Any[]
    for var in allvars(m, st_state) if !hasrule(var)
        addRule!(host(var), inputs=var, outputs=var, f=identity)
        append!(vars, [symbol(var), ", "])
    end end
    if !isempty(vars)
        println("Assumed identity transitions for statevars: ", vars[1:end-1]..., '.')
    end
end

function validatehost(h::Host)
    # Validate Vars
    for var in eqconds(h)
        @assert host(var) == h
        @assert var in localvars(h)
    end
    if h isa Agent
        @assert host(payoffvar(h)) == h
        @assert payoffvar(h) in localvars(h)
    end
    for (k,var) in vardict(h)
        @assert symbol(var) == k
        if var isa Union{DependentVariable,StateVariable}
            @assert hasrule(var)
            @assert var in outputs(rule(var))
        end
        var isa Parameter && @assert host(var) == model(var)
    end
    # Validate Rules
    for r in localrules(h)
        _stage = stage(r)
        _stage != st_nextstate && @assert isempty(upstreamvars(r) ∩ outputs(r))
        for out in outputs(r)
            @assert host(out) == h
            @assert rule(out) == r
            @assert stage(out) <= _stage
        end
    end
    if haslocalequilibrium(h)
        @assert !isempty(eqconds(h))
    else
        @assert isempty(eqconds(h))
    end
    return nothing
end

function validateagent(a::Agent)
    # Validate factors
    for factor in choicefactors(a)
        @assert stage(factor) < st_choice
    end
    # Validate matches
    for var in matchvars(a)
        _partnerhost = partnerhost(model(a), var)
        @assert members(_partnerhost)[symbol(var)] == a
    end
end

function validatemarket(mr::Market)
    # Validate members
    for (varS, a) in members(mr)
        var = getvar(a, varS)
        @assert var in matchvars(a)
        @assert symbol(mr) == partnersymbol(var)
    end
end

# TODO Automatically move variables backwards to their earliest feasible stage.
