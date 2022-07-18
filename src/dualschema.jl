######################
# Accessor Functions #
######################

# DualSchema
@inline symbol(ds::DualSchema{S}) where S = S
@inline ord2(::DualSchema{S, Order2}) where {S, Order2} = Order2
@inline stages(ds::DualSchema) = ds.stages
@inline argvars(ds::DualSchema) = ds.argvars
@inline duallayout(ds::DualSchema) = ds.duallayout
@inline arglayout(ds::DualSchema) = ds.duallayout.arglayout
@inline nargs(ds::DualSchema) = ds.duallayout.arglayout.nargs
@inline nargspervar(ds::DualSchema) = ds.duallayout.arglayout.nargspervar
@inline argoffsets(ds::DualSchema) = ds.duallayout.arglayout.argoffsets
@inline argoffset(ds::DualSchema, argvari) = ds.duallayout.arglayout.argoffsets[argvari]
@inline argvars_long(ds::DualSchema) = ds.duallayout.arglayout.argvars_long
@inline argidict(ds::DualSchema) = ds.duallayout.arglayout.argidict
@inline templates_ord1(ds::DualSchema) = ds.duallayout.templates_ord1
@inline templates_ord2(ds::DualSchema{S, true}) where S = ds.duallayout.templates_ord2
@inline templates(ds::DualSchema{S, SecondOrder}) where {S, SecondOrder} = SecondOrder ? templates_ord2(ds) : templates_ord1(ds)
@inline templatedual_ord1(ds::DualSchema) = templates_ord1(ds).templatedual
@inline templatedual_ord2(ds::DualSchema{S, true}) where S = templates_ord2(ds).templatedual
@inline templatedual(ds::DualSchema) = templates(ds).templatedual
@inline ownpartials_ord1(ds::DualSchema) = templates_ord1(ds).ownpartials
@inline ownpartials_ord2(ds::DualSchema{S, true}) where S = templates_ord2(ds).ownpartials
@inline ownpartials(ds::DualSchema) = templates(ds).ownpartials
@inline zeropartials_ord1(ds::DualSchema) = templates_ord1(ds).zeropartials
@inline zeropartials_ord2(ds::DualSchema{S, true}) where S = templates_ord2(ds).zeropartials
@inline zeropartials(ds::DualSchema) = templates(ds).zeropartials
"Get a Partials object in DualSchema ds with the i-th entry set to 1 (all zero if i==0)."
@inline ownpartials_ord1(ds::DualSchema, i::Int) = i==0 ? zeropartials_ord1(ds) : ownpartials_ord1(ds)[i]
@inline ownpartials_ord2(ds::DualSchema{S, true}, i::Int) where S = i==0 ? zeropartials_ord2(ds) : ownpartials_ord2(ds)[i]
@inline ownpartials(ds::DualSchema, i::Int) = i==0 ? zeropartials(ds) : ownpartials(ds)[i]
# Nothing (representing the null schema) #TODO FloatSchema <:Schema
#stages(::Nothing) = Vector{Stage}[]
#TODO ValueSchema <: Schema instead of Nothing
# Try
@inline ds(tds::Try) = tds.ds

################
# Constructors #
################

# DualSchema #
#------------#

function DualSchema{dssymbol, SecondOrder}(
    stages::Vector{Stage}, argvars::Vector{<:AbstractVariable}
) where {dssymbol, SecondOrder}
    @assert dssymbol isa Symbol
    @assert SecondOrder isa Bool
    @assert issorted(stages)
    # args
    nargspervar = [isatomistic(var) ? 1 : n(var) for var in argvars]
    nargs = sum(nargspervar)
    argoffsets = pushfirst!(cumsum(nargspervar[1:end-1]), 0)
    argvars_long = mapreduce(fill, vcat, argvars, nargspervar)
    argidict = Dict{Symbol,Int}(symbol.(argvars) .=> argoffsets .+ 1)
    # templates
    ownpartials_ord1 = partials.(1:nargs, nargs)
    zeropartials_ord1 = partials(0, nargs)
    templatedual_ord1 = Dual{dssymbol}(NaN, zeropartials_ord1)
    templates_ord1 = DualTemplates{false}(templatedual_ord1, ownpartials_ord1, zeropartials_ord1)
    if SecondOrder
        ownpartials_ord2 = partials_ord2.(1:nargs, nargs, dssymbol)
        zeropartials_ord2 = partials_ord2(0, nargs, dssymbol)
        templatedual_ord2 = Dual{dssymbol}(templatedual_ord1, zeropartials_ord2)
        templates_ord2 = DualTemplates{true}(templatedual_ord2, ownpartials_ord2, zeropartials_ord2)
    else
        templates_ord2 = nothing
    end
    arglayout = ArgLayout(nargs, nargspervar, argoffsets, argvars_long, argidict)
    duallayout = DualLayout{SecondOrder}(
        arglayout, templates_ord1, templates_ord2
    )
    return DualSchema{dssymbol, SecondOrder}(stages, argvars, duallayout)
end

Try(::Nothing) = nothing

# Dual and Partials, From Values #
#--------------------------------#

partials(i::Int, n::Int; zero=0.0, one=1.0) = Partials(Tuple([i == j ? one : zero for j=1:n]))

function partials_ord2(i::Int, n::Int, T=Nothing)
    secondpartials = Partials(Tuple(zeros(n)))
    onedual = Dual{T}(1.0, secondpartials)
    zerodual = Dual{T}(0.0, secondpartials)
    return partials(i, n; zero=zerodual, one=onedual)
end

"Construct a Dual corresponding to a variable mapped to position `i` in DualSchema `ds``, using cached ownpartials."
dual_ord1(ds::DualSchema{S}, i::Int, val::Float64) where S = Dual{S}(val, ownpartials_ord1(ds, i))
dual(ds::DualSchema{S, false}, i::Int, val::Float64) where S = dual_ord1(ds, i, val)
dual(ds::DualSchema{S, true}, i::Int, val::Float64) where S = Dual{S}(dual_ord1(ds,i,val), ownpartials_ord2(ds,i))

# Dual and Partials, From Variables #
#-----------------------------------#

function arg_i(ds::DualSchema, varsymbol::Symbol, i_inner::Int=1)
    #TODO Check length. Use vars as dict keys.
    #1 <= i_inner <= n(var) || error("i_inner must be within `[1, n(var)]`.")
    1 < i_inner && isatomistic(var) && error("If `var` is atomistic, `i_inner` must equal 1 (default).")
    return get(argidict(ds), varsymbol, 0)
end
arg_i(ds::DualSchema, var::AbstractVariable, i_inner::Int=1) = arg_i(ds, symbol(var), i_inner)

dual(ds::DualSchema, var::AbstractVariable, val::Float64, i_inner::Int=1) = dual(ds, arg_i(ds, var, i_inner), val)

"Construct a Dual or vector of Duals with value `val` corresponding to (possibly vector-valued) variable `var`,
with all partials set to zero."
function dual(ds::DualSchema, var::AbstractVariable, val::Vector{Float64})
    dual.(ds,var,val, isatomistic(var) ? ones(Int,n(var)) : 1:n(var))
end

"Generate a lazy iterator of partials from a Dual in another DualSchema"
function partials_iter_ord1(ds_out::DualSchema, i::Int, dualval::Dual{S}, ds_in::DualSchema{S}) where {S}
    _argvars_long = argvars_long(ds_out)
    return (argi==i ? 1.0 : float(tryderivative(ds_in, dualval, _argvars_long[argi])) for argi in 1:nargs(ds_out))
end

"Generate a Partials object from a Dual in another DualSchema, with derivatives copied over"
function seededpartials_ord1(ds_out::DualSchema, i::Int, dualval::Dual{S}, ds_in::DualSchema{S}) where {S}
    #TODO Use StaticArrays.MVector to mutate these Tuples inplace.
    return Partials(Tuple(partials_iter_ord1(ds_out, i, dualval, ds_in)))
end

"Generate a first-order Dual from a Dual in another DualSchema, with derivatives copied over"
function seededdual_ord1(
    ds_out::DualSchema{S_OUT}, i::Int, dualval::Dual{S_IN}, ds_in::DualSchema{S_IN}
) where {S_OUT, S_IN}
    return Dual{S_OUT}(float(dualval), seededpartials_ord1(ds_out, i, dualval, ds_in))
end
function seededdual_ord1(ds_out::DualSchema, var::AbstractVariable, dualval::Dual{S_IN}, ds_in::DualSchema{S_IN}) where {S_IN}
    return seededdual_ord1(ds_out, arg_i(ds_out, var), dualval, ds_in)
end

"Generate a second-order lazy iterator of partials from a first-order Dual in another DualSchema, with derivatives copied over"
function partials_iter_ord2(
    ds_out::DualSchema{S_OUT, true}, i::Int, dualval::Dual{S_IN}, ds_in::DualSchema{S_IN, false}
) where {S_OUT, S_IN}
    zp1 = zeropartials_ord1(ds_out)
    return (Dual{S_OUT}(val, zp1) for val in partials_iter_ord1(ds_out, i, dualval, ds_in))
end

"Generate a second-order Partials object from a first-order Dual in another DualSchema, with derivatives copied over"
function seededpartials_ord2(
    ds_out::DualSchema{S_OUT, true}, i::Int, dualval::Dual{S_IN}, ds_in::DualSchema{S_IN, false}
) where {S_OUT, S_IN}
    return Partials(Tuple(partials_iter_ord2(ds_out, i, dualval, ds_in)))
end

"Generate a second-order Dual from a first-order Dual in another DualSchema, with derivatives copied over.
The derivative of the output w/r/t the i-th argument is set to 1 (none if i==0)"
function seededdual_ord2(
    ds_out::DualSchema{S_OUT, true}, i::Int, dualval::Dual{S_IN}, ds_in::DualSchema{S_IN, false}
) where {S_OUT, S_IN}
    return Dual{S_OUT}(seededdual_ord1(ds_out, i, dualval, ds_in), seededpartials_ord2(ds_out, i, dualval, ds_in))
end

"Generate a second-order Dual from a first-order Dual in another DualSchema, with derivatives copied over"
function seededdual_ord2(
    ds_out::DualSchema{S_OUT,true}, var::AbstractVariable, dualval::Dual{S_IN}, ds_in::DualSchema{S_IN,false}, 
) where {S_OUT,S_IN}
    return seededdual_ord2(ds_out, arg_i(ds_out, var), dualval, ds_in)
end

"Dispatch `seededdual`` according to the order of the target `ds_out`"
function seededdual(
    ds_out::DualSchema{S_OUT,true}, var::AbstractVariable, dualval::Dual{S_IN}, ds_in::DualSchema{S_IN,false}
) where {S_OUT,S_IN}
    return seededdual_ord2(ds_out, var, dualval, ds_in)
end
function seededdual(
    ds_out::DualSchema{S_OUT,false}, var::AbstractVariable, dualval::Dual{S_IN}, ds_in::DualSchema{S_IN}
) where {S_OUT,S_IN}
    return seededdual_ord1(ds_out, var, dualval, ds_in)
end

# Dual and Partials, From PartialsSeeds #
#---------------------------------------#
"Generates a PartialsSeed (aliasing Dict{Symbol,<:Union{Float64}}) from a Dual,
converting its stored partials into a dict"
function PartialsSeed!(seed::PartialsSeed, ds::DualSchema{S, false}, dualval::Dual{S}) where S
    for var in argvars(ds)
        seed[symbol(var)] = derivative(ds, dualval, var)
    end
    return seed
end

"Generate blank PartialsSeeds"
blankseeds(argvars::Vector{<:AbstractVariable}) = Dict(symbol(var)=>PartialsSeed() for var in argvars)
blankseeds(ds::DualSchema) = blankseeds(argvars(ds))

"Generate a lazy iterator of partials from a seed."
function partials_iter_ord1(ds::DualSchema, i::Int, seed::PartialsSeed)
    _argvars_long = argvars_long(ds)
    return (argi==i ? 1.0 : get(seed, _argvars_long[argi], 0.0) for argi in 1:nargs(ds))
end

seededpartials_ord1(ds::DualSchema, i::Int, seed::PartialsSeed) = Partials(Tuple(partials_iter_ord1(ds,i,seed)))

"Generate a first-order Dual from a seed"
function seededdual_ord1(ds::DualSchema{S}, i::Int, val::Float64, seed::PartialsSeed) where {S}
    return Dual{S}(val, seededpartials_ord1(ds, i, seed))
end

############
# Defaults #
############

"DualSchema to take derivatives of value function w/r/t dynamic choices,
for the purpose of updating policy function."
function dualschemaA(m::Model)
    #TODO Rename :dsV
    symbol = :A
    stages = ALLSTAGES[6:end]
    # Arguments consist of all dynamic choice variables
    argvars = filter(var->isdynamic(host(var)), allcontinuousvars(m, st_choice))
    return DualSchema{symbol, false}(stages, argvars)
end

"DualSchema to take derivatives of equilibrium conditions (eqconds) w/r/t
equilibrium quantities (equil) and static choices, for the purpose of solving
for intra-period equilibrium."
function dualschemaB(m::Model)
    #TODO Rename dsFOC
    symbol = :B
    stages = [st_equil, st_equildep, st_choice, st_choicedep]
    # Arguments consist of all equilibrium and static choice variables.
    equilargvars = allcontinuousvars(m, st_equil)
    staticchoiceargvars = filter(var->!isdynamic(host(var)), allcontinuousvars(m, st_choice))
    argvars = vcat(equilargvars, staticchoiceargvars)
    return DualSchema{symbol, false}(stages, argvars)
end

dualschemasAB(m::Model) = hasequilibrium(m) ? Dict(:A => dualschemaA(m), :B => dualschemaB(m)) : Dict(:A => dualschemaA(m))


#####################################
# Derivative Information Extraction #
#####################################

gradient(d::Dual) = float.(d.partials)
jacobian(v::Vector{<:Dual}) = mapreduce(gradient, hcat, v)' # Allocating

function jacobian!(J::AbstractMatrix{Float64}, v::Vector{<:Dual}) # Less allocating
    n = length(v)
    @assert size(J,1) == size(J,2) == n
    for i = 1:n
        J[i,:] .= gradient(v[i])
    end
    return J
end

function jacobian!(J::Matrix{Float64}, dv::Vector{<:Dual}, vars::Vector{<:AbstractVariable}, ds::DualSchema)
    vars_inds = arg_i.(ds, vars)
    J .= jacobian(dv)[:,vars_inds]
    return J
end

function jacobian(dv::Vector{<:Dual}, vars::Vector{<:AbstractVariable}, ds::DualSchema)
    return jacobian!(zeros(length(dv), length(vars)), dv, vars, ds)
end

derivative(ds::DualSchema{S}, d::Dual{S}, var::AbstractVariable, i_inner=1) where {S} = d.partials[arg_i(ds, var, i_inner)]
function tryderivative(ds::DualSchema{S}, d::Dual{S}, var::AbstractVariable) where {S}
    argi = arg_i(ds, var)
    return argi == 0.0 ? 0.0 : d.partials[argi]
end

function derivative(md::ModelData, ds::DualSchema, outvar::AbstractVariable, invar::AbstractVariable, i_inner::Int=1; outvari=nothing, outvar_futurestate::Bool=false)
    if isnothing(outvari)
        @assert n(outvar) == 1
        outdual = getval(hostdata(md, outvar), outvar; ds, usefuturestate=outvar_futurestate)
        return derivative(ds, outdual, invar, i_inner)
    else
        @assert 1 <= outvari <= n(outvar)
        error("Not yet implemented.")
    end
end

function secondderivative(ds::DualSchema{S, true}, d::Dual, var1::AbstractVariable, var2::AbstractVariable, i_inner1=1, i_inner2=1) where S
    return d.partials[arg_i(ds, var1, i_inner1)].partials[arg_i(ds, var2, i_inner2)]
end

#################################
# Data Transfer Between Schemas #
#################################

float(val::Float64) = val
float(d::Dual) = float(ForwardDiff.value(d))

"Convert vals from one schema to another."
@inline toschema(dualval::Dual, ::AbstractVariable, ds::Nothing, ::DualSchema) = float(dualval) # Dual to Float
@inline toschema(floatval::Float64, ::AbstractVariable, ds::Nothing, ds_in::Nothing=nothing)::Float64 = floatval # Float to Float (identity)
@inline toschema(floatval::Float64, var::AbstractVariable, ds::DualSchema, ds_in::Nothing=nothing) = dual(ds, var, floatval) # Float to Dual
@inline toschema(dualval::Dual{S}, ::AbstractVariable, ds::DualSchema{S}, ds_in::Nothing=nothing) where S = dualval # Dual to same Dual (identity)
@inline toschema(dualval::Dual{S}, ::AbstractVariable, ds::DualSchema{S}, ds_in::DualSchema{S}) where S = dualval # Dual to same Dual (identity)
@inline function toschema(dualval::Dual{S_IN}, var::AbstractVariable, ds::DualSchema, ds_in::DualSchema{S_IN}) where {S_IN} # Dual to different Dual
    return seededdual(ds, var, dualval, ds_in)
end
#@inline toschema(enumval::Enum, ::AbstractVariable, ::Nothing, ::Nothing) = enumval

function setargvarsduals!(md::ModelData; ds_out::Schema, ds_in::Schema=nothing)
    setval!.(hostdata.(md, argvars(ds_out)), argvars(ds_out); ds_out, ds_in)
end

function dedual!(hd::HostData, var::AbstractVariable; ds::AbstractDualSchema, usefuturestate::Bool=false)
    setval!(hd, var; ds_in=ds, usefuturestate)
end

function dedual!(md::ModelData, stage::Stage; ds::AbstractDualSchema)
    usefuturestate = stage == st_nextstate
    #TODO This is temporary, until dualschemas support arbitrary sets of variables, not just stages
    vars = [var for var in allcontinuousvars(model(md), stage) if !any(isnan.(getval(md, var; ds, usefuturestate)))]
    setval!.(hostdata.(md, vars), vars; ds_in=ds, usefuturestate)
end

dedual!(md::ModelData, ds::AbstractDualSchema) = dedual!.(md, stages(ds); ds)
