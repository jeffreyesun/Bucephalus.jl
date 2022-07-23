
##################################
# Save Dual Schemas to ModelData #
##################################

function adddualschemas!(md::ModelData, ds_dict::Dict{Symbol,<:DualSchema})
    merge!(dualschemasdict(md), ds_dict)
    for hd in allhostdata(md), level in keys(ds_dict)
        vardatadualdict(hd)[level] = Dict{Symbol,Ref}()
        nextstatedatadualdict(hd)[level] = Dict{Symbol,Ref}()
    end
    return md
end

####################
# Preallocate Data #
####################

# Note: Variable data on heterogeneous hosts are stored as arrays. Variable data on
#   non-heterogeneous hosts are stored as Refs.

function storepreallocatedvalue!(hd::HostData, var::StateVariable, arr)
    vardatadict(hd)[symbol(var)] = arr
    nextstatedatadict(hd)[symbol(var)] = deepcopy(arr)
    statedatacheckpointdict(hd)[symbol(var)] = deepcopy(arr)
end

storepreallocatedvalue!(hd::HostData, var::AbstractVariable, arr) = vardatadict(hd)[symbol(var)] = arr

function preallocatevalue!(md::ModelData, var::AbstractVariable)
    T = dtype(var)
    @assert T<:Union{Float64,Enum,Match}
    placeholder = dtype(var) <: Enum ? first(instances(T)) : zero(T)
    arr = isheterogeneous(var) ? fill(placeholder, n(var)) : Ref(placeholder) #TODO NaNs
    var isa Parameter && _setval!(arr, value(var))
    hd = hostdata(md, var)
    storepreallocatedvalue!(hd, var, arr)
    return md
end

function preallocatedual!(md::ModelData, var::AbstractVariable, nextstate::Bool, ds::DualSchema{S}) where S
    _templatedual = templatedual(ds)
    hd = hostdata(md, var)
    arrdual = isheterogeneous(var) ? fill(_templatedual, n(var)) : Ref(_templatedual)
    vardictdual = (nextstate ? nextstatedatadualdict : vardatadualdict)(hd, S)
    vardictdual[symbol(var)] = arrdual
    return md
end

function preallocatevalues!(md::ModelData)
    for var in allvars(model(md))
        preallocatevalue!(md, var)
    end
    for ad in agentdata(md) if hasbrain(ad)
        for nn in nns(brain(ad))
            for d in layers(nn)
                preallocate_res!(d, n(ad))
            end
        end
    end end
end

function preallocateduals!(md::ModelData)
    for ds in dualschemas(md)
        for stage in stages(ds)
            for var in allcontinuousvars(model(md), stage)
                preallocatedual!(md, var, stage == st_nextstate, ds)
            end
        end
        for ad in agentdata(md) if hasbrain(ad)
            for nn in nns(brain(ad))
                for d in layers(nn)
                    preallocate_res!(d, n(ad), ds)
                end
            end
        end end
    end
end

function preallocate!(md::ModelData)
    preallocatevalues!(md)
    preallocateduals!(md)
    return md
end

#################
# Compile Rules #
#################

function compilerules!(md::ModelData)
    rules = sortrules(model(md))
    for stage in ALLSTAGES push!(sortedrules(md), CompiledRule[]) end
    for r in rules
        compiledrule = compilerule(md, r)
        push!(sortedrules(md, stage(r)), compiledrule)
    end
    return nothing
end
#TODO benchmark for allocations, typehints

##############
# Initialize #
##############

# Fill in Matches #
#-----------------#

function initialize!(md::ModelData, var::Union{StateVariable,EquilibriumVariable,ChoiceVariable})
    val = setval!(md, var, init(var); usefuturestate=true)
end

function initialize!(md::ModelData)
    m = md.description
    for stage in [st_state, st_equil, st_choice]
        initialize!.(md, allvars(m, stage))
    end
    return md
end

###########
# Compile #
###########

function compile(m::Model; dualschemas=dualschemasAB(m), brains=basicbrains(m))
    validate(m)
    md = ModelData(m, brains)
    adddualschemas!(md, dualschemas)
    preallocate!(md)
    initialize!(md)
    updatemembers!(md; usefuturestate=true)
    compilerules!(md)
    return md
end

############
# Validate #
############

# Validation Items:
# - Every variable has a rule
# - Heterogeneity and stuff
# - Rules are sorted properly
# - Buncha stuff involving compiled rules

function validate(md::ModelData)

end

function validate(cr::CompiledRule)

end