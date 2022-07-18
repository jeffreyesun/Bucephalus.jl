
###########
# Iterate #
###########

function iterate_inner!(md::ModelData)
    # Apply equil deps
    # Ask brain for choices
    # Apply choice deps
end

function choose!(md::ModelData, a::Agent, ds::Schema=nothing)
    if isdynamic(a)
        val = get_choice(md, a; ds)
        for var in localvars(a, st_choice)
            setval!(md, var, val[symbol(var)]; ds)
        end
    end
    return nothing
end

function drawshocks!(md::ModelData)
    for shockvar in allvars(model(md), st_shock)
        if isheterogeneous(shockvar)
            data = vardata(md, shockvar)
            rand!(dist(shockvar), data)
        else
            setval!(md, shockvar, rand(dist(shockvar)))
        end
    end
end

function iteratedynamicchoice!(md::ModelData, ds::Schema=nothing)
    for a in agents(model(md)) if isdynamic(a)
        choose!(md, a, ds)
    end end
end

computedeps!(md::ModelData, depstage::Stage, ds=nothing) = apply!.(sortedrules(md, depstage), ds)

function get_iterate_equilibrium(md::ModelData)
    dsB = dualschema(md, :B)

    staticagents = filter(!isdynamic, agents(model(md)))
    if !isempty(staticagents)
        staticagent = only(staticagents) #TODO Support multiple static agents
        staticchoicevars = localvars(staticagent, st_choice)
        staticchoicevals = @. to_R(md, staticchoicevars, getval(md, staticchoicevars)) #TODO Preallocate
    else
        staticchoicevars = AbstractVariable[]
        staticchoicevals = Float64[]
    end
    
    dynamicagent = only(filter(isdynamic, agents(model(md)))) #TODO Suport multiple dynamic agents
    #TODO Save these preallocated arrays in a cache on ModelData and reuse
    
    # x
    equilvars = allvars(model(md), st_equil)
    xvars = vcat(equilvars, staticchoicevars)
    
    # f
    eqcondvars = alleqconds(model(md))
    eqcondvec::Vector{Float64} = getval.(md, eqcondvars)
    n_equil = length(equilvars)
    n_staticchoice = length(staticchoicevars)
    @assert length(eqcondvars) == n_equil + n_staticchoice
    eqconddualvec = getval.(md, eqcondvars; ds=dsB)
    
    # Bounds transformations
    eqvals = @. to_R(md, equilvars, getval(md, equilvars)) #TODO Preallocate
    xinit = vcat(eqvals, staticchoicevals)
    from_R′eqvals = zeros(n_equil) #TODO Preallocate
    from_R′staticchoicevals = zeros(n_staticchoice) #TODO Preallocate
    from_R′xinit = zero(xinit)

    vars_inds = arg_i.(dsB, xvars)
    @assert all(vars_inds .== 1:length(eqcondvars))
    
    "Computes equilibrium conditions (including FOCs) and their respective jacobian,
    unpacking arguments and applying bounds transformations."
    function iterate_equilibrium!(fvec, J, x::Vector{Float64})
        @assert !any(isnan.(x))
        @assert length(x) == n_equil + n_staticchoice
        # Unpack the arguments
        eqvals .= x[1:n_equil]
        staticchoicevals .= x[n_equil+1:end]

        # Apply Bounds transformations
        if !isnothing(J)
            from_R′xinit[1:n_equil] .= from_R′.(md, equilvars, eqvals)
            from_R′xinit[n_equil+1:end] .= from_R′.(md, staticchoicevars, staticchoicevals; ds=dsB)
        end
        @. eqvals = from_R(md, equilvars, eqvals) #TODO ds?
        @. staticchoicevals = from_R(md, staticchoicevars, staticchoicevals; ds=dsB)
        #TODO Get value, derivatives simultaneously?

        # Iterate equilibrium
        iterate_equilibrium!(fvec, J, eqvals, staticchoicevals)

        # Write derivatives to J
        if !isnothing(J)
            J .*= from_R′xinit'
        end
        #println(fvec)
        return fvec, J
    end

    "Computes equilibrium conditions (including FOCs) and their respective jacobian."
    function iterate_equilibrium!(fvec, J, eqvals::Vector{Float64}, staticchoicevals::Vector{Float64})
        # Simulate forward
        setval!.(md, equilvars, eqvals; ds=dsB)
        computedeps!(md, st_equildep, dsB)
        iteratedynamicchoice!(md, dsB)
        setval!.(md, staticchoicevars, staticchoicevals; ds=dsB) #TODO Support heterogeneous static agents
        computedeps!(md, st_choicedep, dsB)

        # Read results
        eqconddualvec .= getval.(md, eqcondvars; ds=dsB)

        # Write values to f
        if !isnothing(fvec)
            @. fvec = float(eqconddualvec)
        end

        isnothing(J) || jacobian!(J, eqconddualvec, xvars, dsB)
        return nothing
    end

    return iterate_equilibrium!, xinit
end

function solve_equilibrium!(md::ModelData)
    fj!, xinit = get_iterate_equilibrium(md)
    res = nlsolve(only_fj!(fj!), xinit; method=:trust_region, linesearch=HagerZhang) #TODO anderson?
    global gres = res
    @assert converged(res)
    return res
end

function iterate!(md::ModelData, advancestate::Bool=true)
    dsA = dualschema(md, :A)
    advancestate && advancestate!(md)
    computedeps!(md, st_statedep)
    if hasequilibrium(model(md))
        dsB = dualschema(md, :B)
        solve_equilibrium!(md)
        dedual!(md, dsB)
    else
        iteratedynamicchoice!(md)
        computedeps!(md, st_choicedep)
    end
    #TODO Name/document this function better
    setargvarsduals!(md; ds_out=dsA)
    computedeps!(md, st_choicedep, dsA)
    drawshocks!(md)
    setval!.(md, allcontinuousvars(model(md), st_shock); ds_out=dsA)
    computedeps!(md, st_shockdep, dsA)
    computedeps!(md, st_nextstate, dsA)
    dedual!(md, dsA)
    #TODO Validate all duals values match
    return md
end

#function simulate!(md::ModelData, T::Int)
#    initialize!(md)
#    for t=1:T
#        iterate!(md)
#    end
#end
