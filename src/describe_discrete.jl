
######################################
# Describe Discrete Choice Variables #
######################################

function ccptocdf(ccp)
    ccpvec = collect(RtoR⁺.(ccp))
    ccpvec ./= sum(ccpvec)
    ccpvec .= cumsum(ccpvec)
    ccpvec[end] = 1
    return ccpvec
end

"Add the selection rule for Enums
(generate choice probabilities from P_NN and choose between them using unif_shock)."
function addSelectionRule!(a::Agent, discretevar::AbstractVariable{S,<:Union{Enum,Match}}, unif_shock::ShockVariable) where {S}
    _dtype = dtype(discretevar)
    _symbol = symbol(discretevar)

    # TODO Once Bounds are fixed, bound the ccps below
    # Generate a vector of conditional choice probability (ccp) variables, one for each option
    n_options = _dtype <: Enum ? length(instances(_dtype)) : n(partnerhost(model(a), _dtype))
    ccp_names = [string(_symbol)*"["*string(i-1)*"]" for i=1:n_options]
    ccp_names = Symbol.(ccp_names)
    ccp_vars = [ChoiceVariable{Float64}(a, symb_i, Bounds(), 1.0, true) for symb_i in ccp_names]
    _addvariable!.(a, ccp_vars)
    ccp_vecvar = VectorVariable{st_choice,Float64}(a, ccp_vars)
    discretevar.choiceprobs = ccp_vecvar

    # Add the rule that the ccps and shock determine the discrete value
    f_choosediscrete(unif_shock, ccp_vars...) = _dtype(findfirst(>=(unif_shock), ccptocdf(ccp_vars)))
    addRule!(a, inputs=[unif_shock, ccp_vars...], outputs=discretevar, f=f_choosediscrete)
end

#=
function addSelectionRule!(a::Agent, discretevar::AbstractVariable{S,<:Match}, unif_shock::ShockVariable) where {S}
    _dtype = dtype(discretevar)
    _symbol = symbol(discretevar)

    function f_choosediscrete(unif_shock)
        
    end
end
=#

"Allow Agent `a` to choose `discretevar` as a choice variable."
function makeChoiceVariable!(a::Agent, discretevar::AbstractVariable{S,<:Union{Enum,Match}}) where {S}
    _symbol = symbol(discretevar)

    discretevar.ischoicevar = true

    # Generate a shock variable that selects the discrete choice from among the options
    shock_symbol = Symbol(string(_symbol)*"%unifshock")
    unif_shock = addShockVariable!(a, Float64, shock_symbol; dist=Uniform(), checksymbol=false)
    dtype(discretevar) <: Union{Enum,Match} && addSelectionRule!(a, discretevar, unif_shock)

    return discretevar
end

"Add a discrete choice variable to the model, along with variables for its associated choice probabilities."
function addDiscreteChoiceVariable!(h::Host, dtype::DataType, symbol::Symbol=Symbol(""))
    @assert dtype <: Union{Enum,Match}
    symbol = _varsymbol(h, symbol)

    discretevar = addDependentVariable!(h::Host, dtype, symbol; stage=st_shockdep)
    makeChoiceVariable!(h, discretevar)
    return discretevar
end