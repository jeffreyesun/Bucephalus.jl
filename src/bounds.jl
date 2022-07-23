
######################
# Accessor Functions #
######################

boundvar(b::Bound) = b.boundvar
inclusive(b::Bound) = b.inclusive
lowerbound(b::Bounds) = b.lower
upperbound(b::Bounds) = b.upper
lowerbound(var::AbstractVariable) = lowerbound(bounds(var))
upperbound(var::AbstractVariable) = upperbound(bounds(var))
# Synthetic Properties
isbounded(b::Bound{T, true}) where {T} = boundvar(b) != -Inf
isbounded(b::Bound{T, false}) where {T} = boundvar(b) != Inf
boundvalue(md::ModelData, b::Bound{<:AbstractVariable}, ds::Schema=nothing) = float.(getval(md, boundvar(b); ds))
boundvalue(::ModelData, b::Bound{Float64}, ::Schema=nothing) = boundvar(b)

#########################################
# Smooth Transformations Between Spaces #
#########################################

# From ℝ #
#--------#

"Take ℝ to ℝ⁺ = (0,∞)."
function RtoR⁺(x)
    x == -Inf && return zero(x)
    x == Inf && return Inf
    return (1 + x + abs(x))/(2 + (abs(x) - x))
end
RtoR⁺_nocheck(x) = (1 + x + abs(x))/(2 + (abs(x) - x))
"Take ℝ⁺ to 𝕀 = (0,1)."
R⁺toI(x) = 1-1/(x+1)
"Take ℝ to 𝕀"
RtoI = R⁺toI∘RtoR⁺

# To ℝ #
#------#

"Take 𝕀 to ℝ⁺"
function ItoR⁺(y)
    @assert 0 <= y <= 1
    return 1/(1-y) - 1
end
"Take ℝ⁺ to ℝ"
function R⁺toR(y)
    @assert y >= 0
    if y >= 0.5
        return y - 0.5
    else
        return 1-1/(2y)
    end
end
"Take 𝕀 to ℝ"
ItoR = R⁺toR∘ItoR⁺

# Between I and finite intervals (a,b) #
#--------------------------------------#

Itoab(x, a, b) = (@assert 0 <= x <= 1; @assert a < b; a + x*(b-a))
abtoI(y, a, b) = (@assert a <= y <= b; (y-a)/(b-a))

# Between ℝ and Finite Intervals (a,b) #
#--------------------------------------#

Rtoab(x, a, b) = Itoab(RtoI(x), a, b)
abtoR(y, a, b) = ItoR(abtoI(y, a, b))

# Between ℝ and Not Necessarily Finite Intervals #
#------------------------------------------------#

function from_R(x, lowerval::Float64, upperval::Float64)
    if lowerval != -Inf && upperval != Inf
        return Rtoab(x, lowerval, upperval)
    elseif lowerval != -Inf
        return lowerval + RtoR⁺(x)
    elseif upperval != Inf
        return upperval - RtoR⁺(x)
    end
    return x
end

function to_R(y, lowerval::Float64, upperval::Float64)
    if lowerval != -Inf && upperval != Inf
        return abtoR(y, lowerval, upperval)
    elseif lowerval != -Inf
        return R⁺toR(y - lowerval)
    elseif upperval != Inf
        return R⁺toR(upperval - y)
    end
    return y
end

from_R′(x, lowerval::Float64, upperval::Float64) = ForwardDiff.derivative(x) do x from_R(x, lowerval, upperval) end
to_R′(y, lowerval::Float64, upperval::Float64) = ForwardDiff.derivative(y) do y to_R(y, lowerval, upperval) end

# Between ℝ and Bound-Defined Spaces #
#------------------------------------#

BoundableVariable = Union{ChoiceVariable,EquilibriumVariable} #TODO Holy Trait?

function from_R(md::ModelData, var::BoundableVariable, x; ds::Schema=nothing)
    b = bounds(var)
    lowerval = boundvalue(md, lowerbound(b), Try(ds))
    upperval = boundvalue(md, upperbound(b), Try(ds))
    return from_R.(x, lowerval, upperval)
end

function to_R(md::ModelData, var::Union{ChoiceVariable,EquilibriumVariable}, y; ds::Schema=nothing)
    b = bounds(var)
    lowerval = boundvalue(md, lowerbound(b), ds)
    upperval = boundvalue(md, upperbound(b), ds)
    return to_R.(y, lowerval, upperval)
end

from_R′(md::ModelData, var::BoundableVariable, x; ds::Schema=nothing) = ForwardDiff.derivative(x) do x
    from_R(md, var, x; ds)
end

to_R′(md::ModelData, var::BoundableVariable, y; ds::Schema=nothing) = ForwardDiff.derivative(y) do y
    to_R(md, var, y; ds)
end


###############
# Constructor #
###############

function Bounds(lower::Union{Float64,AbstractVariable}=-Inf, upper::Union{Float64,AbstractVariable}=Inf;
    lowerinclusive::Bool=false, upperinclusive::Bool=false
)
    #lower < upper || error("Lower bound must be lower than upper bound.")
    #@assert stages make sense
    @assert !(lowerinclusive || upperinclusive)
    lowerbound = Bound{typeof(lower),true}(lower, lowerinclusive)
    upperbound = Bound{typeof(upper),false}(upper, upperinclusive)
    return Bounds(lowerbound, upperbound)
end
