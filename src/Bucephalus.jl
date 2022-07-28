module Bucephalus

###########
# Exports #
###########
export @parameter, @statevar, @matchvar, @choicevar, @choicevar!, @statedep, @equilvar, @equildep, @choicedep, @choicedep, @equilibriumcondition, @utility!, @shock, @transition
export @agent, @utility, @segmentation
export Model, compile, iterate!, solve
export addParameter!, addStateVariable!, addMatchVariable!, addEquilibriumVariable!, addChoiceVariable!, addShockVariable!, addDependentVariable!, addRule!
export By, ByMembers
export getval, train
export Bounds, Match # maybe
export addAgent!, addMarket!
export partnerhost
# Exports to eventually remove (maybe)
export EnumSampler, makeChoiceVariable!, addEquilbriumConditions!, addfactor!, addPayoffRule!, addShockVariable!
export st_param, st_state, st_statedep, st_equil, st_equildep, st_choice, st_choicedep, st_shock, st_shockdep, st_nextstate
export train

###########
# Imports #
###########
using ForwardDiff
using Distributions
using NLsolve
using Knet: relu, elu, Param, param, param0, params, xavier_uniform, @diff, mat, sigm, grad, *
using Statistics: mean
import ForwardDiff.Dual
import ForwardDiff.Partials
using LinearAlgebra
using StatsBase
import Random: rand, rand!
import Base: zero, zeros

###############
# Source Code #
###############
include("types.jl")
include("bounds.jl")
include("model_description.jl")
include("dualschema.jl")
include("brain.jl")
include("model_data.jl")

include("describe.jl")
include("describe_discrete.jl")
include("compile.jl")
include("simulate.jl")
include("learn.jl")

#TODO Separate these two groups of files as folders

end