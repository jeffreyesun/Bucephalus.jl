
# Bucephalus.jl

Bucephalus.jl is a package that aims to automate the solution of heterogeneous-agent economic models using model-based reinforcement learning.

This is very much a work in progress! Do not expect your model to converge correctly at this stage.

## Usage

To install and use this package, run from the Julia command line:

`] add "https://github.com/jeffreyesun/Bucephalus.jl"`


## Tutorials and Documentation

This package is severely under-documented at the moment. The formal documentation is currently confined to this readme. Function-level documentation will be added more seriously after reworking the backend to use existing packages, and thorough documentation will be added by the time this package is ready for public consumption (see roadmap). If you're at all interested in this project, feel free to contact me through github (jeffreyesun) or email (jesun at princeton dot edu).

## Use Cases

This package is designed to solve heterogenous-agent macroeconomic models, with possible market or spatial segmentation.

In these models, many agents interact in an environment in which each agent has an individual state (e.g. wealth), and the environment also may have some state variables common to all agents (e.g. total factor productivity). Agents make choices each period to maximize the net present value of present and future period utilities (called the "return" in RL), given their individual state. Stochasticity takes the form of random variables drawn each period which are taken as inputs to the transition function.

One key detail is that, in this framework, agents are assumed to be either "representative" or "small", in that they do not internalize their individual effects on aggregate quantities (see roadmap).

## Framework

Bucephalus has two components
1. A `Model` object, which represents a description of a heterogenous-agent economic model.
2. A learning algorithm which, given a `Model` object, attempts to solve for the policy function, value function, and steady-state distribution of that model.

In future, we aim to include several solvers in this package, as well as an easy and accessible interface for defining your own solvers and learning algorithms.

## The `Model` Object

The `Model` object is a description of the model being solved, and is described using Julia macros. (There is also an under-the-hood syntax for defining models which does not use macros.) Examples of model description can be found in the `examples` folder. In each example, the `Model` object is defined, and then the algorithm is called using the `train` function.

### Syntax

The model is defined by defining agent types and possibly segmentations (such as locations), and then defining variables.

There are nine "stages" of variables, in terms of when they are computed in the course of a single iteration of the simulation:
0. Parameters. These are set to constants at the beginning of the model definition and never changed.
1. State variables. These are variables that define the state of the economy at the beginning of a period -- the output of the transition function.
2. State-dependent variables. These are variables that depend only on state variables (and parameters), but are not given as the output of the transition function.
3. Equilibrium variables. Conventionally thought of as "prices," these are variables that are defined in terms of an equilibrium condition (e.g. "market clearing conditions").
4. Equilibrium-dependent variables. Variables that depend on only equilibrium (and earlier-stage) variables.
5. Choice variables. Variables chosen by agents -- the output of the policy function.
6. Choice-dependent variables. These include utility, or reward.
7. Shock varaibles. Variables with a specified random distribution. These can be used as inputs to the transition function to model stochasticity.
8. Next-state variables. Strictly speaking, these are the same as the state variables, but must be assigned a rule for computing them: the transition functions.

In general, each variable has the following components:
- A stage, determining when it is computed in the simulation of the model.
- A host: either an agent type, a segmentation, or the model itself. This is the level of the model that the variable is specific to. For example, a variable with the model as a host is common to all agents, e.g. total factor productivity. A variable with an agent type as a host will be specific to each agent, e.g. wealth or labor endowment. A variable with a segmentation as a host will have a different value for each segmentation. For example, a variable "location-specific productivity" might have a different value for each location.
- A function or distribution defining it. This determines how the variable is computed or drawn.

The general syntax looks like this:

`@vartype host.var(inputvar1, inputvar2) = f_var(inputvar1,inputvar2)`

When an input variable `inputvar` has `host` as its host, the function will implicitly take only the individual value of `inputvar` as its input. To take the whole vector of values as input, use `Ref(inputvar)`:

`@statedep hh.meanpeerwealth(Ref(wealth)) = mean(wealth)`

This will differ somewhat by stage. For example, choice variables will not have an associated function or distribution, because they are implicitly taken from the policy function.

A macro defining a variable will create a Julia variable in the macro's calling context with the same name as the model variable. For example,

`@statevar hh.l = LogNormal()`

is unrolled into

`l = Bucephalus.addStateVariable!(hh, Float64, :l; init=LogNormal())`

When segmentations are present, an additional variable type, `Match` is provided, along with two additional operators on variables: `By` and `ByMembers`.

A `Match` defines a relationship between an agent and a segmentation. Take "location" as an example. An agent does not simply "belong" to a single location, but may have relations such as `residence` and `workplace` that take a location as a value. These are declared using the `@matchvar` macro:

`@matchvar hh.residence::Match{:loc} = rand(1:2, n_hh)`

This defines a variable `residence` as a state variable of agent type `hh`, which takes a `loc` as a value. The resulting variable `residence` will have a property `host` (in this case, `hh`) and `partnerhost` (in this case, `loc`).  I use the term "match" in the hopes of one day supporting relations between agents, such as "spouse" or "landlord."

The construct `By(var, matchvar)` is a construct that acts like a variable. The host of the `By` construct must be the host of `matchvar`, and `var` must be a variable on the partnerhost of `matchvar`. For example, `By(rent, residence)` refers to the rent in an agent's residence location. Rent is a location-specific value, and so `By(rent, residence)` is computed by finding the residence location of an agent, and returning the rent at that location.

The construct `ByMembers(var, matchvar)` is similarly a construct that acts like a variable. The host of the `ByMembers` construct must be the partnerhost of `matchvar`, and `var` must be a variable whose host is the host of the `matchvar`. For example, `ByMembers(wealth, residence)` refers to the vector of the `wealth` values of all the residents of a certain location.

These descriptions are quite abstract. Please check out the examples to see them in action!

## The Learning Algorithm

The learning algorithm currently implemented is a model-based actor-critic algorithm, some details of which are tailored to the our use case.

One non-standard feature of this setting is that we can differentiate through the transition function. This allows us to use automatic differentiation to directly compute the derivative of the future state with respect to the (continuous) action. Thus, we update the policy function parameters by directly computing the derivative of the state-value function with respect to the policy function parameters.

A representation in pseudocode of the algorithm is given in algorithm_pseudocode.pdf

Note: A performance improvement can be obtained by specifying in the model that equilibrium variables are not used as inputs to actions. E.g. agents choose expenditure instead of quantities, so that their choice will be valid regardless of prices.


#### Limitations

This algorithm has many limitations:
1. The algorithm may not perform well in the presence of strong non-convexities in the value function. Because the policy function network parameters are updated using gradient descent, it is quite possible for it to become stuck at local minima.
2. When the steady state of the model is non-ergodic, the algorithm may fail to cover the state space well. That is, because the steady state is traversed simply by simulating the model forward, the algorithm will fail to cover the steady state space when the steady state exhibits path-dependence. In future, I will try to remedy this by providing the option to begin the simulation at various starting points and simulating in parallel, or by periodically restarting the simulation from a new random initial state.
3. Currently, the inputs to the policy function must be specified manually, and performance is poor when the entire state space is used. This will be unnecessary after implementing the "generalized moments" of [DeepHAM](https://arxiv.org/pdf/2112.14377.pdf).

## Roadmap

1. Replace large parts of the hand-coded backend with existing packages. This should be a great improvement to the package, in addition to significantly reducing the size of its source code.
2. Implement the generalized moments of [DeepHAM](https://arxiv.org/pdf/2112.14377.pdf). In the case of segmentations, implement nested generalized moments, where one set of generalized moments is computed for each segmentation, and another set is computed using the vector of generalized moments for each segmentation as its input.
3. Implement other solvers and learning algorithms, including the brute-force solver.
4. Find robust strategies to draw starting states for learning episodes. Allow the user to configure the method used.
5. Implement methods to test the optimality of policy networks, such as Euler equation errors.
6. Allow agents to internalize their effect on aggregate quantities. This is necessary, e.g., for models of monopolistic competition.