---
header-includes:
  - \usepackage[ruled,vlined,linesnumbered]{algorithm2e}
---

# Bucephalus.jl

Bucephalus.jl is a package that aims to automate the solution of heterogeneous-agent economic models using model-based reinforcement learning.

This is very much a work in progress! Do not expect your model to converge correctly at this stage.

## Usage

To install and use this package, run from the Julia command line:

`] add "https://github.com/jeffreyesun/Bucephalus.jl"`


## Tutorials and Documentation

The documentation is currently confined to this readme, although more will be added by the time this package is ready for public consumption. If you're at all interested in this project, feel free to contact me through github (jeffreyesun) or email (jesun at princeton dot edu).

## Use Cases

This package is designed to solve heterogenous-agent macroeconomic models, with possible market or spatial segmentation.

In these models, many agents interact in an environment in which each agent has an individual state (e.g. wealth), and the environment also may have some state variables common to all agents (e.g. total factor productivity). Agents make choices each period to maximize the net present value of present and future period utilities (called the "return" in RL), given their individual state.

Here I describe the difference between the usual goals of reinforcement learning and my specific domain.

## Framework

Bucephalus has two components
1. A `Model` object, which represents a description of a heterogenous-agent economic model.
2. A learning algorithm which, given a `Model` object, attempts to solve for the policy function, value function, and steady-state distribution of that model.

In future, we aim to include several solvers in this package, as well as an easy and accessible interface for defining your own solvers and learning algorithms.

### The `Model` Object

The `Model` object is a description of the model being solved, and is described using Julia macros. (There is also an under-the-hood syntax for defining models which does not use macros.) Examples of model description can be found in the `examples` folder. In each example, the `Model` object is defined, and then the algorithm is called using the `train` function.


### The Learning Algorithm

The learning algorithm currently implemented is a model-based actor-critic algorithm, some details of which are tailored to the our use case.

One non-standard feature of this setting is that we can differentiate through the transition function. This allows us to use automatic differentiation to directly compute the derivative of the future state with respect to the (continuous) action. Thus, we update the policy function parameters by directly computing the derivative of the state-value function with respect to the policy function parameters.

A representation in pseudocode of the algorithm is given in algorithm_pseudocode.pdf

Note: A performance improvement can be obtained by specifying in the model that equilibrium variables are not used as inputs to actions. E.g. agents choose expenditure instead of quantities, so that their choice will be valid regardless of prices.


#### Limitations

This algorithm has many limitations:
1. The algorithm may not perform well in the presence of strong non-convexities in the value function. Because the policy function network parameters are updated using gradient descent, it is quite possible for it to become stuck at local minima.
2. When the steady state of the model is non-ergodic, the algorithm may fail to cover the state space well. That is, because the steady state is traversed simply by simulating the model forward, the algorithm will fail to cover the steady state space when the steady state exhibits path-dependence. In future, I will try to remedy this by providing the option to begin the simulation at various starting points and simulating in parallel, or by periodically restarting the simulation from a new random initial state.
