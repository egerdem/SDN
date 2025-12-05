
Here are the one-sentence explanations for each script:

research/optimisation_singleC.py
: Optimizes a scalar c value independently for each source position, resulting in a different optimal parameter for every source location.
research/optimisation_wallC.py
: Optimizes a 6-element vector (one coefficient per wall) globally across all source positions, finding the best directional weighting that works for everyone.
research/optimisation_globalC.py
: Optimizes a single scalar c value globally across all source positions, finding the one "best fit" parameter that works on average for everyone.

previous variables in sdn_core is changed to:
injection_vector -> source_injection_vector
injection_c_vector -> node_weighting_vector

source_injection_vector: The explicit [c, cn, cn, cn, cn] shape (5 elements).
node_weighting_vector: The list of c parameters for each wall (6 elements).

A colleague suggested slighlty new optimisation trials:

1) Assign random SW coefficients to all nodes, making sure that the l2 norm is equal to one and then optimise individual norms by multiplying vectors with constants – highest priority 

Suggestion 1 separates the shape of the injection from its power.

Random Shape: Instead of your specific [c, cn, cn, cn, cn] distribution, you generate a random 5-element source_injection_vector for each of the 6 walls.
Normalization: You normalize these random vectors so their length (L2 norm) is exactly 1. This fixes their "direction" in parameter space.
Optimization: You then optimize only 6 scalar values (one multiplier per wall).
Why do this? This tests if the specific distribution (the shape) matters at all.

If this random method works as well as your carefully designed c parameter, it proves that only the total energy per wall matters, and the specific way it distributes to neighbors is irrelevant.
If it performs worse, it validates that your [c, cn...] structure is physically important.

However what he missed is that the sum of the coefficients of source_injection_vector should be equal to K-1 = 5 not to disrupt the first order reflection amplitudes. That constraint comes from the SDN structure.

That is why instead of norm = 1, I did summation=5. but now I am not sure what can we infer from this experiment. or how to do it such that the aim of colleageue is achieved by also satisfying the sum constraint?

2) Optimise all 30 SW coefficients across all locations – second highest priority – you can assign same loading coefficients to all nodes, so you have 5 parameter, but make the problem symmetric through the selection of source and receiver positions  

Explanation: This approach tries to find the single best 5-element injection pattern that works for the whole room.

The Constraint: Instead of assuming the pattern must be [c, cn, cn, cn, cn] (where neighbors are equal), you allow any 5 values: [w1, w2, w3, w4, w5].
Global Sharing: You force every wall to use this same 5-element vector. This reduces the problem from 30 parameters (6 walls $\times$ 5 weights) down to just 5 parameters. (or 30 could also be tried)
Symmetry: Since you are finding one "universal" vector, you must ensure your source/receiver positions in the optimization dataset are symmetric. If they aren't, the optimization might learn a biased vector (e.g., "always send more energy to the left") that only works for one specific source position.
Why do this? This checks if there is a better fundamental injection shape than your proposed [c, cn...]. For example, it might discover that sending negative energy to the opposite neighbor is beneficial, which your current model doesn't allow.