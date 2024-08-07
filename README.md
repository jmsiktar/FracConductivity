# FracConductivity
Developers: Abner J. Salgado, Tadele A. Mengesha, Joshua M. Siktar

This repository includes code used to simulate a nonlocal optimal conductivity problem, where the cost functional to be minimized is of a compliance type and the constraint is of a weighted fractional laplacian type. This code is written in the Python finite element library PyNucleus (originally written in PyNucleus v 1.0, Python V3.9). The PyNucleus repository is available here: https://github.com/sandialabs/PyNucleus.

Mathematical features:

~Kernels are of a truncated fractional type, so the user inputs are the horizon parameter and the fractional parameter 

~The cost functional is a compliance term (with fixed forcing data) plus a regularization term

~The admissible designs are positive and have pointwise bounds from above and below

~The design space is discretized with piecewise constant functions, whereas the state space is discretized with continuous piecewise linear functions

In addition to a nonlocal code file title: 2DNLocConductivity.py), a code for the corresponding local problem is also included (file title: 2DLocConductivity.py). This code uses the same cost functional, but the constraint is a weighted Poisson equation.

The developers of this code are thankful for Christian Glusa answering questions pertaining to the development of our code.
