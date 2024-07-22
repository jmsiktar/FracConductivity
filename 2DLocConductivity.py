#############################################################################
#Instructions for running in Jupyter notebook: launch from link in Github repo: github.com/sandialabs/PyNucleus. Upload file to online notebook, open Terminal online, then type in bash, and then python3 2DLocConductivity.py
#To obtain plots: instead of using original command in Jupyter, type in python3 2DLocConductivity.py --plotFolder=. This puts the plots in a "local" folder on Jupyter, then we can click on the plots and download them

#############################################################################
#Needed imports
import matplotlib.pyplot as plt
import numpy as np
from multiprocessing.pool import ThreadPool
import time
from numpy import sqrt
from PyNucleus import (driver,
                       kernelFactory, twoPointFunctionFactory,
                       meshFactory, dofmapFactory,
                       functionFactory, solverFactory,
                       NO_BOUNDARY)
from PyNucleus_fem.mesh import plotManager

#############################################################################
#Now we initialize everything needed for the main body of code

#Start timer
start = time.time()

#If True, plots mesh on top of solutions; if False does not (user input)
meshOn = False

#Mesh construction
numRefinements = 3
mesh = meshFactory('disc', n = 8, radius = 1.0) #the n is a geometric parameter for disc construction, leave it at 8 unless changing the domain to a different shape

#Indicates we refine the mesh numRefinements times
for _ in range(numRefinements):
    mesh = mesh.refine()
mesh

# Two FE spaces corresponding to the meshes
coefficient_dm = dofmapFactory('P0d', mesh, NO_BOUNDARY) #Piecewise constant discretization for coefficients
eqn_dm = dofmapFactory('P1c', mesh) #Continuous piecewise linear mesh for states (with zero Dirichlet boundary values)

#If True, prints data to track intermediate values of designs and states (user input)
verbose = True

#Macro parameters of the code (user inputs)
eta = 1.0 #regularization parameter (in the paper this parameter is not formally introduced and its default value is 1)
assert(eta > 0.0)

aminCst = 0.1 #pointwise bounds on the design class
amaxCst = 2.0
amin = functionFactory('constant', aminCst)
amax = functionFactory('constant', amaxCst)
assert(aminCst > 0 and amaxCst > aminCst)

#These lines construct an initial guess for an optimal design and then discretize it
avgCons = 0.5 * (aminCst + amaxCst)
avg = functionFactory('constant', avgCons) #constant function of original value; this serves as an initial guess for an optimal design
afunction = avg #design initial guess
afem = coefficient_dm.interpolate( afunction ) #design function initial guess fe version

#The initial guess for the state (this is really just a placeholder variable)
u = eqn_dm.zeros()

#Solver parameters
stepSize = 0.15 #corresponds to the parameter tau in the paper
numIterations = 20

#Parallelization parameters
wantProc = 0 #set to 1 if want multiprocessing, 0 if not
numProcs = 5 #How many processes to use (not used if wantProc == 1)

# Forcing term (user input), g in L^2
g = functionFactory('constant', 1.0)

# Discretizes g so we can calculate compliance with respect to discrete states
gDisc = eqn_dm.interpolate(g)

# These functions are tools for solving the discrete state equation and applying the projection operator at the appropriate places in the descent algorithm
w = eqn_dm.assembleRHS(g)
aminfem = coefficient_dm.interpolate( amin )
amaxfem = coefficient_dm.interpolate( amax )

#Number of DOFs for each FE space
coeffDOFCount = coefficient_dm.num_dofs
eqDOFCount    = eqn_dm.num_dofs

#Prints number of DOFs for each FE space
print( "eqDOFCount = ", eqDOFCount )
print( "coeffDOFCount = ", coeffDOFCount )

#Used to compute relevant norms (with no coefficient)
eqnMass = eqn_dm.assembleMass()
laplacianOne = eqn_dm.assembleStiffness()


#We only need to assemble the mass matrix for the coefficients once, so we do it outside of the main loop
coefficientMass = coefficient_dm.assembleMass()
coefficient_direct = solverFactory('lu', A = coefficientMass)
coefficient_direct.setup()

#Auxiliary vectors created for use in descent algorithm
firstRHSVecControl = coefficient_dm.zeros()
temp = eqn_dm.zeros()

#############################################################################
#For the calculation of the descent direction, we need to assemble a stiffness matrix with respect to each shape function associated with the discretized space of coefficients. This section of the code performs that task, utilizing parallelization since the individual matrix assemblies do not need to collect data from each other.

def createDerivStiffness(t):
    if verbose: #Tracks the index of created stiffness matrices when set to True
      print(t)
    thisShapeFcn = coefficient_dm.getGlobalShapeFunction(t)
    outMatrix = eqn_dm.assembleStiffness(diffusivity = thisShapeFcn)
    return outMatrix

#Creates a placeholder matrix to be a stiffness matrix for use in descent algorithm
localStiffness_aa = eqn_dm.assembleStiffness()

#Creates a placeholder matrix array for use in descent algorithm
eqnStiffness = np.array( np.ones( coeffDOFCount ), ndmin = 1, dtype = type( localStiffness_aa ) )

print('Begin assembly of derivative')

if wantProc == 1: #multiprocessing trick
  if __name__ == '__main__':
    pool = ThreadPool( numProcs )
    eqnStiffness = pool.map( createDerivStiffness, range( coeffDOFCount ) )

elif wantProc == 0: #no parallelization
  eqnStiffness = [ createDerivStiffness(i) for i in range( coeffDOFCount ) ]

print('Passed assembly of derivative')


#############################################################################
#This is the main Projected Gradient Descent Loop

 
print('Starting gradient descent')
descent_start_time = time.time()

#Projected Gradient Descent Algorithm
for k in range(numIterations):
  if verbose:
    print( 'k = ', k )
    print( '  afem.min = ', afem.min(), '  afem.max = ', afem.max() )
    L2NormState = sqrt(u.inner(eqnMass * u)) #L2 norm state
    L2NormCoefficient = sqrt(afem.inner(coefficientMass * afem)) #L2 norm coefficient
    
    ##compute values of cost functinoal
    compliance = gDisc.inner(eqnMass * u) #this is the compliance term \int_{\Omega}gu
    regularization = eta * 0.5 * L2NormCoefficient * L2NormCoefficient #regularixation eta/2 * \|a\|^2_{L^2(\Omega)}
    costFunc = compliance + regularization
    print('The current value of the cost functional is :', costFunc)
    
  #At each step we assemble and solve the state equation
  laplacian = eqn_dm.assembleStiffness(diffusivity = 0.5 * afunction) #For the scalar 2D design problem we want to assemble the weighted Poisson System, i.e., bi-linear form is B_0(a)(u, v) = 1/n\int_{\Omega}a(x) \grad u(x) * \grad v(x)dx.

  #Set up an LU direct solver for the state equation
  solver_direct = solverFactory('lu', A=laplacian)
  solver_direct.setup()
  solver_direct(w, u) #Solve state equation with RHS w
  
  
  for i in range(coeffDOFCount):
    temp = eqnStiffness[i] * u
    firstRHSVecControl[i] = stepSize * u.inner(temp) #generates \tau \int_{\Omega}phi_i(x)|\grad u(x)|^2dx where phi_i is the ith shape function associated with the discretized coefficient class

  secondRHSTermControl = (1.0 - eta * stepSize) * afunction #goal is to calculuate (1 - tau)\int_{\Omega}a(x)b(x) where a is current/previous coefficient and b is control space test function

  #this is the RHS for the equation solved to calculate the next candidate (unprojected) coefficient
  descentRHS = coefficient_dm.assembleRHS(secondRHSTermControl) + firstRHSVecControl
  
  #Find the new coefficient by solving mass system with RHS descentRHS, stores result in the variable afem
  coefficient_direct(descentRHS, afem)
  
  #Applies the projection operation to assure afem obeys the prescribed pointwise bounds
  for dof in range( coeffDOFCount ):
    afem[dof] = max( aminfem[dof], min( amaxfem[dof], afem[dof] ) )

  #Makes the current design function an algebraic function
  afunction = functionFactory('lookup', mesh, coefficient_dm, afem)
  print("==========================")
  #This marks the end of the projected gradient descent loop
  
descent_end_time = time.time()

#Outputs timer info for descent algorithm
print( "================ Problem solved! ================ ")
print( "eqDOFCount = ", eqDOFCount )
print( "coeffDOFCount = ", coeffDOFCount )
print( "The iterations took {time:.2f}sec".format( time = descent_end_time - descent_start_time ) )

#############################################################################
#Postprocessing: we compute relevant norms for the calculated optimal design and state, and then plot our results

L2NormState = sqrt(u.inner(eqnMass * u)) #L2 norm state
H1NormState = sqrt(u.inner(laplacianOne * u)) + L2NormState #H10 norm state
L2NormCoefficient = sqrt(afem.inner(coefficientMass * afem)) #L2 norm coefficient

#Print relevant norms
print('Extrema of optimal design:  afem.min = ', afem.min(), '  afem.max = ', afem.max() )
print('L2 norm state: ', L2NormState)
print('H1 norm state: ', H1NormState)
print('L2 norm design: ', L2NormCoefficient)

#Compute values of cost functinoal
compliance = gDisc.inner(eqnMass * u) #this is the compliance term \int_{\Omega}gu (where u is optimal state)
regularization = eta * 0.5 * L2NormCoefficient * L2NormCoefficient #regularixation eta/2 * \|a\|^2_{L^2(\Omega)} (where a is optimal design)
costFunc = compliance + regularization
print('The approximate value of the compliance is :', compliance)
print('The approximate value of the cost functional is :', costFunc)

#Create plots

d = driver()
d.declareFigure('optimal_design')
d.declareFigure('optimal_state')
d.process()

#Formatting info for plots
plotDefaults = {'flat': True}

d.set('plotFolder','RESULTS')

#Plots optimal design
if d.startPlot('optimal_design'):
  pm = plotManager(coefficient_dm.mesh, coefficient_dm, defaults = plotDefaults)
  pm.add(afem, label = 'Local Optimal Design')
  pm.plot('optimal design')
  if meshOn:
    mesh.plot()

if d.startPlot('optimal_state'): #plots optimal state
  pm = plotManager(eqn_dm.mesh, eqn_dm, defaults = plotDefaults)
  pm.add(u, label = 'Local Optimal State')
  pm.plot('optimal state')
  if meshOn:
    mesh.plot()

d.savePlot('optimal_state')

  
d.finish()

#Stop timer
end = time.time()
print("Time elapsed during the calculation: {time:.2f}sec".format( time = end - start ) )

