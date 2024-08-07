#############################################################################
#Instructions for running in Jupyter notebook: launch from link in Github repo: github.com/sandialabs/PyNucleus. Upload file to online notebook, open Terminal online, then type in bash, and then python3 2DLocConductivity.py
#To obtain plots: instead of using original command in Jupyter, type in python3 2DNLocConductivity.py --plotFolder=. This puts the plots in a "local" folder on Jupyter, then we can click on the plots and download them

#############################################################################
#Needed imports
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt
from multiprocessing import Pool
import time
from PyNucleus import (driver,
                       kernelFactory, twoPointFunctionFactory,
                       meshFactory, dofmapFactory,
                       functionFactory, solverFactory,
                       NO_BOUNDARY, HOMOGENEOUS_DIRICHLET)
from PyNucleus_fem.mesh import plotManager

#############################################################################
#Now we initialize everything needed for the main body of code

# Start timer
start = time.time()

# If True, plots mesh on top of solutions; if False does not
meshOn = False

#Mesh construction
numRefinements = 2
mesh = meshFactory('disc', n = 8, radius = 1.0) #unit disc

#Indicates we refine the mesh numRefinements times
for _ in range(numRefinements):
    mesh = mesh.refine()
    
# Two FE spaces corresponding to the meshes
coefficient_dm = dofmapFactory( 'P0d', mesh, NO_BOUNDARY ) #Piecewise constant discretization for coefficients
eqn_dm         = dofmapFactory( 'P1c', mesh ) #Continuous piecewise linear mesh for states (with zero nonlocal boundary values)

#If True, prints data to track intermediate values of designs and states (user input)
Verbose = True

#Macro parameters of the code (user inputs)
# Problem parameters
eta = 1.0 #regularization parameter (in the paper this parameter is not formally introduced and its default value is 1)
assert(eta > 0.0)

delta = 0.01 #horizon parameter
assert(delta > 0.0 and delta < 1.0) #permissible values for this particular mesh

s = 0.99 #fractional parameter
assert(s > 0.0 and s < 1.0)

kernel = kernelFactory('fractional', dim = 2, s = s, horizon = delta ) #kernel

#Admissible design class
aminCst = 0.1 #pointwise bounds on the design class (user input)
amaxCst = 2.0
assert(aminCst > 0 and amaxCst > aminCst)
amin = functionFactory( 'constant', aminCst )
amax = functionFactory( 'constant', amaxCst )
avgCons = 0.5 * ( aminCst + amaxCst )
avg = functionFactory('constant', avgCons) #constant function of original value

# Now we define initial guesses
afunction = avg
afem = coefficient_dm.interpolate( afunction )

rescaledAFunction = twoPointFunctionFactory( 'Lambda', lambda x, y: 0.5 * 0.5*( afunction(x) + afunction(y) ), 1 )

# The initial guess for the state
u = eqn_dm.zeros()

# Placeholders for other vectors that will be used to determine the descent step at each iteration.
firstRHSVecControl = coefficient_dm.zeros()
descentRHS = coefficient_dm.zeros()
temp = eqn_dm.zeros()

# Solver parameters
stepSize = 0.25 #corresponds to the parameter tau in the paper
numIterations = 150 #user input

# Parallelization parameters
wantProc = 1  #set to 1 if want multiprocessing, False if not
numProcs = 20 #how many processes to use (not used if wantProc == False)

# Forcing term (user input), g in L^2
#xcenter = [-0.2,0.1]
#radius = 0.25
#intensity = 3.0
#these are two sample options for forcing terms that were used to generate plots in the paper
#g = functionFactory( 'Lambda', lambda x : intensity if (x[0] - xcenter[0])**2 + ( x[1] - xcenter[1] )**2 < radius*radius else 0.0 )
g = functionFactory( 'constant', 1.0 )

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
Xkernel = kernelFactory( 'fractional', dim = 2, s = s, horizon = delta)
XStiffness = eqn_dm.assembleNonlocal( Xkernel, matrixFormat='H2' )

##creates a placeholder matrix array for use in descent algorithm
eqnStiffness = [XStiffness] * coeffDOFCount

#We only need to assemble the mass matrix for the coefficients once, so we do it outside of the main loop
coefficientMass = coefficient_dm.assembleMass()
coefficient_cg = solverFactory( 'cg', A = coefficientMass )
coefficient_cg.setup()

#############################################################################
#For the calculation of the descent direction, we need to assemble a stiffness matrix with respect to each shape function associated with the discretized space of coefficients. This section of the code performs that task, utilizing parallelization since the individual matrix assemblies do not need to collect data from each other.

def createDerivNLStiffness( t ):
    if Verbose: #Tracks the index of created stiffness matrices when set to True
      print(t)
    eqnStiffness[t] = eqn_dm.assembleNonlocal( kernel, matrixFormat = 'H2', diffusivity =  coefficient_dm.getGlobalShapeFunction(t) )
    return eqn_dm.assembleNonlocal( kernel, matrixFormat = 'H2', diffusivity =  coefficient_dm.getGlobalShapeFunction(t) )

#############################################################################
#This is the main Projected Gradient Descent Loop

print( 'Starting gradient descent')
descent_start_time = time.time()

#Projected Gradient Descent Algorithm
for k in range(numIterations):
  if Verbose:
    print( 'k = ', k )
    print( '  afem.min = ', afem.min(), '  afem.max = ', afem.max() )
    L2NormState = sqrt( u.inner( eqnMass*u ) ) #L2 norm state
    L2NormCoefficient = sqrt( afem.inner( coefficientMass*afem ) ) #L2 norm coefficient
    
    #Compute values of cost functinoal
    compliance = w.inner( u )
    regularization = 0.5 * eta * L2NormCoefficient * L2NormCoefficient #Regularization term is of the form eta/2 * \|a\|^2_{L^2(\Om)}
    costFunc = compliance + regularization

    print( '  compliance = ', compliance )
    print( '  stepsize = ', stepSize )
    print( '  cost = ', costFunc )

  #In each iteration of the descent, the first step is to assemble and solve the state equation, starting with the bi-linear form
  kernel = kernelFactory( 'fractional', dim = 2, s = s, horizon = delta, phi = rescaledAFunction )
  nonlocalStiffness = eqn_dm.assembleNonlocal( kernel, matrixFormat = 'H2', symmetric = True )

  #Set up a CG iterative solver
  solver_CG = solverFactory( 'cg', A = nonlocalStiffness )
  solver_CG.setup()
  solver_CG( w, u ) #solve state equation
  print('  u.max = ', u.max() ) #just for debugging purposes
  
  #If we are in the first iteration of the descent algorithm, we assemble all of the nonlocal stiffness matrices with respect to the shape functions for the discretized control space, that can be used for the remainder of the program's lifespan
  if k == 0:
    print("~~~~~~~~~~~~~~~~~~~~~~`")
    print('Begin assembly of derivative')
    if wantProc == 1: #pools trick
      pool = Pool( processes = numProcs )
      pool.map( createDerivNLStiffness, range( coeffDOFCount ) )
      pool.close()
      pool.join()
      
    else: #no parallelization
      for i in range (coeffDOFCount):
        print(i)
        thisShapeFcn = coefficient_dm.getGlobalShapeFunction(i)
        thisShapeFcn2Pt = twoPointFunctionFactory( 'Lambda', lambda x, y: 0.5*( thisShapeFcn(x) + thisShapeFcn(y) ), 1 ) #conversion of the coefficient to the form A(x, y) = (a(x) + a(y))/2 for use in calculating the appropriately weighted kernel
        kkt = kernelFactory('fractional', dim = 2, s = s, horizon = delta, phi = thisShapeFcn2Pt )
        eqnStiffness[i] = eqn_dm.assembleNonlocal( kkt, matrixFormat = 'H2'  )
    print( 'Passed assembly of derivative ')
    print("~~~~~~~~~~~~~~~~~~~~~~`")
  
  #now we actually use the nonlocal stiffness matrices that were constructed for the shape functions of the discretized control space
  for i in range( coeffDOFCount ):
    if k == 0 :
      thisShapeFcn = coefficient_dm.getGlobalShapeFunction(i)
      thisShapeFcn2Pt = twoPointFunctionFactory( 'Lambda', lambda x, y: 0.5*( thisShapeFcn(x) + thisShapeFcn(y) ), 1 )
      kkt = kernelFactory('fractional', dim = 2, s = s, horizon = delta, phi = thisShapeFcn2Pt )
      eqnStiffness[i] = eqn_dm.assembleNonlocal( kkt, matrixFormat = 'H2'  )

    #Now we may assemble the right-hand side of the equation to be solved to calculate the descent direction.
    temp = eqnStiffness[i]*u
    firstRHSVecControl[i] = stepSize * u.inner( temp )
    
  #This is the RHS for the equation solved to calculate the next candidate (unprojected) coefficient
  descentRHS = coefficient_dm.assembleRHS( ( 1.0 - eta*stepSize )*afunction ) + firstRHSVecControl
  
  #Find the new coefficient by solving mass system with RHS descentRHS, stores result in the variable afem
  coefficient_cg( descentRHS, afem )
  
  #Applies the projection operation to assure afem obeys the prescribed pointwise bounds
  for dof in range( coeffDOFCount ):
    afem[dof] = max( aminfem[dof], min( amaxfem[dof], afem[dof] ) )

  #Conversion of the coefficient to the form A(x, y) = (a(x) + a(y))/2 for use in the next iteration
  afunction = functionFactory( 'lookup', mesh, coefficient_dm, afem )
  rescaledAFunction = twoPointFunctionFactory( 'Lambda', lambda x, y: 0.5 * 0.5*( afunction(x) + afunction(y) ), 1 )
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

#Computes relevant norms
L2NormState = sqrt(u.inner(eqnMass * u)) #L2 norm state
XNormState = sqrt(u.inner(XStiffness * u)) + L2NormState #X norm state
L2NormCoefficient = sqrt(afem.inner(coefficientMass * afem)) #L2 norm coefficient

#Print norms and other data
print('Extrema of optimal design:  afem.min = ', afem.min(), '  afem.max = ', afem.max() )
print('L2 norm state: ', L2NormState)
print('X norm state: ', XNormState)
print('L2 norm design: ', L2NormCoefficient)

##Compute values of cost functional
compliance = w.inner( u ) #this is the compliance term \int_{\Omega}gu (where u is optimal state)
regularization = 0.5*eta*L2NormCoefficient*L2NormCoefficient #regularixation eta/2 * \|a\|^2_{L^2(\Omega)} (where a is optimal design)
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


if d.startPlot('optimal_design'): #plots optimal control
  pm = plotManager(coefficient_dm.mesh, coefficient_dm, defaults = plotDefaults)
  designLabel = 'Nonlocal Optimal Design, delta = ' + str(delta) + ', s = ' + str(s)
  pm.add(afem, label = designLabel)
  pm.plot('optimal design')
  if meshOn:
    mesh.plot()

if d.startPlot('optimal_state'): #plots optimal state
  pm = plotManager(eqn_dm.mesh, eqn_dm, defaults = plotDefaults)
  stateLabel = 'Nonlocal Optimal State, delta = ' + str(delta) + ', s = ' + str(s)
  pm.add(u, label = stateLabel)
  pm.plot('optimal state')
  if meshOn:
    mesh.plot()

d.savePlot( 'optimal_state' )
  
d.finish()

#Stop timer
end = time.time()

print("Time elapsed during the calculation: {time:.2f}sec".format( time = end - start ) )

