from pymc import *
import numpy as np
import theano.tensor as T
import patsy

np.seterr(invalid = 'raise')

size = 100
true_intercept = 1
true_slope = 2

x = np.linspace(0, 1, size)
y = true_intercept + x*true_slope + np.random.normal(scale=.1, size=size)

model = Model()
Var = model.Var
Data = model.Data

beta = Var('beta', Normal(0, tau=1.), shape=2)
dmatrix = np.array(patsy.dmatrix('x', {'x': x}))
eps = Var('eps', Uniform(0, 10))
data = Data(y, Normal((T.dot(dmatrix, beta)), tau=eps**-2))

#start sampling at the MAP
start = find_MAP(model)
h = approx_hess(model, start) #find a good orientation using the hessian at the MAP

#step_method = split_hmc_step(model, model.vars, hess, chain, hmc_cov)
step = HamiltonianMC(model, model.vars, h, is_cov = False)

ndraw = 3e3
history, state, t = sample(ndraw, step, start)

print "took :", t
traceplot(history)