import timeit 
import math 
import numpy

function_calls = ['beta_like(x, alpha, beta)', 
#'binomial_like(xd, n, p)',
'betabin_like(xd, alpha, beta, n)',
'cauchy_like(x, alpha, beta)',
'chi2_like(x, nu)',
'exponential_like(x, beta)',
#'exponweib_like(x, alpha, k, loc=0, scale=1)',
'gamma_like(x, alpha, beta)',
#'gev_like(x, xi, mu=0, sigma=1)',
'geometric_like(xdg, p)',
'half_cauchy_like(x, alpha, beta)',
'half_normal_like(x, tau)',
#'hypergeometric_like(x, n, m, N)',
'inverse_gamma_like(x, alpha, beta)',
#'inverse_wishart_like(X, n, C)',
'laplace_like(x, mu, tau)',
#'logistic_like(x, mu, tau)',
'lognormal_like(x, mu, tau)',
#'multinomial_like(x, n, p)',
#' multivariate_hypergeometric_like(x, m)',
#'mv_normal_like(x, mu, tau)',
#'mv_normal_cov_like(x, mu, C)',
#'mv_normal_chol_like(x, mu, sig)',
#'negative_binomial_like(xd, mu, alpha)',
'normal_like(x, mu, tau)',
#'von_mises_like(x, mu, kappa)',
#'pareto_like(x, alpha, m)',
#'truncated_pareto_like(x, alpha, m, b)',
'poisson_like(xd,mu)',
#'truncated_poisson_like(x,mu,k)',
#'truncated_normal_like(x, mu, tau, a=None, b=None)',
#'skew_normal_like(x,mu,tau,alpha)',
't_like(x, nu)',
'noncentral_t_like(x, mu, lam, nu)',
#'discrete_uniform_like(x,lower, upper)',
'uniform_like(x,lower, upper)',
'weibull_like(x, alpha, beta)',
#'wishart_like(X, n, Tau)'
]

setup="""
import pymc.old_distributions as od
import pymc.distributions as d

import numpy as np
x = np.random.normal(size = {size} )**2
alpha = np.random.normal(size = {size} )**2
beta = np.random.normal(size = {size} )**2
nu = np.random.normal(size = {size} )**2 + .5
mu = np.random.normal(size = {size} )**2
tau = np.random.normal(size = {size} )**2
lam = np.random.normal(size = {size} )**2
lower = -np.random.normal(size = {size} )**2
upper = np.random.normal(size = {size} )**2

n = np.random.randint(4,5,size = {size})
xd = np.random.randint(0,1,size = {size})
xdg = np.random.randint(1,2,size = {size})
p = np.random.rand({size})
""" 

scales = [10**i for i in range(6)]
max_scale = max(scales)

def time_at_scales(call, setup, scales):
    return numpy.array([timeit.Timer(call, setup.format(size = scale)).timeit(100) for scale in scales])
numpy.set_printoptions(precision = 3)
print scales
for function in function_calls:
    print function, time_at_scales('d.' + function, setup, scales)/time_at_scales('od.' + function, setup, scales)
        
    
