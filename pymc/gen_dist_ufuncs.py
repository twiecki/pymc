import ufunc_gen as ug
import os

def dist_ufuncs(name, docstring, signature,constraints, like_calc, grad_calcs = {}):
    vars = [ug.NumpyVarDefinition(*var.split(':')) for var in signature.split(',')]
    
    return [ug.UFuncDefinition(name +'_like', vars, 'float', constraints, like_calc, docstring = docstring, array_out=False),
            [ug.UFuncDefinition(name +'_' + var + 'grad', vars, 'float', constraints, calc, array_out=True) for var, calc in grad_calcs.iteritems()]]


ufunc_definitions = [
dist_ufuncs('bernoulli',
            '',
    'x : int, p : float',
    'x >= 0 and x <= 1 and p >= 0.0 and p <= 1.0',
    """
    if x == 1:
        out = out + p
    else :
        out = out + 1.0 - p
    """,
    {'p' : """
    if x == 1:
        out = out + 1.0/p
    else :
        out = out -1.0/(1. - p)
    """}),

dist_ufuncs('beta',
          '',
          'x : float, alpha : float, beta :float',
          'x >= 0.0 and x <= 1.0 and alpha > 0.0 and beta > 0.0',
          'out = out + gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(x) + (beta-1)*log(1-x)',
          {'x'      : 'out = (alpha - 1)/x - (beta - 1)/(1 - x)',
          'alpha'   : 'out = log(x) - psi(alpha) + psi(alpha+beta)',
          'beta'    : 'out = log(1 - x) - psi(beta) + psi(alpha+beta)'}),

dist_ufuncs('binomial','' ,
    'x :int, n : int, p : float',
    'x >= 0 and n >= x and p >= 0.0 and p <= 1.0',
    'out = out + x*log(p) + (n-x)*log(1-p) + factln(n)-factln(x)-factln(n-x)',
    {'p' : 'out = x/p - (n-x)/(1-p)'}),

dist_ufuncs('betabin',
    ' ',
    'x : int, alpha : float, beta :float, n : int',
    'x >= 0 and alpha > 0.0 and beta > 0.0 and n >= x',
    'out = out + gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta)+ gammaln(n+1)- gammaln(x+1)- gammaln(n-x +1) + gammaln(alpha+x)+ gammaln(n+beta-x)- gammaln(beta+alpha+n)',
    {'alpha' : 'out = psi(alpha+beta)-psi(alpha)+ psi(alpha+x) -psi(alpha+beta+n)',
     'beta'  : 'out = psi(alpha+beta)+ psi(n + beta - x)- psi(alpha+beta+n)'}),

dist_ufuncs('cauchy',
    '',
    'x : float, alpha :float , beta : float',
    ' beta > 0.0',
    'out = out + -log(beta) -  log( 1 + ((x-alpha) / beta) ** 2 )',
    {'value' :'out = - 2 * (x - alpha)/(beta**2 + (x - alpha)**2)',
    'alpha' : 'out = 2 * (x - alpha)/(beta**2 + (x-alpha)**2)',
    'beta'  : 'out = -1.0/beta + 2 * (x-alpha)**2/(beta**3 *(1 +(x-alpha)**2/beta**2))'}),
dist_ufuncs('gamma',
    '',
    'x : float, alpha : float, beta :float',
    'x >= 0.0 and alpha > 0.0 and beta > 0.0',
    """
    out = out + -gammaln(alpha) + alpha*log(beta) - beta*x 
    if alpha != 1.0: 
        out = out + (alpha - 1.0)*log(x)""",
    {'value': 'out = (alpha - 1)/x - beta',
     'alpha': 'out = -psi(alpha) + log(x) + log(beta)',
     'beta' : 'out = -x + alpha/beta'}),
dist_ufuncs('normal',
     '',
     'x : float, mu : float, tau :float',
     'tau > 0.',
     'out = out - 0.5 * tau * (x-mu)**2 + 0.5*log(0.5*tau/pi)',
     {'x'   : 'out  = -(x - mu) * tau',
      'mu'  : 'out = (x - mu) * tau',
      'tau' : 'out = 1.0 / (2 * tau) - .5 * (x - mu)**2'})
]
includes = ['special']
path = os.path.join(os.path.dirname( os.path.abspath(os.path.realpath( __file__ ))), 'pymc/dist_ufuncs.pyx')

ug.generate_ufuncs(path, ufunc_definitions,  includes )


