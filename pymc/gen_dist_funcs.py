import ufunc_gen as ug
import os

import pydevd
pydevd.set_pm_excepthook()

def dist_ufuncs(name, docstring, signature,constraints, like_calc, grad_calcs = {}):
    vars = [ug.NumpyVarDefinition(*var.split(':')) for var in signature.split(',')]
    
    return [ug.UFuncDefinition(name +'_like', vars, 'float', constraints, like_calc, docstring = docstring, array_out=False),
            [ug.UFuncDefinition(name +'_' + var + 'grad', vars, 'float', constraints, calc, array_out=True) for var, calc in grad_calcs.iteritems()]]


ufunc_definitions = [
dist_ufuncs('bernoulli',
"""Bernoulli log-likelihood

    The Bernoulli distribution describes the probability of successes (x=1) and
    failures (x=0).

    .. math::  f(x \mid p) = p^{x} (1-p)^{1-x}

    :Parameters:
      - `x` : Series of successes (1) and failures (0). :math:`x=0,1`
      - `p` : Probability of success. :math:`0 < p < 1`.

    :Example:
       >>> bernoulli_like([0,1,0,1], .4)
       -2.8542325496673584

    .. note::
      - :math:`E(x)= p`
      - :math:`Var(x)= p(1-p)`

    """,
    'x : int, p : float',
    'x >= 0 and x <= 1 and p >= 0. and p <= 1.',
    """
    if x = 1:
        out = out + p
    else :
        out = out + 1. - p
    """,
    {'p' : """
    if x = 1:
        out = out + 1./p
    else :
        out = out -1./(1. - p)
    """}),

dist_ufuncs('beta',
          """
    beta_like(x, alpha, beta)

    Beta log-likelihood. The conjugate prior for the parameter
    :math:`p` of the binomial distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}

    :Parameters:
      - `x` : 0 < x < 1
      - `alpha` : alpha > 0
      - `beta` : beta > 0

    :Example:
      >>> beta_like(.4,1,2)
      0.18232160806655884

    .. note::
      - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
      - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """,
          'x : float, alpha : float, beta :float',
          'x >= 0 and x <= 1 and alpha > 0 and beta > 0',
          'out = out + gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + (alpha- 1)*log(x) + (beta-1)*log(1-x)',
          {'x'      : 'out = (alpha - 1)/xv - (beta - 1)/(1 - x)',
          'alpha'   : 'out = log(x) - psi(alpha) + psi(alpha+beta)',
          'beta'    : 'out = log(1 - x) - psi(beta) + psi(alpha+beta)'}),

dist_ufuncs('binomial',"""
    binomial_like(x, n, p)

    Binomial log-likelihood.  The discrete probability distribution of the
    number of successes in a sequence of n independent yes/no experiments,
    each of which yields success with probability p.

    .. math::
        f(x \mid n, p) = \frac{n!}{x!(n-x)!} p^x (1-p)^{n-x}

    :Parameters:
      - `x` : [int] Number of successes, > 0.
      - `n` : [int] Number of Bernoulli trials, > x.
      - `p` : Probability of success in each trial, :math:`p \in [0,1]`.

    .. note::
       - :math:`E(X)=np`
       - :math:`Var(X)=np(1-p)`
    """,
    'x :int, n : int, p : float',
    'x >= 0 and n >= x and p >= 0. and p <= 1.',
    'out = out + x*log(p) + (n-x)*log(1-p) + factln(n)-factln(x)-factln(n-x)',
    {'p' : 'out = x/p - (n-x)/(1-p)'}),

dist_ufuncs('betabin',
    R"""
    betabin_like(x, alpha, beta)

    Beta-binomial log-likelihood. Equivalent to binomial random
    variables with probabilities drawn from a
    :math:`\texttt{Beta}(\alpha,\beta)` distribution.

    .. math::
        f(x \mid \alpha, \beta, n) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha)} \frac{\Gamma(n+1)}{\Gamma(x+1)\Gamma(n-x+1)} \frac{\Gamma(\alpha + x)\Gamma(n+\beta-x)}{\Gamma(\alpha+\beta+n)}

    :Parameters:
      - `x` : x=0,1,\ldots,n
      - `alpha` : alpha > 0
      - `beta` : beta > 0
      - `n` : n=x,x+1,\ldots

    :Example:
      >>> betabin_like(3,1,1,10)
      -2.3978952727989

    .. note::
      - :math:`E(X)=n\frac{\alpha}{\alpha+\beta}`
      - :math:`Var(X)=n\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """,
    'x : int, alpha : float, beta :float, n : int',
    'x >= 0 and alpha > 0 and beta > 0 and n >= x',
    'out = out + gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta)+ gammaln(n+1)- gammaln(x+1)- gammaln(n-x +1) + gammaln(alpha+x)+ gammaln(n+beta-x)- gammaln(beta+alpha+n)',
    {'alpha' : 'out = psi(alpha+beta)-psi(alpha)+ psi(alpha+x) -psi(alpha+beta+n)',
     'beta'  : 'out = psi(alpha+beta)+ psi(n + beta - x)- psi(alpha+beta+n)'}),

dist_ufuncs('cauchy',
    R"""
    cauchy_like(x, alpha, beta)

    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    :Parameters:
      - `alpha` : Location parameter.
      - `beta` : Scale parameter > 0.

    .. note::
       - Mode and median are at alpha.
    """,
    'x : float, alpha :float , beta : float',
    ' beta > 0',
    'out = out + -log(beta) -  log( 1 + ((x-alpha) / beta) ** 2 )',
    {'value' :'out = - 2 * (x - alpha)/(beta**2 + (x - alpha)**2)',
    'alpha' : 'out = 2 * (x - alpha)/(beta**2 + (x-alpha)**2)',
    'beta'  : 'out = -1/beta + 2 * (x-alpha)**2/(beta**3 *(1 +(x-alpha)**2/beta**2))'}),
dist_ufuncs('gamma',
    R"""
    gamma_like(x, alpha, beta)

    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables, each
    of which has mean beta.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    :Parameters:
      - `x` : math:`x \ge 0`
      - `alpha` : Shape parameter (alpha > 0).
      - `beta` : Scale parameter (beta > 0).

    .. note::
       - :math:`E(X) = \frac{\alpha}{\beta}`
       - :math:`Var(X) = \frac{\alpha}{\beta^2}`

    """,
    'x : float, alpha : float, beta :float',
    'x >= 0 and alpha > 0 and beta > 0',
    """
    out = out + -gammaln(alpha) + alpha*log(beta) - beta*x 
    if alpha != 1.0: 
        out = out + (alpha - 1.0)*log(x))""",
    {'value': 'out = (alpha - 1)/x - beta',
     'alpha': 'out = -psi(alpha) + log(x) + log(beta)',
     'beta' : 'out = -x + alpha/beta'})]
includes = ['special']

ug.generate_ufuncs(os.path.join(os.path.dirname( os.path.realpath( __file__ )),'dist_ufuncs.pxy'  ), ufunc_definitions,  includes )


