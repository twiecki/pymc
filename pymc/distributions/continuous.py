"""
pymc.distributions

A collection of common probability distributions for stochastic
nodes in PyMC.

"""

from dist_math import *

__all__ = ['Uniform', 'Flat', 'Normal', 'Beta','Exponential', 'T', 'Cauchy', 'Gamma', 'Bound', 'Tpos']

@quickclass
def Uniform(lower=0, upper=1):
    """
    Continuous uniform log-likelihood.

    .. math::
        f(x \mid lower, upper) = \frac{1}{upper-lower}

    Parameters
    ----------
    lower : float
        Lower limit (defaults to 0)
    upper : float
        Upper limit (defaults to 1)
    """
    support = 'continuous'

    def logp(value):
        return bound(
                -log(upper - lower),
                lower <= value, value <= upper)

    mean = (upper + lower)/2.
    median = mean

    default = mean


    logp.__doc__ = """
        Uniform log-likelihood with parameters lower={0} and upper={1}.

        Parameters
        ----------
        value : float
            :math:`lower \leq x \leq upper`
        """

    return locals()

@quickclass
def Flat():
    """
    Uninformative log-likelihood that returns 0 regardless of
    the passed value.
    """
    support = 'continuous'

    def logp(value):
        return zeros_like(value)

    logp.__doc__ = """
        Uninformative log-likelihood that returns 0 regardless of
        the passed value.
        """

    median = 0

    return locals()

@quickclass
def Normal(mu=0.0, tau=1.0):
    """
    Normal log-likelihood.

    .. math::
        f(x \mid \mu, \tau) = \sqrt{\frac{\tau}{2\pi}} \exp\left\{ -\frac{\tau}{2} (x-\mu)^2 \right\}

    Parameters
    ----------
    mu : float
        Mean of the distribution.
    tau : float
        Precision of the distribution, which corresponds to
        :math:`1/\sigma^2` (tau > 0).

    .. note::
    - :math:`E(X) = \mu`
    - :math:`Var(X) = 1/\tau`

    """
    support = 'continuous'

    def logp(value):

        return bound(
                (-tau * (value-mu)**2 + log(tau/pi/2.))/2.,
                tau > 0)

    mean = mu
    variance = 1./tau

    median = mean
    mode = mean

    logp.__doc__ = """
        Normal log-likelihood with paraemters mu={0} and tau={1}.

        Parameters
        ----------
        value : float
            Input data.
        """.format(mu, tau)

    return locals()

@quickclass
def Beta(alpha, beta):
    """
    Beta log-likelihood. The conjugate prior for the parameter
    :math:`p` of the binomial distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\Gamma(\alpha + \beta)}{\Gamma(\alpha) \Gamma(\beta)} x^{\alpha - 1} (1 - x)^{\beta - 1}

    Parameters
    ----------
    alpha : float
        alpha > 0
    beta : float
        beta > 0

    .. note::
    - :math:`E(X)=\frac{\alpha}{\alpha+\beta}`
    - :math:`Var(X)=\frac{\alpha \beta}{(\alpha+\beta)^2(\alpha+\beta+1)}`

    """
    support = 'continuous'
    def logp(value):

        return bound(
                  gammaln(alpha+beta) - gammaln(alpha) - gammaln(beta) + logpow(value, alpha -1) + logpow(1-value, beta-1),
                  0 <= value, value <= 1,
                  alpha > 0,
                  beta > 0)

    mean = alpha/(alpha + beta)
    variance = alpha*beta/(
            (alpha + beta)**2 * (alpha + beta +1))

    logp.__doc__ = """
    Beta log-likelihood with parameters alpha={0} and beta={1}.

    Parameters
    ----------
    value : float
        0 < x < 1
    """.format(alpha, beta)

    return locals()

@quickclass
def Exponential(lam):
    """
    Exponential distribution

    Parameters
    ----------
    lam : float
        lam > 0
        rate or inverse scale
    """

    support = 'continuous'
    mean = 1./lam
    median = mean * log(2)
    mode = 0

    variance = lam**-2

    def logp(value):
        return log(lam) - lam*value

    return locals()


@quickclass
def T(nu, mu=0, lam=1):
    """
    Non-central Student's T log-likelihood.

    Describes a normal variable whose precision is gamma distributed. If
    only nu parameter is passed, this specifies a standard (central)
    Student's T.

    .. math::
        f(x|\mu,\lambda,\nu) = \frac{\Gamma(\frac{\nu +
        1}{2})}{\Gamma(\frac{\nu}{2})}
        \left(\frac{\lambda}{\pi\nu}\right)^{\frac{1}{2}}
        \left[1+\frac{\lambda(x-\mu)^2}{\nu}\right]^{-\frac{\nu+1}{2}}

    Parameters
    ----------
    nu : int
        Degrees of freedom
    mu : float
        Location parameter (defaults to 0)
    lam : float
        Scale parameter (defaults to 1)
    """
    support = 'continuous'

    def logp(value):
        return bound(
                  gammaln((nu+1.0)/2.0) + .5 * log(lam / (nu * pi)) - gammaln(nu/2.0) - (nu+1.0)/2.0 * log(1.0 + lam *(value - mu)**2/nu),
                  lam > 0,
                  nu > 0)

    mean = mu
    variance = switch((nu >2)*1, nu/(nu - 2) / lam , inf)


    logp.__doc__ = """
        Student's T log-likelihood with paramters nu={0}, mu={1} and lam={2}.

        Parameters
        ----------
        value : float
            Input data
        """

    return locals()

@quickclass
def Cauchy(alpha, beta):
    """
    Cauchy log-likelihood. The Cauchy distribution is also known as the
    Lorentz or the Breit-Wigner distribution.

    .. math::
        f(x \mid \alpha, \beta) = \frac{1}{\pi \beta [1 + (\frac{x-\alpha}{\beta})^2]}

    Parameters
    ----------
    alpha : float
        Location parameter
    beta : float
        Scale parameter > 0

    .. note::
    Mode and median are at alpha.

    """
    support = 'continuous'
    def logp(value):
        return bound(
                  -log(pi) -log(beta) - log( 1 + ((value-alpha) / beta) ** 2 ),
                  beta > 0)

    logp.__doc__ = """
        Cauchy log-likelihood with parameters alpha={0} and beta={0}.

        Parameters
        ----------
        value : float
            Input data.
        """.format(alpha, beta)
    return locals()

@quickclass
def Gamma(alpha, beta):
    """
    Gamma log-likelihood.

    Represents the sum of alpha exponentially distributed random variables, each
    of which has mean beta.

    .. math::
        f(x \mid \alpha, \beta) = \frac{\beta^{\alpha}x^{\alpha-1}e^{-\beta x}}{\Gamma(\alpha)}

    Parameters
    ----------
    x : float
        math:`x \ge 0`
    alpha : float
        Shape parameter (alpha > 0).
    beta : float
        Rate parameter (beta > 0).

    .. note::
    - :math:`E(X) = \frac{\alpha}{\beta}`
    - :math:`Var(X) = \frac{\alpha}{\beta^2}`

    """
    support = 'continuous'

    mean = alpha/beta
    median = max((alpha -1)/beta, 0)
    variance = alpha / beta**2

    def logp(value):
        return bound(
                -gammaln(alpha) + logpow(beta, alpha) - beta*value + logpow(value, alpha-1),

                value >= 0,
                alpha > 0,
                beta > 0)

    logp.__doc__ = """
        Gamma log-likelihood with paramters alpha={0} and beta={1}.

        Parameters
        ----------
        x : float
            math:`x \ge 0`
        """.format(alpha, beta)

    return locals()


@quickclass
def Bound(dist, lower = -inf, upper = inf):
    support = dist.support
    def logp(value):
        return bound(
                dist.logp(value),

                lower <= value, value <= upper)

    return locals()

@quickclass
def Wfpt(a, v, z):
    support = 'continuous'

    def logp(value):
        p, k = T.dscalars(['k', 'p'])
        tt = value/a**2
        kl = T.switch(np.pi*tt*err<1, # if
                      T.max(T.stack(T.sqrt(-2*T.log(np.pi*tt*err)/(np.pi**2*tt)), 1/(np.pi*T.sqrt(tt))), axis=0), # then
                      1/(np.pi*T.sqrt(tt))) # else

        ks = T.switch(2*T.sqrt(2*np.pi*tt)*err<1, # if error threshold is set low enough
                      T.max(T.stack(2+T.sqrt(-2*tt*T.log(2*T.sqrt(2*np.pi*tt)*err)), # bound
                                    T.sqrt(tt)+1), axis=0), # ensure boundary conditions are met
                      2) # if error threshold was set too high -> minimal kappa for that case
        Ks = T.ceil(ks)
        Kl = T.ceil(kl)

        ps = lambda k, z, tt: T.sum((z+2*k)*T.exp(-((z+2*k)**2)/2/tt))
        pl = lambda k, z, tt: T.sum(k*T.exp(-(k**2)*(np.pi**2)*tt/2)*T.sin(k*np.pi*z))

        ps_range = T.arange(-T.floor((Ks-1)/2), T.ceil((Ks-1)/2.), 1)
        pl_range = T.arange(1, Kl, 1)

        ps_terms, tmp = theano.map(ps, ps_range, non_sequences=(z, tt))
        pl_terms, tmp = theano.map(pl, pl_range, non_sequences=(z, tt))

        ps_out = ps_terms.sum() * np.pi
        pl_out = pl_terms.sum() / T.sqrt(2*np.pi*T.pow(tt,3))

        p_out = T.switch(ks < kl, ps_out, pl_out)

        out = p_out*T.exp(-v*a*z -(T.pow(v,2))*value/2.)/(T.pow(a,2))

        return log(out)

    return locals()

def Tpos(nu, mu=0, lam=1):
    """
    Student-t distribution bounded at 0
    see T
    """
    return Bound(T(nu, mu, lam), 0)

