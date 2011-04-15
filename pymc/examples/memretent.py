# Memory Retention Model
#
# Converted from an example from the book "A Course in Bayesian
# Graphical Modeling for Cognitive Science" by Michael D. Lee
# (mdlee_AT_uci.edu) and Eric-Jan Wagenmakers
# (ej.wagenmakers_AT_gmail.com)
# http://users.fmg.uva.nl/ewagenmakers/BayesCourse/BayesBook.html
#
# The original model was implemented in matlab, R and WinBUGS.
#
# Model converted to python and pymc by Thomas V. Wiecki
# (thomas.wiecki_AT_gmail.com) (c) 2010, GPLv3
#
# Description: Modeled after the following experiment: Subjects are
# given 18 items to remember. At different times (t) they are tested
# on how many items they remembered. With increasing delay, subjects
# remember less and less. The rate of memory retention follows an
# exponential with two parameters: alpha (slope) and beta
# (baseline). For more details, read chapter 8.3 of the above
# mentioned book.

import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

###########
# Data
###########

# Items to be remembered
n = 18
# How many items were actually remembered by each subject. None is a
# missing data point. Note that for subject 4 we have no data at all.
data_array = [np.array([18,18,16,13,9,6,4,4,4,None]),
              np.array([17,13, 9, 6,4,4,4,4,4,None]),
              np.array([14,10, 6, 4,4,4,4,4,4,None])]
              #np.array([None,None,None,None,None,None,None,None,None,None])]

masked_data = [np.ma.masked_equal(x, value=None) for x in data_array]

# Times points for test
t = np.array([1,2,4,7,12,21,35,59,99,200])

num_subjs = len(data_array)
num_trials = t.shape[0]

# Priors For Group Distributions
alpha_mu = pm.Uniform('alpha_mu', value=.5, lower=0.01, upper=1, plot=True)
alpha_sigma = pm.Uniform('alpha_sigma', value=1., lower=0, upper=100, plot=True)
beta_mu = pm.Uniform('beta_mu', value=.5, lower=0, upper=1, plot=True)
beta_sigma = pm.Uniform('beta_sigma', value=1., lower=0.01, upper=100, plot=True)

# Containers for individual distributions
alpha = np.empty(num_subjs, dtype=object)
beta = np.empty(num_subjs, dtype=object)
retention = np.empty(num_subjs, dtype=object)
k = np.empty(num_subjs, dtype=object)

for i in range(num_subjs):
    alpha[i] = pm.TruncatedNormal('alpha_%i'%i, mu=alpha_mu, tau=alpha_sigma**-2, a=0, b=1, plot=True)
    beta[i] = pm.TruncatedNormal('beta_%i'%i, mu=beta_mu, tau=beta_sigma**-2, a=0, b=1, plot=True)

    @pm.deterministic
    def retention_i(alpha=alpha[i], beta=beta[i], t=t):
        theta = np.exp(-alpha*t) + beta # Exponential decay in memory retention
        # Boundary conditions
        theta[theta>=1] = .999
        theta[theta<0] = 0
        return theta

    retention_i.__name__ = 'retention_%i'%i
    retention[i] = retention_i

    k[i] = pm.Binomial('k_%i'%i, value=masked_data[i], p=retention[i], n=n*np.ones(num_trials), observed=True, plot=False)

# PLOTTING FUNCTION
def plot_joint(M):
    import matplotlib.pyplot as plt

    alpha_vars = ['alpha_'+str(i) for i in range(num_subjs)]
    beta_vars = ['beta_'+str(i) for i in range(num_subjs)]
    bins=100

    fig = plt.figure()
    spacing = 0.01
    wdth_cond = 0.1
    x_wdth = 0.7
    y_wdth = 0.75
    lft = 0.05
    lower_spc = .1
    lower = lower_spc+wdth_cond+spacing
    
    ax1 = fig.add_axes([lft, lower, y_wdth, x_wdth])
    # beta
    ax2 = fig.add_axes([lft+y_wdth+spacing, lower, wdth_cond, x_wdth], sharey=ax1)
    # alpha
    ax3 = fig.add_axes([lft, lower_spc, y_wdth, wdth_cond], sharex=ax1)

    for i, (alpha, beta) in enumerate(zip(alpha_vars, beta_vars)):
        alpha_trace = M.trace(alpha)()
        beta_trace = M.trace(beta)()

        alpha_trace_part = alpha_trace[::50]
        beta_trace_part = beta_trace[::50]

        ax1.plot(alpha_trace_part, beta_trace_part, ['x','o','.', '+'][i], label="Subject: "+str(i+1))
    
        histo = np.histogram(beta_trace, range=(0,1), bins=bins)[0]
        histo = histo/np.float(np.max(histo))
        ax2.plot(histo, np.linspace(0,1,bins))

        histo = np.histogram(alpha_trace, range=(0,1), bins=bins)[0]
        histo = histo/np.float(np.max(histo))
        ax3.plot(np.linspace(0,1,bins), histo)

    ax3.axes.set_xlabel(r'$\alpha$')
    ax2.axes.set_ylabel(r'$\beta$')
    ax1.legend(loc=0)
    ax1.axes.set_xlim(0,1)
    ax1.axes.set_ylim(0,1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    ax2.yaxis.label_position = 'right'
    for tick in ax2.yaxis.get_major_ticks():
        tick.label1On = False
        tick.label2On = True

    plt.show()
    
def main(plot=True):
    model_vars = [alpha, beta, retention, alpha_mu, alpha_sigma, beta_mu, beta_sigma, k]
    M = pm.MCMC(model_vars)
    #M.use_step_method(pm.AdaptiveMetropolis, alpha)
    #M.use_step_method(pm.AdaptiveMetropolis, beta)
    M.sample(20000, burn=15000)
    if plot:
        plot_joint(M)
	pm.Matplot.plot(M)
    plt.show()
    return M

if __name__ == '__main__':
    main()
