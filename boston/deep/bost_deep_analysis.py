"""Script to create the learning curve for deep network on Boston housing dataset"""

import numpy as np
import matplotlib.pyplot as plt

# read in the rms errors
errors = np.load('deep_errors3.npy')
average_errors = np.average(errors, axis=(2, 4))  # average error for different architectures and training lengths
std_errors = np.std(errors, axis=(2,4))  # std for different architectures and training lengths
print(average_errors)

# Create the log-log plot with error bars
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['xtick.major.size'] = 5
plt.rcParams['xtick.major.width'] = 1.5
plt.rcParams['xtick.minor.size'] = 2.5
plt.rcParams['xtick.minor.width'] = 1.5
plt.rcParams['ytick.major.size'] = 5
plt.rcParams['ytick.major.width'] = 1.5
plt.rcParams['ytick.minor.size'] = 2.5
plt.rcParams['ytick.minor.width'] = 1.5
plt.rcParams['errorbar.capsize'] = 12.0
plt.rcParams["legend.loc"] = 'best'
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=36)

x = np.array([300, 600, 1000, 1400])
logsp = np.around(np.logspace(np.amin(np.log2(np.subtract(average_errors[0][0], std_errors[0][0]))), \
                              np.amax(np.log2(np.add(average_errors[1][0], std_errors[1][0]))), 4, base=2), 2)
fig = plt.figure()
fig.set_size_inches(12,9)
ax = fig.gca()
plt.tight_layout(pad=1.6, h_pad=None, w_pad=None, rect=None)
ax.errorbar(x, average_errors[0][0], yerr=std_errors[0][0], label='Small', marker='x', \
            ms=16, lw=3, color="b", mew=6)
ax.errorbar(x, average_errors[1][0], yerr=std_errors[1][0], label='Large', marker='o', \
            ms=16, lw=3, color="r", mew=6)
ax.set_xlabel('Number of training iterations')
ax.set_ylabel('RMSE')
ax.set_xscale('log',basex = 2)
ax.set_yscale('log',basey = 2)
ax.set_xticks(x)
ax.set_xticklabels(['$\\mathrm{' + t + '}$' for t in x.astype(str)])
ax.set_yticks(logsp)
ax.set_yticklabels(['$\\mathrm{' + t + '}$' for t in logsp.astype(str)])
ax.legend(loc="lower left", fontsize=34)
plt.savefig('bost_lc_deep.png')
plt.show()