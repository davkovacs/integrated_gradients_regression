"""Script to create the learning curve for shallow network on Boston housing dataset"""

import numpy as np
import matplotlib.pyplot as plt

# read in the rms errors
errors = np.load('shallow_errors.npy')
errors2 = np.load('shallow_errors2.npy')
errors3 = np.load('shallow_errors3.npy')
# calculate average error and std for different architectures and training lengths
average_errors = np.average(errors, axis=(2, 4))
average_errors2 = np.average(errors2, axis=(2, 4))
average_errors3 = np.average(errors3, axis=(2, 4))
std_errors = np.std(errors, axis=(2,4))
std_errors2 = np.std(errors2, axis=(2,4))
std_errors3 = np.std(errors3, axis=(2,4))
average_errors_128 = np.concatenate((average_errors[0][1], average_errors2[0][0]))
std_errors_128 = np.concatenate((std_errors[0][1], std_errors2[0][0]))
average_errors_256 = np.concatenate((average_errors[1][1], average_errors2[1][0]))
std_errors_256 = np.concatenate((std_errors[1][1], std_errors2[1][0]))

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
plt.rc('font', size=34)

x = np.array([300, 600, 1000, 1400])
logsp = np.around(np.logspace(np.amin(np.log2(np.subtract(average_errors_128, std_errors_128))), \
                              np.amax(np.log2(np.add(average_errors_128, std_errors_128))), 4, base=2), 2)
fig = plt.figure()
fig.set_size_inches(12,9)
ax = fig.gca()
plt.tight_layout(pad=1.6, h_pad=None, w_pad=None, rect=None)
ax.errorbar(x, average_errors_128, yerr=std_errors_128, label='Small', marker='x', \
            ms=16, lw=3, color="b", mew=6)
ax.errorbar(x, average_errors_256, yerr=std_errors_256, label='Large', marker='o', \
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
plt.savefig('bost_lc_shallow.png')
plt.show()