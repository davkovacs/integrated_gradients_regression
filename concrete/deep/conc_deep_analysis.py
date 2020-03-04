"""Script to create the learning curve for deep network on concrete dataset"""

import numpy as np
import matplotlib.pyplot as plt

# read in the rms errors
errors = np.load('conc_deep_errors.npy')
errors2 = np.load('conc_deep_errors2.npy')
# calculate average error for different architectures and training lengths
average_errors = np.append(np.average(errors, axis=(1, 3)), np.average(errors2, axis=(1,3)), axis=1)
# calculate  std for different architectures and training lengths
std_errors = np.append(np.std(errors, axis=(1, 3)), np.std(errors2, axis=(1,3)), axis=1)
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
plt.rc('font', size=28)

x = np.array([200, 300, 600, 1000, 1400, 2100, 3000, 4000, 8000])
logsp = np.around(np.logspace(np.amin(np.log2(np.subtract(average_errors[0], std_errors[0]))), \
                              np.amax(np.log2(np.add(average_errors[1], std_errors[1]))), 4, base=2), 2)
fig = plt.figure()
fig.set_size_inches(12,9)
ax = fig.gca()
plt.tight_layout(pad=1.6, h_pad=None, w_pad=None, rect=None)
ax.errorbar(x, average_errors[0], yerr=std_errors[0], label='Small', marker='x', \
            ms=16, lw=3, color="b", mew=6)
ax.errorbar(x, average_errors[1], yerr=std_errors[1], label='Large', marker='o', \
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
plt.savefig('conc_lc_deep.png')
plt.show()
