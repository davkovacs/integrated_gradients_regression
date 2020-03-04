"""Integrated gradients for NNs and plotting of relevant figures"""

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tqdm
import sys
sys.path.append("/home/cdt1906/Documents/cdt/mphil/MiniProject_2/")
from proj2_base import *
import matplotlib.gridspec as gridspec

generate_IGs = True  # True if you want to recalculate the integrated gradients and save to a file
generate_hist = True  # True if you want to generate a histogram of attributions for each feature
generate_bar_chart = True  # True if you want to generate a bar chart to compare the feature importance \
                           # for shallow and deep networks
generate_cor_plot = True  # True if you want to generate the plot of attributions vs correlation coefficients

dataset = "concrete"  # "boston" or "concrete"
arch = "shallow"  # "deep" or "shallow" the architecture of the network
width = 128  # 44 or 86 for deep and 128 or 256 for shallow

# define global variables for the datasets
if dataset == "boston":
    n_feature = 13
    model_name = "bostnet_" + arch
    length = 1400
    attributes = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', \
                      'TAX', 'PTRATIO', 'B1000', 'LSTAT']
elif dataset == "concrete":
    n_feature = 8
    model_name = "concnet_" + arch
    length = 8000
    attributes = ["Cement", "Slag", "Ash", "Water", "Plasticizer", "Coarse aggr.", "Fine aggr.", "Age"]

if generate_IGs:
    model_list = []
    if dataset == "boston":
        dat = load_bost_preprocessed()  # load the preprocessed dataset
        IGs = np.empty((50, 127, 13))  # will contain the IGs for the 50 models, on 127 test examples and 13 features
        model_dir = "/home/cdt1906/Documents/cdt/mphil/MiniProject_2/boston/" + arch + "/"
    elif dataset == "concrete":
        dat = load_concrete_preprocessed()  # load the preprocessed dataset
        IGs = np.empty((50, 258, 8))  # will contain the IGs for the 50 models, on 258 test examples and 8 features
        model_dir = "/home/cdt1906/Documents/cdt/mphil/MiniProject_2/concrete/" + arch + "/"
    j1 = 0  # dummy variable of the model number
    for j3, rs in enumerate(tqdm.tqdm([1, 3, 6, 10, 15, 21, 28, 36, 45, 55])):  # random seed for train-test split
        if dataset == "boston":
            X = dat.drop('Price', axis=1).to_numpy()  # features
            Y = dat['Price'].to_numpy()  # target values
        elif dataset == "concrete":
            X = dat.drop('csMPa', axis=1).to_numpy()  # features
            Y = dat['csMPa'].to_numpy()  # target values
        x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(X, Y, test_size=0.25, random_state=rs)
        x_test = torch.from_numpy(x_test_np)  # convert training data to torch tensor
        for j5 in range(5):  # load the five models for each train-test split
            if dataset == "boston":
                model = torch.load(model_dir + model_name + "_" + str(width) + "_" + str(0.002) + "_" \
                                       + str(length) + "_" + str(j3+1) + "_" + str(j5) + ".pt")
            elif dataset == "concrete":
                model = torch.load(model_dir + model_name + "_" + str(width) + "_" \
                                       + str(length) + "_" + str(j3+1) + "_" + str(j5) + ".pt")
            baseline = torch.zeros([1, n_feature], dtype=torch.double)  # define the baseline as the 0 vector
            gradients = torch.zeros([1, n_feature], dtype=torch.double)  # vector to store the gradients
            steps = 50  # number of steps used to discretise the integral as a Riemann sum
            IG = np.empty((len(x_test_np), n_feature))  # to store the IGs for the current model
            for j in range(0, len(x_test_np)):  # iterate over all test inputs
                # create the straight line path from baseline to input
                scaled_inputs = [baseline + i / steps * (x_test[j] - baseline) for i in range(0, steps + 1)]
                for inp in scaled_inputs:  # iterate over the straight line path
                    inp.requires_grad = True
                    output = model(inp)  # predict the output using the model
                    model.zero_grad()
                    output.backward()  # calculate the gradients
                    gradients += inp.grad  # sum the gradients to approximate the integral
                avg_grads = gradients / steps
                IG[j] = ((x_test[j] - baseline) * avg_grads).numpy()  # calculate the final attribution
            IGs[j1] = IG
            j1 += 1
    np.save(model_name + str(width) + "_IG.npy", IGs)  # save the attriubutions to a binary file

# set up some parameters for plotting
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
plt.rc('text', usetex=True, color='k')
plt.rc('font', family='serif')
plt.rc('font', size=34)

if generate_hist:
    IGs = np.load(model_name + str(width) + "_IG.npy")
    mean_med_abs_attr = np.average(np.median(np.abs(IGs), axis=1), axis=0)  # mean median absolute attributions
    std_med_abs_attr = np.std(np.median(np.abs(IGs), axis=1), axis=0)  # std of median absolute attributions
    np.save(model_name + str(width) + "mean_med_abs_attr.npy", mean_med_abs_attr)
    np.save(model_name + str(width) + "std_med_abs_attr.npy", std_med_abs_attr)
    IGs = IGs.transpose(2, 0, 1).reshape(n_feature, -1)  # reshape the IGs
    median_abs_attr = np.median(np.abs(IGs), axis=1)  # median absolute attribution for all of the models
    np.save(model_name + str(width) + "med_abs_attr.npy", median_abs_attr)
    # remove the largest and smallest 1% attribution for each feature to make plots look nicer
    if dataset == "boston":
        IGs_normalised = np.empty((n_feature, int(np.floor(0.98 * len(IGs[0])) + 1)))
    elif dataset == "concrete":
        IGs_normalised = np.empty((n_feature, int(np.floor(0.98 * len(IGs[0])))))
    for i in range(n_feature):
        l = IGs[i]
        remove = int(np.floor(0.01 * len(IGs[0])))
        argmax = np.argsort(l)[-1 * remove :]
        l = np.delete(l, argmax)
        argmin = np.argsort(l)[: remove]
        l = np.delete(l, argmin)
        IGs_normalised[i] = l
    # create the histogram (not included in the final paper)
    fig = plt.figure()
    if dataset == "boston":
        x = 3
        y = 5
    if dataset == "concrete":
        x = 2
        y = 4
    gs = gridspec.GridSpec(x, y, figure=fig)
    fig.set_size_inches(22, 14)
    plt.tight_layout(pad=2.2, h_pad=None, w_pad=None, rect=None)
    axs = []
    for i in range(x):
        for j in range(y):
            if dataset == "boston":
                if 5 * i + j < 10:
                    axs.append(fig.add_subplot(gs[i, j : j + 1]))
                    axs[5 * i + j].hist(IGs_normalised[5 * i + j], bins=20)
                    axs[5 * i + j].set_xlabel(attributes[5 * i + j])
                elif 5 * i + j < 13:
                    axs.append(fig.add_subplot(gs[i, j + 1 : j + 2]))
                    axs[5 * i + j].hist(IGs_normalised[5 * i + j], bins=20)
                    axs[5 * i + j].set_xlabel(attributes[5 * i + j])
            elif dataset == "concrete":
                    axs.append(fig.add_subplot(gs[i, j]))
                    axs[4 * i + j].hist(IGs_normalised[4 * i + j], bins=20)
                    axs[4 * i + j].set_xlabel(attributes[4 * i + j])
    fig.show()
    fig.savefig(model_name + str(width) + "_IG_hist.png")

if generate_bar_chart:
    # load the mean median absolute attributions and their errors
    if dataset == "boston":
        shal = np.load("bostnet_shallow128mean_med_abs_attr.npy")
        shal_std = np.load("bostnet_shallow128std_med_abs_attr.npy")
        dee = np.load("bostnet_deep44mean_med_abs_attr.npy")
        dee_std = np.load("bostnet_deep44std_med_abs_attr.npy")
    elif dataset == "concrete":
        shal = np.load("concnet_shallow128mean_med_abs_attr.npy")
        shal_std = np.load("concnet_shallow128std_med_abs_attr.npy")
        dee = np.load("concnet_deep44mean_med_abs_attr.npy")
        dee_std = np.load("concnet_deep44std_med_abs_attr.npy")
    # sort the data to make plot look nicer
    shal_std = [x for _,x in sorted(zip(shal, shal_std), reverse=True)]
    attributes = [x for _,x in sorted(zip(shal, attributes), reverse=True)]
    dee_std = [x for _,x in sorted(zip(shal, dee_std), reverse=True)]
    dee = [x for _,x in sorted(zip(shal, dee), reverse=True)]
    shal = sorted(shal, reverse=True)
    # create the bar chart with error bars
    index = np.arange(n_feature)
    bar_widths = 0.35
    fig, ax = plt.subplots()
    fig.set_size_inches(24, 18)
    shal_ch = ax.bar(index, shal, bar_widths, yerr=shal_std, error_kw=dict(lw=3.5, capsize=12, capthick=5), label="Shallow")
    deep_ch = ax.bar(index + bar_widths, dee, bar_widths, yerr=dee_std, error_kw=dict(lw=3.5, capsize=12, capthick=5), label="Deep")
    ax.set_ylabel("Mean Median Absolute Attribution")
    ax.set_xticks(index + bar_widths / 2)
    ax.set_xticklabels(attributes, rotation=30)
    # ax.set_title(dataset)
    ax.legend(prop={'size': 70})
    plt.tight_layout()
    fig.savefig(dataset + "mean_med_attr.png")
    fig.show()

if generate_cor_plot:
    # correlations were calculated using the seaborn statistical package
    if dataset == "boston":
        correlations = [-0.39, 0.36, -0.48, 0.18, -0.43, 0.70, -0.38, 0.25, -0.38, -0.47, -0.51, 0.33, -0.74]
        attr_shallow = np.load("bostnet_shallow128mean_med_abs_attr.npy")
        attr_deep = np.load("bostnet_deep44mean_med_abs_attr.npy")
    elif dataset == "concrete":
        correlations = [0.50, 0.13, -0.11, -0.29, 0.37, -0.16, -0.17, 0.33]
        attr_shallow = np.load("concnet_shallow128mean_med_abs_attr.npy")
        attr_deep = np.load("concnet_deep44mean_med_abs_attr.npy")
    # print out the correlation coefficients between the feature importance and the statistical correlations
    print(np.corrcoef(np.abs(correlations), attr_shallow))
    print(np.corrcoef(np.abs(correlations), attr_deep))
    # make a simple scatter plot to illustrate the lack of correlation
    fig = plt.figure()
    fig.set_size_inches(12, 9)
    ax = fig.gca()
    plt.tight_layout(pad=1.6, h_pad=None, w_pad=None, rect=None)
    ax.scatter(np.abs(correlations), attr_shallow, label='Shallow', marker='x', lw=3, s=294, color="b")
    ax.scatter(np.abs(correlations), attr_deep, label='Deep', marker='o', lw=3, s=222, color="r")
    ax.set_xlabel('Absolute Correlation Coefficient')
    ax.set_ylabel('Mean Median Absolute Attribution')
    ax.set_xticks([0.1, 0.3, 0.5, 0.7])
    ax.legend(fontsize=42)
    plt.tight_layout()
    plt.savefig(dataset + "correlations.png")
    plt.show()
