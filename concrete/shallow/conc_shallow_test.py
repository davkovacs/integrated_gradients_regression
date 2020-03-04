"""A hyperparameter optimisation and training for Concrete dataset using a shallow network"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tqdm
import os
os.chdir("/home/cdt1906/Documents/cdt/mphil/MiniProject_2/")
from proj2_base import *

data = load_concrete_preprocessed()  # load preprocessed data

errors = np.zeros((2, 10, 4, 5))
errors2 = np.zeros((2, 10, 1, 5))
for j1, width in enumerate([128, 256]):  # number of nodes of the hidden layers
    print("The width is " + str(width))
    for j2, rs in enumerate(tqdm.tqdm([1, 3, 6, 10, 15, 21, 28, 36, 45, 55])):  # random seeds for train - test split
        # Split training and testing
        X = data.drop('csMPa', axis=1).to_numpy()
        Y = data['csMPa'].to_numpy()
        x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(X, Y, test_size=0.25, random_state=rs)
        x_train = torch.from_numpy(x_train_np)
        x_test = torch.from_numpy(x_test_np)
        y_train = torch.from_numpy(y_train_np).reshape([772, 1])
        y_test = torch.from_numpy(y_test_np)
        for j3, length in enumerate([8000]):  # length of training 200, 300, 600, 1000, 1400, 2100, 3000, 4000, 8000
            for j4 in range(5):  # to optimise 5 times the same net with SGD
                concnet_shallow = Net_shallow(n_feature=8, width_hidden=width)  # define the model
                concnet_shallow = concnet_shallow.double()
                optimizer = torch.optim.SGD(concnet_shallow.parameters(), lr=0.002)  # SGD for optimisation
                loss_func = torch.nn.MSELoss()  # mean squared loss for regression
                # training
                for i in range(length):
                    prediction = concnet_shallow(x_train)
                    loss = loss_func(prediction,y_train)
                    optimizer.zero_grad()  # clear gradients
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()

                # save the model
                torch.save(concnet_shallow, "./concnet_shallow_" + str(width) + "_" \
                           + str(length) + "_" + str(j2+1) + "_" + str(j4) + ".pt")
                # testing
                y_pred = concnet_shallow(x_test)
                y_pred_np = y_pred.detach().numpy()
                score = r2_score(y_test_np, y_pred_np)
                rmse = np.sqrt(mean_squared_error(y_test_np,y_pred_np))
                errors2[j1][j2][j3][j4] = rmse

# save the test errors
np.save("conc_shallow_errors5.npy", errors2)
file = open("conc_shallow_errors5.txt", "w")
file.write(str(errors2))
file.close()
