"""A hyperparameter optimisation and training for Concrete dataset using the deepo network"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tqdm
import os
os.chdir("/home/cdt1906/Documents/cdt/mphil/MiniProject_2/")
from proj2_base import *

data = load_concrete_preprocessed()  # load preprocessed data

errors = np.zeros((2, 10, 1, 5))
for j1, width in enumerate([44, 86]):  # number of nodes of the hidden layers
    print("The width is " + str(width))
    for j2, rs in enumerate(tqdm.tqdm([1, 3, 6, 10, 15, 21, 28, 36, 45, 55])):  # random seeds for train - test split
        X = data.drop('csMPa', axis=1).to_numpy()
        Y = data['csMPa'].to_numpy()
        x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(X, Y, test_size=0.25, random_state=rs)
        x_train = torch.from_numpy(x_train_np)
        x_test = torch.from_numpy(x_test_np)
        y_train = torch.from_numpy(y_train_np).reshape([772, 1])
        y_test = torch.from_numpy(y_test_np)
        for j3, length in enumerate([8000]):  # length of training 200, 300, 600, 1000, 1400, 2100, 3000, 4000, 8000
            for j4 in range(5):  # to optimise 5 times the same net with SGD
                concnet_deep = Net_deep(n_feature=8, width_hidden=width)  # define the model
                concnet_deep = concnet_deep.double()
                optimizer = torch.optim.SGD(concnet_deep.parameters(), lr=0.0005)  # SGD for optimisation
                loss_func = torch.nn.MSELoss()  # mean squared loss for regression
                # training
                for i in range(length):
                    prediction = concnet_deep(x_train)
                    loss = loss_func(prediction,y_train)
                    optimizer.zero_grad()  # clear gradients
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()

                # save the trained model
                torch.save(concnet_deep, "./concnet_deep_" + str(width) + "_" \
                           + str(length) + "_" + str(j2+1) + "_" + str(j4) + ".pt")
                # testing
                y_pred = concnet_deep(x_test)
                y_pred_np = y_pred.detach().numpy()
                score = r2_score(y_test_np, y_pred_np)
                rmse = np.sqrt(mean_squared_error(y_test_np,y_pred_np))
                errors[j1][j2][j3][j4] = rmse

# save the test errors
np.save("conc_deep_errors2.npy",errors)
file = open("conc_deep_errors2.txt", "w")
file.write(str(errors))
file.close()


