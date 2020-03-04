"""A hyperparameter optimisation for Boston dataset using a shallow network"""

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import tqdm
import os
os.chdir("/home/cdt1906/Documents/cdt/mphil/MiniProject_2/")
from proj2_base import *

data = load_bost_preprocessed()  # load preprocessed data

errors = np.zeros((2, 2, 10, 3, 5))
errors2 = np.zeros((2, 1, 10, 1, 5))
errors3 = np.zeros((2, 1, 10, 1, 5))
for j1, width in enumerate([128, 256]):  # number of nodes of the hidden layers
    print("The width is " + str(width))
    for j2, learning_rate in enumerate([0.002]):
        print("The learning rate is " + str(learning_rate))
        for j3, rs in enumerate(tqdm.tqdm([1, 3, 6, 10, 15, 21, 28, 36, 45, 55])):  # random seeds for train-test split
            X = data.drop('Price', axis=1).to_numpy()
            Y = data['Price'].to_numpy()
            x_train_np, x_test_np, y_train_np, y_test_np = train_test_split(X, Y, test_size=0.25, random_state=rs)
            x_train = torch.from_numpy(x_train_np)
            x_test = torch.from_numpy(x_test_np)
            y_train = torch.from_numpy(y_train_np).reshape([379, 1])
            y_test = torch.from_numpy(y_test_np)
            for j4, length in enumerate([200]):  # length of training [200, 300, 600, 1000]
                for j5 in range(5):  # to optimise 5 times the same net with SGD
                    bostnet_shallow = Net_shallow(n_feature=13, width_hidden=width)  # define the model
                    bostnet_shallow = bostnet_shallow.double()
                    optimizer = torch.optim.SGD(bostnet_shallow.parameters(), lr=learning_rate)  # SGD for optimisation
                    loss_func = torch.nn.MSELoss()  # mean squared loss for regression
                    # training
                    for i in range(length):
                        prediction = bostnet_shallow(x_train)
                        loss = loss_func(prediction,y_train)
                        optimizer.zero_grad()  # clear gradients
                        loss.backward()  # backpropagation, compute gradients
                        optimizer.step()

                    # save the model
                    torch.save(bostnet_shallow, "./bostnet_shallow_" + str(width) + "_" + str(learning_rate) + "_" \
                               + str(length) + "_" + str(j3+1) + "_" + str(j5) + ".pt")
                    # testing
                    y_pred = bostnet_shallow(x_test)
                    y_pred_np = y_pred.detach().numpy()
                    score = r2_score(y_test_np, y_pred_np)
                    rmse = np.sqrt(mean_squared_error(y_test_np,y_pred_np))
                    errors3[j1][j2][j3][j4][j5] = rmse
# save the test errors
np.save("shallow_errors3.npy",errors3)
file = open("shallow_errors3.txt", "w")
file.write(str(errors3))
file.close()


