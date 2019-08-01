The main files are project1.py and project1_b.py 
The first one is for network data and the second one is for housing data.
All the regressions are in the regression.py file but they are called in the two files project1.py and project1_b.py.
You can change the regression type used by typing the name in the k_fold_x_validation function which takes that x, y , k , type of regression and and one variable that can change the variable of regression (for example depth for random forest, ...).
The k_fold_x_validation function returns all the MSEs for all the test data in k_fold_cross_validation along with the MSE for training data and the regressor.