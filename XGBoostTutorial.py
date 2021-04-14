# XGBoost tutorial https://www.datacamp.com/community/tutorials/xgboost-in-python
# XGBoost (Extreme Gradient Boosting) - combines a set of weak learners and delivers improved
# prediction accuracy.


import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# This is importing an example dataset (Boston Housing Dataset)
from sklearn.datasets import load_boston
boston = load_boston()
# print(boston.keys()) # Checks for its keys
# print(boston.data.shape) # Checking the size of the dataset
# print(boston.feature_names) # Returning the feature names/column names
# print(boston.DESCR) # Printing the description of the dataset

# This is where we begin using pandas
# We first import pandas and call DataFrame() function passing in the data
# To label the names of the columns we use .columns and assign boston.feature_names
data = pd.DataFrame(boston.data)
data.columns = boston.feature_names

# Looking at the top 5 rows of the dataframe
data.head()

# There is no column called PRICE in the DataFrame -> except boston.target so we append the name
data['PRICE'] = boston.target

# .info() gives us useful information about the data
data.info()

#  Gives a summary of stats of columns which are continuous in nature & not categorical
data.describe()
## Note that if XGBoost is used on a dataset with categorical features ->
## -> consider applying some encoding (one-hot encoding) to such features before training

# We are building this model using Trees as base learners (default)
# Root Mean Squared error is the square root of the mean of the squared differences
# between the actual and predicted values.

# Separate the target variable and rest of the variables using .iloc to subset the data
X, y = data.iloc[:,:-1],data.iloc[:,-1]

# We convert the dataset to an optimized data structure called Dmatrix supported by XGBoost
# and gives it acclaimed performance and efficiency gains.
data_dmatrix = xgb.DMatrix(data=X,label=y)

# There are several tuning parameters for XGBoost https://xgboost.readthedocs.io/en/latest/parameter.html#general-parameters
# learning_rate: step size shrinkage used to prevent overfitting. Range is [0,1]
# max_depth: determines how deeply each tree is allowed to grow during any boosting round.
# subsample: percentage of samples used per tree. Low value can lead to underfitting.
# colsample_bytree: percentage of features used per tree. High value can lead to overfitting.
# n_estimators: number of trees you want to build.
# objective: determines the loss function to be used like reg:linear for regression problems, reg:logistic for classification problems with only decision, binary:logistic for classification problems with probability.
# XGBoost also supports regularization parameters to penalize models as they become more complex and reduce them to simple (parsimonious) models.
#
# gamma: controls whether a given node will split based on the expected reduction in loss after the split. A higher value leads to fewer splits. Supported only for tree-based learners.
# alpha: L1 regularization on leaf weights. A large value leads to more regularization.
# lambda: L2 regularization on leaf weights and is smoother than L1 regularization.

# Creating the train and test set for cross-validation of the results
from sklearn.model_selection import train_test_split

# Test size is 20% of the data, to maintain reproducibility of results randomstate is assigned
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

# Instantiating an XGBoost regressor object (for classification we would use XGBClassifier())
xg_reg = xgb.XGBRegressor(objective ='reg:linear', colsample_bytree = 0.3, learning_rate = 0.1,
                          max_depth = 5, alpha = 10, n_estimators = 10)

# Fitting the regressor to the training set
xg_reg.fit(X_train,y_train)

# Making predictions
preds = xg_reg.predict(X_test)

# Computing the RMSE
rmse = np.sqrt(mean_squared_error(y_test, preds))

# Printing the result
print("RMSE: %f" % (rmse))


# K-fold cross validation where all entries in original training dataset are used for training and validation
# XGBoost supports k-fold with cv() method with nfolds specification

# Creating a hyper-parameter dictionary params to hold all hyperparameters and values as key values pairs
# Excludes the n_estimators from hyper parameter dictionary because we use num_boost_rounds instead
params = {"objective":"reg:linear",'colsample_bytree': 0.3,'learning_rate': 0.1,
          'max_depth': 5, 'alpha': 10}

# Using the parameters to build a 3-fold cross validation model by invoking XGBoost's cv()
# storing results in cv_results DataFrame -> contains train and test RMSE metrics for each boosting round
cv_results = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,num_boost_round=50,
                    early_stopping_rounds=10,metrics="rmse", as_pandas=True, seed=123)

print(cv_results.head())
print((cv_results["test-rmse-mean"]).tail(1))
# Price prediction has reduced compared to the previous as its 4 per 10000$

# Visualizing Boosting Trees and Feature Importance
# XGBoost has a plot_tree() function that visualizes - once the model is trained and passed
# with the number of trees you want to plot
xg_reg = xgb.train(params=params, dtrain=data_dmatrix, num_boost_round=10)

# Plotting the first tree with the matplotlib
import matplotlib.pyplot as plt
# This is visually doing it using graphviz
xgb.plot_tree(xg_reg,num_trees=0)
plt.rcParams['figure.figsize'] = [50, 10]
plt.show()

# This is another way of visualizing as a bar graph
xgb.plot_importance(xg_reg)
plt.rcParams['figure.figsize'] = [5, 5]
plt.show()
# This shows RM as the highest importance score among all the features

