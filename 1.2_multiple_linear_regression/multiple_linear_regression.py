# multiple linear regression ( ALL IN )

# importing libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

# handling of categorical data (this should be done before splitting of dataset)
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
colTrans = ColumnTransformer([("Country", OneHotEncoder(), [3])], remainder = 'passthrough')
X = colTrans.fit_transform(X)

#Avoiding the dummy variable trap
X = X[:,1:]

# splitting into test and training dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# feature scaling , handled automatically

# fitting multiple regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# predicting values
y_pred = regressor.predict(X_test)

# here it is difficult to visualize because for representing 5 variables simulataenously, we need 5 dimensions


# multiple linear regression ( BACKWARD ELIMINATION )

# building the optimal backaward elimination
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
# to prevent type error
X_opt = np.array(X_opt, dtype = float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# removing col with index 2 because it is having highest p-value
X_opt = X[:, [0, 1, 3, 4, 5]]
# to prevent type error
X_opt = np.array(X_opt, dtype = float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# removing col with index 1 because it is having highest p-value now
X_opt = X[:, [0, 3, 4, 5]]
# to prevent type error
X_opt = np.array(X_opt, dtype = float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# removing col with index 2 because it is having highest p-value now
X_opt = X[:, [0, 3, 5]]
# to prevent type error
X_opt = np.array(X_opt, dtype = float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

# removing col with index 2 because it is having highest p-value now
X_opt = X[:, [0, 3]]
# to prevent type error
X_opt = np.array(X_opt, dtype = float)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()