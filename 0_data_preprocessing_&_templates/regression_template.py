# regression template

# importing libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values    # just to treat it as a matrix
y = dataset.iloc[:, 2].values

# no need of splitting into test and training dataset, because of small dataset
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# feature scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit.fit_transform(y_train)'''

# fitting the regression model to the dataset
# create your regressor here

# predicting a new result
y_pred = regressor.predict([[6.5]])

# visualizing the regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (Regression results)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()

# Visualising the regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Regression results)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()