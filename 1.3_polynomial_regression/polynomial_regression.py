# polynomial regression

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

# feature scaling, handled automatically

# fitting linear regression to the dataset (for comparison purposes)
from sklearn.linear_model import LinearRegression
linReg = LinearRegression()
linReg.fit(X, y)

# fitting polynomial regressoin to the dataset
from sklearn.preprocessing import PolynomialFeatures
polyReg = PolynomialFeatures(degree = 3)
X_poly = polyReg.fit_transform(X)

linReg2 = LinearRegression()
linReg2.fit(X_poly, y)

# visualizing linear regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg.predict(X), color = 'blue')
plt.title('Truth or Bluff (Linear Regression results)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()

# visualising polynomial regression results
plt.scatter(X, y, color = 'red')
plt.plot(X, linReg2.predict(polyReg.fit_transform(X)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression results)')
plt.xlabel('Position levels')
plt.ylabel('Salary')
plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'blue')
plt.title('Truth or Bluff (Polynomial Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()