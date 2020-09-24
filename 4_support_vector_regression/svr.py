# svr

# importing libraries and dataset
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing datasets
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values    # just to treat it as a matrix
y = dataset.iloc[:, 2].values.reshape(-1, 1)

# no need of splitting into test and training dataset, because of small dataset
'''from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)'''

# feature scaling, svr do not handles it automatically
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

# fitting svr to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,y)

# predicting new result
# sc_X.transform() is applied because we feature scaled all the X values
# transform method needs array as input
# sc_y.inverse_transform() to get the value back in original format
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

# visualizing the svr results
plt.scatter(X, y, color = 'red')
plt.plot(X, regressor.predict(X), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.xlabel('Salray')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()