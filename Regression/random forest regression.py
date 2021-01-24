# Mengimpor library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Mengimpor dataset
dataset = pd.read_csv('D:\Mastering Machine Learning\Dataset\Posisi_gaji.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

# Membuat model random forest regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()
regressor.fit(X,y)

# Visualisasi hasil random forest regression
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Jujur atau tidak (Random Forest Regression)')
plt.xlabel('Tingkat posisi')
plt.ylabel('Gaji')
plt.show()