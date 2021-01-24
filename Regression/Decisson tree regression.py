# Mengimpor library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Mengimpor dataset
dataset = pd.read_csv('D:\Mastering Machine Learning\Dataset\Posisi_gaji.csv')
X = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2:3].values

# Membuat model regresi decision tree
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X,y)

# Memprediksi hasil model
y_pred = regressor.predict([[6.5]])

# Visualisasi hasil regresi decision tree
X_grid = np.arange(min(X),max(X),0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color="red")
plt.plot(X_grid,regressor.predict(X_grid),color="blue")
plt.title("Sesuai atau tidak decission tree regression")
plt.xlabel("level posisi")
plt.ylabel("gaji")
plt.show()