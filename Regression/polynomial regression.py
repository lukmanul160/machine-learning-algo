# Mengimpor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv('D:\Mastering Machine Learning\Dataset\Posisi_gaji.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values
 
# Fitting Linear Regression ke dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
 
# Fitting Polynomial Regression ke dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)  ## nantinya degree diganti menjadi 4
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)
 
# Visualisasi hasil regresi sederhana
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title('Sesuai atau tidak (Linear Regression)')
plt.xlabel('Level posisi')
plt.ylabel('Gaji')
plt.show()
 
# Visualisasi hasil regresi polynomial
plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg_2.predict(X_poly), color = 'blue')
plt.title('Sesuai atau tidak (Polynomial Regression)')
plt.xlabel('Level posisi')
plt.ylabel('Gaji')
plt.show()
 
# Memprediksi hasil dengan regresi sederhana
lin_reg.predict([[6.5]])
# Memprediksi hasil dengan regresi polynomial
lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))