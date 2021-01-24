# Mengimpor library yang diperlukan
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv("D:/Mastering Machine Learning/Dataset/50_Startups.csv")
X = dataset.iloc[:, :-1].values
Tampilkan_X = pd.DataFrame(X) #visualisasi X
y = dataset.iloc[:, 4].values
 
# Encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()
 
# Menghindari jebakan dummy variabel
#X = X[:, 1:]
 
# Membagi data menjadi the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
 
# Membuat model Multiple Linear Regression dari Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 
# Memprediksi hasil Test set
y_pred = regressor.predict(X_test)

#Memilih model multiple regresi yang paling baik dengan metode backward propagation
#import statsmodels.api as sma
#X = sma.add_constant(X)
#import statsmodels.api as sm
#X_opt = X[:, [0, 1, 2, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:, [0, 1, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:, [0, 3, 4, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:, [0, 3, 5]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
#X_opt = X[:, [0, 3]]
#regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
#regressor_OLS.summary()
 
# Memilih model multiple regresi yang paling baik dengan metode backward propagation
import statsmodels.api as sm
def backwardElimination(x, sl):
    numVars = len(x[0])
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        if maxVar > sl:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
SL = 0.05
X_opt = X[:, [0,1,2,3,4,5]]
X_Modeled = backwardElimination(X_opt, SL)

regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()

