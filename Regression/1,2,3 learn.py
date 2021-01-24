# Mengimpor library yang diperlukan
import numpy as np
import pandas as pd

# Import data ke python
dataset = pd.read_csv('D:\Mastering Machine Learning\Dataset\Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

# Memproses data yang hilang (missing)
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.nan, strategy = 'mean')
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
 
# Encoding data kategori dan variabel independen
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
transformer = ColumnTransformer(
        [('Negara', OneHotEncoder(), [0])],
        remainder='passthrough')
X = np.array(transformer.fit_transform(X), dtype=np.float)
 
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Membagi menjadi training set dan test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

