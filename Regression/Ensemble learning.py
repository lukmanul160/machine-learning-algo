import pandas as pd
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC



dataset = pd.read_csv('D:\Mastering Machine Learning\Dataset\heart.csv')
X = dataset.iloc[:,0:13].values
y = dataset.iloc[:,13].values

#split test train
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2 ,random_state=10)

#DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train,y_train)
dt.score(X_train,y_train),dt.score(X_test,y_test)

#RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=10)
rfc.fit(X_train,y_train)
rfc.score(X_train,y_train),rfc.score(X_test,y_test)

#begging classifier
from sklearn.ensemble import BaggingClassifier
bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0)
bg.fit(X_train,y_train)
bg.score(X_train,y_train),bg.score(X_test,y_test)

#AdaBoostClassifier
from sklearn.ensemble import AdaBoostClassifier
adb = AdaBoostClassifier(DecisionTreeClassifier(),n_estimators=10,learning_rate=1)
adb.fit(X_train,y_train)
adb.score(X_train,y_train),adb.score(X_test,y_test)

#VotingClassifier
from sklearn.ensemble import VotingClassifier
lc = LogisticRegression()
svm = SVC(kernel='poly',degree=2)
bg = BaggingClassifier(DecisionTreeClassifier(),max_samples=0.5,max_features=1.0)
rfc = RandomForestClassifier(n_estimators=10)

evc = VotingClassifier(estimators = [('lc',lc),('svm',svm),('bg',bg),('rfc',rfc)],voting='hard')
evc.fit(X_train,y_train)
evc.score(X_train,y_train),evc.score(X_test,y_test)
