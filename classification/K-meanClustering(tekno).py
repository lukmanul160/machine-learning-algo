import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns

#masukkan dataset
df = pd.read_csv('D:\Mastering Machine Learning\Dataset\movies_metadata.csv')

df= df[['budget','revenue','runtime','vote_count','vote_average','title']]

#check kosong data bisa diisi mean atau dihapus
df.isna().sum()
df.dropna(inplace=True)

#check distribusi
df['vote_count'].describe()

#mengukur panjang dataset check di terminal
len(df)

df2 = df[df['vote_count']>30]

#check data yang kosong/NUll
df2.isna().sum

#membuat scaling data <[0,1]> atau [-1,1]
from sklearn import preprocessing
sc = preprocessing.MinMaxScaler().fit_transform(df2.drop('title',axis=1))
df3 = pd.DataFrame(sc,index=df2.index,columns=df2.columns[:-1])

#hitung dulu score
#scr=[]
#for i in range(1,20):
#    score = KMeans(n_clusters=i).fit(df3).score(df3)
#    print(score)
#    scr.append(score)
#    
#plt.plot(scr)

kmeans = KMeans(n_clusters=5)
kmeans.fit(df3)

kmeans.labels_

df3['cluster'] = kmeans.labels_


#plt.hist(df3['cluster'])



