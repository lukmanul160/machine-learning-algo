# Mengimpor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Mengimpor dataset
dataset = pd.read_csv('optimisasi_retail.csv', header = None )
jumlah_baris = dataset.shape[0]
jumlah_kolom = dataset.shape[1]
transaksi = [[str(dataset.values[i,j]) for j in range(0, jumlah_kolom) if str(dataset.values[i,j])!='nan']
                                              for i in range(0, jumlah_baris)]
 
# Membuat variabel items yang berisikan transaksi yang tidak terduplikasi  
items = list()
for t in transaksi:
    for x in t:
        if(not x in items):
            items.append(x)
           
# Membuat list bernama eclat yang merupakan pasangan 2 item, dengan nilai support awal nol (0)
eclat = list()
for i in range(0, len(items)):
    for j in range(i+1, len(items)):
        eclat.append([(items[i],items[j]),0])
       
# Menghitung nilai support untuk setiap pasangan 2 item
for p in eclat:
    for t in transaksi:
        if(p[0][0] in t) and (p[0][1] in t):
            p[1] += 1
    p[1] = p[1]/len(transaksi)
   
# Merubah hasil eclat ke dalam data frame dan mengurutkannya berdasarkan nilai support tertinggi
Hasil_eclat = pd.DataFrame(eclat, columns = ['rule','support']).sort_values(by = 'support', ascending = False)