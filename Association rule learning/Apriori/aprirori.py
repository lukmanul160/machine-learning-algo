# Mengimpor library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
 
# Mengimpor dataset
dataset = pd.read_csv('optimisasi_retail.csv', header = None)
 
# Preprocessing
transaksi = []
for i in range(0, len(dataset)):
    transaksi.append([str(item) for item in dataset.values[i,:] if str(item)!='nan'])
 
# Melatih algoritma apriori ke dataset
from apyori import apriori
batasan = apriori(transaksi, min_support = 0.004, min_confidence = 0.25, min_lift = 4, min_length = 2)
 
# Visualisasi hasil apriori versi 1
hasil = list(batasan)
analisis_hasil = []
for i in range(0, len(hasil)):
    analisis_hasil.append('RULE:\t{}\nSUPP:\t{}\nCONF:\t{}\nLIFT:\t{}\n'.format(list(hasil[i][0]), str(hasil[i][1]), str(hasil[i][2][0][2]), str(hasil[i][2][0][3])))
