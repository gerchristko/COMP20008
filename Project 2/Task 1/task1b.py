import pandas as pd
import numpy as np
import nltk
import csv

FILENAME1 = 'abt_blocks.csv'
file1 = open(FILENAME1 ,'w+')
writer1 = csv.writer(file1)
writer1.writerow(['block_key', 'product_id'])

FILENAME2 = 'buy_blocks.csv'
file2 = open(FILENAME2 ,'w+')
writer2 = csv.writer(file2)
writer2.writerow(['block_key', 'product_id'])

abt = pd.read_csv("abt.csv", encoding='ISO-8859-1')
buy = pd.read_csv("buy.csv", encoding='ISO-8859-1')

for i in abt.index:
    item = abt['name'][i]
    token = nltk.word_tokenize(item)
    manufacturer = token[0]
    writer1.writerow([manufacturer.lower(), abt['idABT'][i]])
    
for i in buy.index:
    brand = buy['manufacturer'][i]
    token = nltk.word_tokenize(str(brand))
    if len(token) == 1:
        writer2.writerow([str(brand).lower(), buy['idBuy'][i]])
    elif len(token) == 0:
        writer2.writerow([np.nan, buy['idBuy'][i]])
    else:
        writer2.writerow([token[0].lower(), buy['idBuy'][i]])
        
file1.close()
file2.close()