import pandas as pd
import csv
from fuzzywuzzy import fuzz
import textdistance
import nltk

FILENAME1 = 'task1a.csv'
file1 = open(FILENAME1 ,'w+')
writer1 = csv.writer(file1)
writer1.writerow(['idAbt', 'idBuy'])

abt = pd.read_csv("abt_small.csv", encoding='ISO-8859-1')
buy = pd.read_csv("buy_small.csv", encoding='ISO-8859-1')
pairs = []
for i in abt.index:
    levs = []
    jaro = []
    dups = []
    length = []
    count = 0
    i_name = abt['name'][i]
    tokeni = nltk.word_tokenize(i_name)
    code = tokeni[-1]
    for j in buy.index:
        j_name = buy['name'][j]
        tokenj = nltk.word_tokenize(j_name)
        for word in tokenj:
            lev = textdistance.levenshtein.normalized_similarity(code, word)
            levs.append(lev)
            jar = textdistance.jaro_winkler(code, word)
            jaro.append(jar)
        levenshtein = max(levs)
        jaro_winkler = max(jaro)
        Ratio = fuzz.ratio(i_name.lower(), j_name.lower())
        Partial_Ratio = fuzz.partial_ratio(i_name.lower(), j_name.lower())
        Token_Sort_Ratio = fuzz.token_sort_ratio(i_name.lower(), j_name.lower())
        Token_Set_Ratio = fuzz.token_set_ratio(i_name.lower(), j_name.lower())
        minimum = min([Ratio, Partial_Ratio, Token_Sort_Ratio, Token_Set_Ratio])
        condition1 = (Token_Set_Ratio >= 80 and jaro_winkler >= 0.94)
        condition2 = (minimum >= 70 and jaro_winkler >= 0.87)
        condition3 = (minimum >= 80 and jaro_winkler >= 0.75)
        condition4 = (jaro_winkler >= 0.95 and Token_Set_Ratio >= Token_Sort_Ratio and Token_Set_Ratio > 45)
        condition5 = (jaro_winkler == 1 and levenshtein == 1 and Token_Set_Ratio >= Token_Sort_Ratio and Token_Set_Ratio > 50)
        condition6 = (Token_Set_Ratio >= 90 and jaro_winkler >= 0.73 and jaro_winkler * 100 +  Token_Set_Ratio >= 150)
        condition7 = (Token_Set_Ratio >= 85 and jaro_winkler >= 0.6 and jaro_winkler * 100 + Token_Set_Ratio >= 160)
        
        
        if (condition1 or condition2 or condition3 or condition4 or condition5 or condition6 or condition7):
            pairs.append([jaro_winkler, abt['idABT'][i], buy['idBuy'][j]])
            dups.append([abt['idABT'][i], buy['idBuy'][j]])
            length.append(jaro_winkler)
    
    try:
        count+=1
        index = length.index(max(length))
        writer1.writerow(dups[index])
    except:
        count+=1
    
file1.close()
    