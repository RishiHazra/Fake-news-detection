# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 19:11:33 2018

@author: Rishi
"""

import pandas as pd
import nltk
from nltk.stem import SnowballStemmer
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize
import seaborn as sb
import numpy as np
import re
from sklearn.preprocessing import MultiLabelBinarizer

features=['ID','Label','statement','subject','speaker','job-title','state info','affliation','barely true','false','half true','mostly true','pants on fire','context']


        
print("Loading data...")

with open('test+train.tsv',encoding='utf8') as tsvfile:
     train_test_data = pd.read_csv(tsvfile, delimiter='\t',header=None)
     tsvfile.close()
     
with open('test.tsv',encoding='utf8') as tsvfile:
    test_data=pd.read_csv(tsvfile, delimiter='\t', header=None)
    tsvfile.close()
    
with open('train.tsv',encoding='utf8') as tsvfile:
    train_data=pd.read_csv(tsvfile, delimiter='\t', header=None)
    tsvfile.close()


#data observation
def data_obs():
    print("training dataset size:")
    print(train_data.shape)
    print(train_data.head(10))

    #below dataset were used for testing and validation purposes
    print(test_data.shape)
    print(test_data.head(10))
    
    

#distribution of classes for prediction
def create_distribution(dataFile):
    
    return sb.countplot(x=dataFile.iloc[:,1], data=dataFile, palette='hls')
    

#by calling below we can see that training, test and valid data seems to be fairry evenly distributed between the classes
create_distribution(train_test_data)
create_distribution(test_data)
create_distribution(train_data)


#data integrity check (missing label values)
#none of the datasets contains missing values therefore no cleaning required
def data_qualityCheck():
    
    print("Checking data qualitites...")
    train_data.isnull().sum()
    train_data.info()
        
    print("check finished.")

    #below datasets were used to 
    test_data.isnull().sum()
    test_data.info()


#run the below function call to see the quality check results
data_qualityCheck()




#========================= Processing the metadata ===============================#



# one hot encoding of the categorical features
one_hot_encode=np.zeros([len(train_data),1])
mlb= MultiLabelBinarizer()

for i,cat_i in enumerate(np.array([3,4,5,6,7,13])):
    l=[]
    for word in train_data.iloc[:,3:14][cat_i]:
        a=re.split(',',str(word))
        l.append(a)
    print(mlb.fit_transform(l).shape)
    one_hot_encode= np.concatenate((one_hot_encode,mlb.fit_transform(l)),axis=1)
class_names=list(set(train_data[1]))

#(10240, 143)
#(10240, 2911)
#(10240, 1308)
#(10240, 86)
#(10240, 24)
#(10240, 4460)

X_data = np.concatenate((train_data.iloc[:,8:12],one_hot_encode),axis=1)

del i,l,a,word,cat_i

def data_return():
    return X_data, class_names



#========================= Processing the text data ===============================#



eng_stemmer = SnowballStemmer('english')
stopwords = set(nltk.corpus.stopwords.words('english'))

#Stemming
def stem_tokens(tokens, stemmer):
    stemmed = []
    for token in tokens:
        stemmed.append(stemmer.stem(token))
    return stemmed

#process the data
def process_data(data,exclude_stopword=True,stem=True):
    tokens = [w.lower() for w in data]
    tokens_stemmed = tokens
    tokens_stemmed = stem_tokens(tokens, eng_stemmer)
    tokens_stemmed = [w for w in tokens_stemmed if w not in stopwords ]
    return tokens_stemmed


porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]





