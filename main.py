import scipy.io.arff as arff
import numpy as np
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from kfold import KFoldValidation
#First let's get all the headers from the KDD training set

kddFile = open('KDDTrain_20Percent.arff','r', encoding='utf-8')

kddData, kddMeta = arff.loadarff(kddFile)
kddDF = pd.DataFrame(kddData)


features = kddDF.columns
#This is the feature that define whether or not the traffic is normal
targets = set(kddDF['class'].values)

'''
We need to do some cleanup before we can use the data
certain datasets will not be usable because they are 
'''

#Drop duplicates
kddDF.drop_duplicates(keep='first',inplace=True)

#clear nulls
kddDF.dropna()

#Encode the values, simple at the moment
kddDF['protocol_type'] = kddDF['protocol_type'].astype('category')
kddDF['service'] = kddDF['service'].astype('category')
kddDF['flag'] = kddDF['flag'].astype('category')
kddDF['class'] = kddDF['class'].astype('category')
kddDF['land'] = kddDF['land'].astype('category')
kddDF['logged_in'] = kddDF['logged_in'].astype('category')
kddDF['is_host_login'] = kddDF['is_host_login'].astype('category')
kddDF['is_guest_login'] = kddDF['is_guest_login'].astype('category')
_columns = kddDF.select_dtypes(['category']).columns
kddDF[_columns] = kddDF[_columns].apply(lambda x: x.cat.codes)
#however we might need to one hot encode them, we'll see later.

output_features = features[41]
input_features = features[0:41]
y = pd.DataFrame(data=kddDF, columns=[output_features])
x = pd.DataFrame(data=kddDF, columns=input_features)

decision_tree = KFoldValidation(DecisionTreeClassifier(),5)
decision_tree.validate(x,y)