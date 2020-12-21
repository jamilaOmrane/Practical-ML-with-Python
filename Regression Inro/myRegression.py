import pandas as pd
import quandl
import math
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 

# my key API mkqsuCzpnNzDnDbEKhyB
quandl.ApiConfig.api_key = "mkqsuCzpnNzDnDbEKhyB"
df = quandl.get('WIKI/GOOGL')
#print(df.head()) 
df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. Open'] - df['Adj. Close']) / df['Adj. Close'] * 100
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100

df = df[['Adj. Close','HL_PCT', 'PCT_change','Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out =  int(math.ceil(0.01*len(df))) # this will return the number of days out # using data that came 10 days ago to predict tooday
df ['label'] = df[forecast_col].shift(-forecast_out)

#print(forecast_out)
df.dropna(inplace = True)
#print(df.head())

X = np.array(df.drop(['label'],1))
y = np.array(df['label'])

X= preprocessing.scale(X)

#print(len(X), len(y))

X_tain, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR(kernel='poly')
clf.fit(X_tain, y_train)
acc = clf.score(X_test, y_test)
print(acc)
