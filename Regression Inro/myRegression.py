import pandas as pd
import quandl, math, datetime
import numpy as np
from sklearn import preprocessing, svm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

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

#print(df.head())

X = np.array(df.drop(['label'],1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_lately = X[-forecast_out:]


df.dropna(inplace = True)
y = np.array(df['label'])


#print(len(X), len(y))

X_tain, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

clf = LinearRegression(n_jobs=-1)
#clf = svm.SVR(kernel='poly')
# clf.fit(X_tain, y_train)
# with open('linearRegression.pickle', 'wb') as f:
#     pickle.dump(clf, f)

pickel_in = open ('linearRegression.pickle', 'rb')
clf = pickle.load(pickel_in)

acc = clf.score(X_test, y_test)

forecast_set = clf.predict(X_lately)
#print(forecast_set, acc, forecast_out)

df['Forecast'] = np.nan
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix =  last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

















