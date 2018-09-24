import pandas
import quandl
import math
import datetime
import numpy
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

data_frame = quandl.get('WIKI/GOOGL')

data_frame = data_frame[['Adj. Open', 'Adj. High', 'Adj. Close', 'Adj. Volume']]
data_frame['HL_PCT'] = (data_frame['Adj. High'] - data_frame['Adj. Close']) / data_frame['Adj. Close'] * 100.0
data_frame['PCT_change'] = (data_frame['Adj. Close'] - data_frame['Adj. Open']) / data_frame['Adj. Open'] * 100.0

data_frame = data_frame[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
data_frame.fillna(-9999, inplace=True)

forecast_out = int(math.ceil(0.1*len(data_frame)))
print(forecast_out)

data_frame['label'] = data_frame[forecast_col].shift(-forecast_out)

X = numpy.array(data_frame.drop(['label'], 1))
X = preprocessing.scale(X)  

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

data_frame.dropna(inplace=True)
y = numpy.array(data_frame['label'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
with open('linearregression.picle', 'wb') as f:
    pickle.dump(clf, f)

pickle_in = open('linearregression.picle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)

data_frame['Forecast'] = numpy.nan

last_date = data_frame.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    data_frame.loc[next_date] = [numpy.nan for _ in range(len(data_frame.columns)-1)] + [i]

data_frame['Adj. Close'].plot()
data_frame['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
