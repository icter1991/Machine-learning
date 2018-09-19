import numpy
from sklearn import preprocessing, cross_validation, neighbors
import pandas

data_frame = pandas.read_csv('breast-cancer-wisconsin.data.txt')
data_frame.replace('?', -99999, inplace=True)
data_frame.drop(['id'], 1, inplace=True)

X = numpy.array(data_frame.drop(['class'], 1))
y = numpy.array(data_frame['class'])

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)