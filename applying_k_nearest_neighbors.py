import numpy
from sklearn import preprocessing, model_selection, neighbors
import pandas

df = pandas.read_csv("breast-cancer-wisconsin.data.txt")
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

X = numpy.array(df.drop(['class'], 1))
y = numpy.array(df['class'])

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)

print(accuracy)

example_measures = numpy.array([4, 2, 1, 1, 1, 2, 3, 2, 1])

example_measures = example_measures.reshape(1, -1)

predistion = clf.predict(example_measures)

print(predistion)
