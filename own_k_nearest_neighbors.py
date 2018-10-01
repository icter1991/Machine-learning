import numpy
import matplotlib.pyplot as plt
import warnings
from matplotlib import style
from collections import Counter

style.use("fivethirtyeight")

dataset = {'k': [[1, 2], [2, 3], [3, 1]], 'r': [[6, 5], [7, 7], [8, 6]]}
new_feature = [5, 7]


def k_nearest_neighbours(data, predict, k=3):
    if len(data) >= k:
        warnings.warn("K is set to value less than total voting groups!")

    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = numpy.linalg.norm(numpy.array(features)-numpy.array(predict))
            distances.append([euclidean_distance, group])

    votes = [i[1] for i in sorted(distances)[:k]]
    votes_result = Counter(votes).most_common(1)[0][0]

    return votes_result


result = k_nearest_neighbours(dataset, new_feature)
print(result)

[[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
plt.scatter(new_feature[0], new_feature[1], color=result)
plt.show()
