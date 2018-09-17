from statistics import mean
import numpy
import matplotlib.pyplot as plt
from matplotlib import style

style.use('ggplot')
x_axis = numpy.array([1, 2, 3, 4, 5, 6], dtype=numpy.float64)
y_axis = numpy.array([5, 4, 1, 5, 9, 7], dtype=numpy.float64)


def line_slope_and_intercept(xs, ys):
    """Returns slope of best fit line as m and  also intercept of ys ad xs as b.

    xs: co-ordinates on X-axis; int
    ys: co-ordinates on Y-axis; int
    :returns m:int, b:int
    """

    m = ((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - (mean(xs**2)))
    b = mean(ys) - m*mean(xs)

    return m, b


# y = m*x+b

slope, intercept = line_slope_and_intercept(x_axis, y_axis)

regression_line = []

for x in x_axis:
    regression_line.append(slope*x+intercept)

# regression_line = [(slope*x)+intercept for x in x_axis]

co_ordinate_x = 7
predict_co_ordinate_y = (slope*co_ordinate_x)+intercept

plt.scatter(x_axis, y_axis)
plt.scatter(co_ordinate_x, predict_co_ordinate_y, color='green')
plt.plot(x_axis, regression_line)
plt.show()
