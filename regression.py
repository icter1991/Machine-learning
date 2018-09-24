from statistics import mean
import numpy
import matplotlib.pyplot as plt
from matplotlib import style
import random



def line_slope_and_intercept(xs, ys):
    """Returns slope of best fit line as m and  also intercept of ys ad xs as b.

    xs: co-ordinates on X-axis; list of ints
    ys: co-ordinates on Y-axis; list of ints
    :returns m:int, b:int
    """

    m = ((mean(xs)*mean(ys)) - mean(xs*ys)) / ((mean(xs)**2) - (mean(xs**2)))
    b = mean(ys) - m*mean(xs)

    return m, b


def mean_squared_error(ys_original, ys_line):
    """Returns the average squared difference between the estimated value and what is estimated.

    ys_original: co-ordinates on Y-axis; list of ints
    ys_line: regression line; list of ints
    :returns squared error
    """

    squared_error = sum((ys_line-ys_original)**2)
    return squared_error


def coefficient_of_determination(ys_original, ys_line):
    """The proportion of variance in the dependent variable that is predictable from independent variables.

    ys_original: co-ordinates on Y-axis; list of ints
    ys_line: regression line; list of ints
    :returns float
    """
    y_mean_line = []
    for y in ys_original:          # generator: y_mean_line = [mean(ys_original) for y in ys_original]
        y_mean_line.append(mean(ys_original))

    squared_error_regression = mean_squared_error(ys_original, ys_line)
    squared_error_y_mean = mean_squared_error(ys_original, y_mean_line)

    return 1 - (squared_error_regression / squared_error_y_mean)


def create_random_data_set(how_many, variance, step=2, correlation=False):
    """Creating random data set for training purpose.

    :returns two lists
    """
    val = 1
    ys = []
    for i in range(how_many):
        y = val + random.randrange(-variance, variance)
        ys.append(y)
        if correlation and correlation == "pos":
            val += step

        elif correlation and correlation == "neg":
            val -= step

    xs = [i for i in range(len(ys))]

    return numpy.array(xs, dtype=numpy.float64), numpy.array(ys, dtype=numpy.float64)


def main():

    style.use('ggplot')
    # x_axis = numpy.array([1, 2, 3, 4, 5, 6], dtype=numpy.float64)
    # y_axis = numpy.array([5, 4, 6, 5, 6, 7], dtype=numpy.float64)

    x_axis, y_axis = create_random_data_set(40, 10, 2, correlation='pos')

    slope, intercept = line_slope_and_intercept(x_axis, y_axis)
    regression_line = []

    for x in x_axis:        # generator: regression_line = [(slope*x)+intercept for x in x_axis]
        regression_line.append(slope*x+intercept)

    co_ordinate_x = 7
    predict_co_ordinate_y = (slope*co_ordinate_x)+intercept     # y = m*x+b

    r_squared = coefficient_of_determination(y_axis, regression_line)
    print(r_squared)

    plt.scatter(x_axis, y_axis)
    plt.scatter(co_ordinate_x, predict_co_ordinate_y, s=100, color='green')
    plt.plot(x_axis, regression_line)
    plt.show()


if __name__ == '__main__':
    main()
