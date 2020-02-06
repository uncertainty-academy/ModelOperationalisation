import numpy as np
import matplotlib.pyplot as plt

num_classes = 3
dimensions = 2
points_per_class = 100


def generate_spiral_data(num_classes=3, dimensions=2, points_per_class=100):
    X = np.zeros((points_per_class * num_classes, dimensions), dtype='float32')
    y = np.zeros(points_per_class * num_classes, dtype='uint8')

    for y_value in xrange(num_classes):
        ix = range(points_per_class * y_value, points_per_class * (y_value + 1))

        radius = np.linspace(0.0, 1, points_per_class)
        theta = np.linspace(y_value * 4, (y_value + 1) * 4, points_per_class) + np.random.randn(points_per_class) * 0.2

        X[ix] = np.column_stack([radius * np.sin(theta), radius * np.cos(theta)])
        y[ix] = y_value

    return X, y


def plot_data(X, y):
    fig = plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
    plt.xlim([-1, 1])
    plt.ylim([-1, 1])
    return fig


# X, y = generate_spiral_data(num_classes, dimensions, points_per_class)
# fig = plot_data(X, y)
# plt.show()
