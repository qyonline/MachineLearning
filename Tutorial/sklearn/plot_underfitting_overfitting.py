print(__doc__)

from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt


# 目标函数
def true_func(x):
    return np.cos(1.5 * np.pi * x)


"""
# 当我们设置相同的seed，每次生成的随机数相同。
# 如果不设置seed，则每次会生成不同的随机数
Example1:
seed(0)
a=randn(1)
seed(0)
b=randn(1)
a==b

Example2:
seed(0)
a=randn(1)
b=randn(1)
seed(0)
c=randn(1)
d=randn(1)
a==c
b==d
"""
np.random.seed(0)

n_samples = 30
degrees = [1, 4, 15]

x = np.sort(np.random.rand(n_samples))
y = true_func(x) + np.random.randn(n_samples) * 0.1

subplot_num = len(degrees)
plt.figure(figsize=(14, 5))

for i in range(subplot_num):
    ax = plt.subplot(1, subplot_num, i + 1)
    #plt.setp(ax, xticks=(), yticks=())

    polynomial_features = PolynomialFeatures(degree=degrees[i], include_bias=False)

    linear_regression = LinearRegression()

    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])

    pipeline.fit(x[:, np.newaxis], y)

    # Evaluate the models using cross-validation
    scores = cross_val_score(pipeline, x[:, np.newaxis], y,
                             scoring="neg_mean_squared_error", cv=10)

    x_test = np.linspace(0, 1, 100)
    plt.plot(x_test, pipeline.predict(x_test[:, np.newaxis]), label="Model")
    plt.plot(x_test, true_func(x_test), label="True function")
    plt.scatter(x, y, edgecolor='b', s=20, label="Samples")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim((0,1))
    plt.ylim((-2,2))
    plt.legend(loc="best")
    plt.title("Degree {}\nMSE = {:.2e}(+/- {:.2e})".format(degrees[i], -scores.mean(), scores.std()))

plt.show()

