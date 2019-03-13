import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt

# 目标函数
def real_func(x):
    return np.sin(2*np.pi*x)

# 拟合函数
"""
ploy1d([a]) => a
ploy1d([a,b]) => ax + b
ploy1d([a,b,c]) => ax^2 + bx + c
ploy1d([a,b,c,d]) => ax^3 + bx^2 + cx +d
"""
def fit_func(p,x):
    f = np.poly1d(p)
    return f(x)

def residuals_func(p,x,y):
    err = fit_func(p,x) - y
    return err

# regularization
def residuals_func_regularization(p, x, y, r):
    err = fit_func(p, x) - y
    err = np.append(err, np.sqrt(0.5*r*np.square(p))) # L2范数作为正则化项
    return err


def fitting(x,y,M):
    """
    n 为 多项式的次数
    """
    # 随机初始化多项式参数
    p_init = np.random.rand(M+1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func, p_init, args=(x, y))
    #print('Fitting Parameters:', p_lsq[0])


    return p_lsq

def fitting_regularization(x,y,M, r):
    """
    n 为 多项式的次数
    """
    # 随机初始化多项式参数
    p_init = np.random.rand(M+1)
    # 最小二乘法
    p_lsq = leastsq(residuals_func_regularization, p_init, args=(x, y, r))
    #print('Fitting Parameters:', p_lsq[0])


    return p_lsq


if __name__ == "__main__":
    # data number
    N = 10
    # data provided
    x_provided = np.linspace(0, 1, N)
    # 加上正态分布噪音的目标函数的值
    y_provided = 0.2 * np.random.randn(N) + real_func(x_provided)

    # real data
    x_real = np.linspace(0, 1, 1000)
    y_real = real_func(x_real)


    M = 9
    regularization = 0.0001

    fig1 = plt.figure()
    for i in range(M):
        title = 'M=' + str(i+1)
        solution = fitting(x_provided, y_provided, i)
        ax = fig1.add_subplot(3,3,i+1,title=title)
        ax.scatter(x_provided,y_provided,label='noise')
        ax.plot(x_real, y_real, label='real')
        ax.plot(x_real, fit_func(solution[0], x_real), label='fitted')
        ax.legend()

    fig2 = plt.figure()
    for i in range(M):
        title = 'M=' + str(i + 1)
        solution = fitting_regularization(x_provided, y_provided, i, regularization)
        ax = fig2.add_subplot(3, 3, i + 1, title=title)
        ax.scatter(x_provided, y_provided, label='noise')
        ax.plot(x_real, y_real, label='real')
        ax.plot(x_real, fit_func(solution[0], x_real), label='fitted')
        ax.legend()

    plt.show()

