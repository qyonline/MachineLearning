import numpy as np
import matplotlib.pyplot as plt

x = np.arange(-5.0, 5.0, 0.1)

##You can adjust the slope and intercept to verify the changes in the graph
y = 2*(x) + 3

fig = plt.figure(figsize=(8,6))



ax = fig.add_subplot(2,3,1)
ax.plot(x, y, 'r', label=r'y = 2x + 3')
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.legend()# since this ax has a label


y = 1*(x**3) + 1*(x**2) + 1*x + 3
ax = fig.add_subplot(2,3,2)
ax.plot(x, y, 'r')
ax.set_xlabel(r'x')
ax.set_ylabel(r'y')
ax.set_title(r'$y = {x^3} + {x^2} + x + 3$')



y = np.power(x,2)
ax = fig.add_subplot(2,3,3)
ax.plot(x, y, 'r')
ax.set_title(r'$y = {x^2}$')


y = np.exp(x)
ax = fig.add_subplot(2,3,4)
ax.plot(x, y, 'r')
ax.set_title(r'$y = {e^x}$')


y = 1-4/(1+np.power(3, x-2))
ax = fig.add_subplot(2,3,5)
ax.plot(x, y, 'r', label=r'$y = 1 - \frac{4}{1 + 3^{x-2}}$')
ax.set_title('Sigmoidal/Logistic')
ax.legend()


x = np.arange(0.01, 5.0, 0.1)
y = np.exp(x)
ax = fig.add_subplot(2,3,6)
ax.plot(x, y, 'r')
ax.set_title(r'$y = {log(x)}$')


fig.suptitle('linear/non-linear')

"""
pad:用于设置绘图区边缘与画布边缘的距离大小
w_pad:用于设置绘图区之间的水平距离的大小
H_pad:用于设置绘图区之间的垂直距离的大小

fig.tight_layout(pad=0.4, w_pad=3.0, h_pad=3.0)
"""
fig.tight_layout(pad=2)
plt.show()