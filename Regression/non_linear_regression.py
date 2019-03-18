import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

df = pd.read_csv("china_gdp.csv")

x_data, y_data = (df["Year"].values, df["Value"].values)

"""
Choosing a model
From an initial look at the plot, we determine that the logistic function 
could be a good approximation, since it has the property of starting with 
a slow growth, increasing growth in the middle, and then decreasing again 
at the end; as illustrated below:
The formula for the logistic function is the following:

𝑌_hat =1/1+𝑒^𝛽1(𝑋−𝛽2)
 
𝛽1 : Controls the curve's steepness,

𝛽2 : Slides the curve on the x-axis.

Building The Model
Now, let's build our regression model and initialize its parameters.
"""


def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1 * (x - Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0


fig = plt.figure(figsize=(8, 5))

#logistic function
Y_pred = sigmoid(x_data, beta_1 , beta_2)
#plot initial prediction against datapoints
ax = fig.add_subplot(1,2,1)
ax.plot(x_data, Y_pred*15000000000000.)
ax.plot(x_data, y_data, 'ro')
ax.set_xlabel('Year')
ax.set_ylabel('GDP')



# Lets normalize our data
xdata =x_data/max(x_data)
ydata =y_data/max(y_data)

popt, pcov = curve_fit(sigmoid, xdata, ydata)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))


x = np.linspace(1960, 2015, 55)
x = x/max(x)

y = sigmoid(x, *popt)

ax = fig.add_subplot(1,2,2)
ax.plot(xdata, ydata, 'ro', label='data')
ax.plot(x, y, linewidth=3.0, label='fit')
ax.legend(loc='best')
ax.set_ylabel('GDP')
ax.set_xlabel('Year')
plt.show()