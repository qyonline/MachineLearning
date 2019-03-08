import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model

df = pd.read_csv('FuelConsumptionCo2.csv')

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

regr1 = linear_model.LinearRegression()
regr2 = linear_model.LinearRegression()
regr3 = linear_model.LinearRegression()
train_x1 = np.asanyarray(train[['ENGINESIZE']])
train_x2 = np.asanyarray(train[['CYLINDERS']])
train_x3 = np.asanyarray(train[['FUELCONSUMPTION_COMB']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr1.fit (train_x1, train_y)
regr2.fit (train_x2, train_y)
regr3.fit (train_x3, train_y)


from sklearn.metrics import r2_score

test_x1 = np.asanyarray(test[['ENGINESIZE']])
test_x2 = np.asanyarray(test[['CYLINDERS']])
test_x3 = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])
test_y1 = regr1.predict(test_x1)
test_y2 = regr1.predict(test_x2)
test_y3 = regr1.predict(test_x3)

print("Mean absolute error: %.2f %.2f %.2f" % (np.mean(np.absolute(test_y1 - test_y)),np.mean(np.absolute(test_y2 - test_y)),np.mean(np.absolute(test_y3 - test_y))))
print("Residual sum of squares (MSE): %.2f %.2f %.2f" % (np.mean((test_y1 - test_y) ** 2),np.mean((test_y2 - test_y) ** 2),np.mean((test_y3 - test_y) ** 2)))
print("R2-score: %.2f %.2f %.2f" % (r2_score(test_y1 , test_y),r2_score(test_y2 , test_y) ,r2_score(test_y3 , test_y)) )

fig = plt.figure(figsize=(4, 12))
p1=fig.add_subplot(311)
p1.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
p1.plot(train_x1, regr1.predict(train_x1), '-r')
p1.set_xlabel("Engine size")
p1.set_ylabel("Emission")

p2=fig.add_subplot(312)
p2.scatter(train.CYLINDERS, train.CO2EMISSIONS,  color='blue')
p2.plot(train_x2, regr2.predict(train_x2), '-r')
p2.set_xlabel("Cylinder")
p2.set_ylabel("Emission")

p3=fig.add_subplot(313)
p3.scatter(train.FUELCONSUMPTION_COMB, train.CO2EMISSIONS,  color='blue')
p3.plot(train_x3, regr3.predict(train_x3), '-r')
p3.set_xlabel("Fuel Consumption")
p3.set_ylabel("Emission")

plt.show()
