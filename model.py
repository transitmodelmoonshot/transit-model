import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math

df = pandas.read_csv(r"C:\Users\marki\Documents\GitHub\transit-model\data.csv")
# importing train_test_split from sklearn

#x = df[['Revenue Hours','Restaurant Bookings','Gas Price (C/L)','University School Season','Employment','WFH','Population Growth Rate','BC Vaccination Rate',	'hospitalizations']]
x = df.drop(['Total Boardings','ISOYear','Month','ISOWeek','Start of Week'],axis=1)
y = df['Total Boardings']


# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)
y_prediction = regr.predict(x_test)
#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
#y_prediction = regr.predict([[0,0,0,0,0,0,0,0,0]])

# importing r2_score module

# predicting the accuracy score
Rsq=r2_score(y_test,y_prediction)
print("R2 value".format(Rsq))
print('Root Mean_sqrd_error is {}'.format(math.sqrt(mean_squared_error(y_test,y_prediction))))
coeff_df = pandas.DataFrame(regr.coef_, x.columns, columns=['Coefficient'])
print(coeff_df)
N = 156
K = 9
fstat = (Rsq/(1-Rsq))*((N-K-1)/K) #find N and K yourself
print(fstat)
