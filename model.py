import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
df = pandas.read_csv("C:\Users\marki\Documents\GitHub\transit-model\data.csv")
# importing train_test_split from sklearn

X = df[['Revenue Hours','Restaurant Bookings','Gas Price (C/L)','University School Season','Employment','WFH','Population Growth Rate','BC Vaccination Rate	hospitalizations']]
y = df['Total boardings']


# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

regr = linear_model.LinearRegression()
regr.fit(x_train, y_train)

#predict the CO2 emission of a car where the weight is 2300kg, and the volume is 1300cm3:
y_prediction = regr.predict([[0,0,0,0,0,0,0,0,0]])

print(y_prediction)

# importing r2_score module
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
# predicting the accuracy score
#score=r2_score(y_test,y_prediction)
#print(‘r2 socre is ‘,score)
#print(‘mean_sqrd_error is==’,mean_squared_error(y_test,y_prediction))
#print(‘root_mean_squared error of is==’,np.sqrt(mean_squared_error(y_test,y_prediction)))
