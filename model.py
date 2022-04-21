#pandas, sklearn, etc. - for the regression
import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import math
import numpy as np
import random
#file/directory/output management
import sys, os
from inspect import getsourcefile
from os.path import abspath
import warnings
warnings.filterwarnings("ignore")
#latin hypercube, normal distribution
from pyDOE import *
from scipy.stats.distributions import norm

directory = abspath(getsourcefile(lambda:0))
#check if system uses forward or backslashes for writing directories
if(directory.rfind("/") != -1):
    newDirectory = directory[:(directory.rfind("/")+1)]
else:
    newDirectory = directory[:(directory.rfind("\\")+1)]
os.chdir(newDirectory)

df = pandas.read_csv(r"data.csv")
df.fillna(0,inplace=True)


#Set active directory to model.py location
def generate_model(factors = ['Restaurant Bookings','Vic Employment','Total Calculated Revenue Hours', 'BC Vaccination Rate','Season 1','Season 2','Season 3','Average Precip','Average Temperature'],y_var = 'Total Boardings'):

    #set explanatory variables to factors
    x = df[factors]
    y = df[y_var] #dependent variable

    # splitting the data
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.15, random_state = random.randint(1,100))

    regr = linear_model.LinearRegression() #create model
    regr.fit(x_train, y_train)
    y_prediction = regr.predict(x_test) #make predictions using test set

    # calculate the accuracy score and other statistics
    Rsq=r2_score(y_test,y_prediction)
    MAE = mean_absolute_error(y_test,y_prediction)
    MAPE = mean_absolute_percentage_error(y_test,y_prediction)
    INT = regr.intercept_
    N = 156
    K = len(factors)
    fstat = (Rsq/(1-Rsq))*((N-K-1)/K)



    print("Intercept: {}".format(INT))
#    print('Rsq: {}'.format(Rsq))
#    print('F stat: {}'.format(fstat))
    print('MAPE: {}'.format(MAPE))

    #print out summary statistics
    sum = pandas.DataFrame(regr.coef_, x.columns, columns=['Coefficient'])

    print(sum)
    return(regr)

"""res = []
route_numbers = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,17,21,22,24,25,26,27,28,30,31,32,35,39,43,46,47,48,50,51,52,53,54,55,56,57,58,59,60,61,63,64,65,70,71,72,75,81,82,83,85,87,88]
for route in route_numbers:
    res.append(generate_model(factors = ['Restaurant Bookings','Vic Employment', 'BC Vaccination Rate','Season 1','Season 2','Season 3','Average Precip','Average Temperature','H{}'.format(route)],y_var = "Route {}".format(route)))
print(np.median(res))
 """
generate_model(factors = ['Restaurant Bookings','Total Calculated Revenue Hours', 'BC Vaccination Rate','Season 1','Season 2','Season 3'])
