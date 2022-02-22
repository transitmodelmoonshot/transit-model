#pandas, sklearn, etc. - for the regression
import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
import numpy as np

#file/directory/output management
import sys, os
from inspect import getsourcefile
from os.path import abspath
import warnings

#latin hypercube, normal distribution
from pyDOE import *
from scipy.stats.distributions import norm

from dump_to_excel import dump_to_excel

#Set active directory to model.py location
directory = abspath(getsourcefile(lambda:0))
#check if system uses forward or backslashes for writing directories
if(directory.rfind("/") != -1):
    newDirectory = directory[:(directory.rfind("/")+1)]
else:
    newDirectory = directory[:(directory.rfind("\\")+1)]
os.chdir(newDirectory)

df = pandas.read_csv(r"data.csv")
#Put the mean /stdev of the independent vars here. MAKE SURE THEY ARE IN THE CORRECT ORDER.
#Revenue Hours, Restaurant Bookings, Gas Price (C/L), University School Season, Employment, WFH, Population Growth Rate, BC Vaccination Rate
factors = ['Revenue Hours', 'Restaurant Bookings', 'Gas Price (C/L)', 'University School Season', 'Employment', 'WFH', 'Population Growth Rate', 'BC Vaccination Rate','Average Temperature','Average Precip']

means = [None,None,None,None,2783,None,None,.95,17,0.4] #Means and stdevs for prediction. Put None if you want to use 2021 mean and stdev.
stdvs = [None,None,None,None,22,None,None,0.1,4.8,0.1]

data2021 = df[105:] #2021 data only
i = 0
for factor in factors: #get the mean and std dev of the 2021 data - used for predicting future values
    if means[i] is None:
        means[i] = np.mean(data2021[factor])
        stdvs[i] = np.std(data2021[factor])
    i =i+1

#exclude the following columns from the explanatory dataset
x = df[factors]
y = df['Total Boardings'] #dependent variable
# splitting the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 5)

regr = linear_model.LinearRegression() #create model
regr.fit(x_train, y_train)
y_prediction = regr.predict(x_test) #make predictions using test set

# predicting the accuracy score, other statistics

Rsq=r2_score(y_test,y_prediction)
print("Constant: {}".format(regr.intercept_))
print("R2 value: {}".format(Rsq))
print('Root Mean_sqrd_error: {}'.format(math.sqrt(mean_squared_error(y_test,y_prediction))))
N = 156
K = len(factors)
fstat = (Rsq/(1-Rsq))*((N-K-1)/K)
print('F stat: {}'.format(fstat))

#print out summary statistics
sum = pandas.DataFrame([], x.columns)
sum["2021 Mean"] = means
sum["2021 stdev"] = stdvs
sum["Coeff"] = regr.coef_
print(sum)

#latin hypercube
num_factors = len(factors)
design = lhs(num_factors,samples=10000) #get latin hypercube samples
for j in range(num_factors):
     design[:, j] = norm(loc=means[j], scale=stdvs[j]).ppf(design[:, j]) #scale values to a normal distribution

warnings.filterwarnings("ignore")
results = []
for sample in design:
    y_prediction = regr.predict([sample]) #make prediction using sample. Note the warning can be ignored.
    results.append(y_prediction[0])

dump_to_excel(results) #write results to an excel file for later graphing and analysis. This overwrites the previous results!
print("Simulations complete and written to predictions.xlsx. Average ridership predicted: {}".format(np.mean(results)))
