#pandas, sklearn, etc. - for the regression
import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import math
import numpy as np
import random
from datetime import date
#file/directory/output management
import sys, os
from inspect import getsourcefile
from os.path import abspath
import warnings
warnings.filterwarnings("ignore")
#latin hypercube, normal distribution
from pyDOE import *
from scipy.stats.distributions import norm
import plotly.graph_objects as go
import plotly.express as px
import chart_studio
import chart_studio.plotly as py

username = 'WillfulAlpaca' # your plotly username
api_key = 'qE6qePKjViNBblgo5VRS' # your plotly api key - go to profile > settings > regenerate key
import pingouin as pg
# Using a Pandas DataFrame `df`:

#Set directory to file location
directory = abspath(getsourcefile(lambda:0))
#check if system uses forward or backslashes for writing directories
if(directory.rfind("/") != -1):
    newDirectory = directory[:(directory.rfind("/")+1)]
else:
    newDirectory = directory[:(directory.rfind("\\")+1)]
os.chdir(newDirectory)

route_numbers = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,17,21,22,24,25,26,27,28,30,31,32,35,39,43,46,47,48,50,51,52,53,54,55,56,57,58,59,60,61,63,64,65,70,71,72,75,81,82,83,85,87,88]
#these default factors are provided in case a list of factors are not included as an argument in function calls
default_route_factors  = ['SD61 Season', 'Restaurant Bookings', 'Vic Employment', 'BC Vaccination Rate','Season 1','Season 2','Season 3','H{}']
default_system_factors = ['SD61 Season', 'Restaurant Bookings','Vic Employment','Total Calculated Revenue Hours', 'BC Vaccination Rate','Season 1','Season 2','Season 3','Average Precip','Average Temperature']
class model():
    def __init__(self,regr,Rsq=0,fstat=0,MAE=0,MAPE=0,intercept=0,hour_coef = 0, hour_coef_p_val=0, y_var="No variable set"):
        self.y_var = y_var
        self.regr = regr
        self.Rsq = Rsq
        self.fstat = fstat
        self.MAE = MAE
        self.MAPE = MAPE
        self.intercept = intercept
        self.hour_coef = hour_coef
        self.hour_coef_p_val = hour_coef_p_val

df = pandas.read_csv(r"data.csv")
df.fillna(0,inplace=True)

def regression_no_tt_split():
    factors = ["SD61 Season",	"Total Calculated Revenue Hours",	"Restaurant Bookings",	"Population Growth Rate",	"BC Vaccination Rate",	"Season 1",	"Season 2",	"Season 3"]
    x = df[factors]
    y = df["Total Boardings"]
    lm = pg.linear_regression(x,y)
    print(lm)
    lm.to_csv("e.csv")
    return

#Set active directory to model.py location
def generate_model(factors = default_system_factors,y_var = 'Total Boardings',print_stats=False):
    print
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
    intercept = regr.intercept_
    N = 156
    K = len(factors)
    fstat = (Rsq/(1-Rsq))*((N-K-1)/K)
    if(print_stats):
        print("Y Var: {}".format(y_var))
        print("Intercept: {}".format(intercept))
        print('Rsq: {}'.format(Rsq))
        print('F stat: {}'.format(fstat))
        print('MAPE: {}'.format(MAPE))
        #print out coefficients
        sum = pandas.DataFrame(regr.coef_, x.columns, columns=['Coefficient'])

        print(sum)
        print("--------")

    #Get service hour coefficient
    if y_var != 'Total Boardings':
        r_num = y_var[y_var.find(" ")+1:]
        hour_coef = (regr.coef_)[list(x.columns).index("H{}".format(r_num))]
        lm = pg.linear_regression(x_train,y_train)
        loc = factors.index("H{}".format(r_num))+1
        hour_coef_p_val = lm['pval'][loc]
        hour_error = lm["CI[97.5%]"][loc]-lm["CI[2.5%]"][loc]
    else:
        hour_coef = (regr.coef_)[list(x.columns).index("Total Calculated Revenue Hours")]
        lm = pg.linear_regression(x_train,y_train)
        loc = factors.index("Total Calculated Revenue Hours")+1
        hour_coef_p_val = lm['pval'][loc]
        hour_error = lm["CI[97.5%]"][loc]-lm["CI[2.5%]"][loc]

    #returns a model object, including the regr model and a bunch of summary statistics, service hour coefficient, etc.
    return(model(y_var=y_var,Rsq=Rsq,MAE=MAE,MAPE=MAPE,fstat=fstat,intercept=intercept,regr=regr,hour_coef=[hour_coef,hour_error],hour_coef_p_val=hour_coef_p_val))

#Creates regression models for individual routes and generates a .csv file with summary statistics.
def summarize_route_models(route_numbers = route_numbers):
    sum = pandas.DataFrame(columns=["Route","Rsq","Hour Coef","Hour Coef Error","hour_coef_p_val"])
    models  = []
    for route in route_numbers:
        default_route_factors[-1] = "H{}".format(route)
        route_model = generate_model(factors = default_route_factors,y_var = "Route {}".format(route),print_stats=False)
        models.append(route_model)
        sum=sum.append(pandas.DataFrame([[route,route_model.Rsq,route_model.hour_coef[0],route_model.hour_coef[1],route_model.hour_coef_p_val]],columns=["Route","Rsq","Hour Coef","Hour Coef Error","hour_coef_p_val"]))

    sum.to_csv("a.csv")
    return

#produces a parallel_coordinates chart of different variables
def para_coord_chart():
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

    fig = px.parallel_coordinates(df, color="Total Boardings",
        dimensions=['Gas Price (C/L)','Average Temperature','Average Precip','Vic Employment','WFH','University School Season','Restaurant Bookings','BC Vaccination Rate','Total Boardings'],
        color_continuous_scale=px.colors.diverging.Tealrose_r)
    x = py.plot(fig,filename="Variable parallel_coordinates")
    return

#produces a Correlation matrix of different variables
def corr_matrix():
    variables = ['Gas Price (C/L)','Average Temperature','Total Calculated Revenue Hours','Average Precip','Vic Employment','WFH','University School Season','Restaurant Bookings','BC Vaccination Rate','Population Growth Rate','BC Vaccination Rate','Hospitalizations',	'Critical Cases','Total Boardings']
    fig = px.imshow(df[variables].corr())
    annotations = [(dict(xref='paper', yref='paper', x=0.5, y=1.05,
                              xanchor='center', yanchor='bottom',
                              text='Variable Correlation Matrix',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))]

    fig.update_layout(annotations=annotations)
    py.plot(fig,filename="Variable Correlation Matrix",auto_open=True)
    return

#Produces a line graph of the variables (y axis is %change since pre-COVID)
def graph_variables():
    actual_x = []
    fig = go.Figure()
    for value in range(1,53):
        actual_x.append(date.fromisocalendar(2019,week=value,day=1))
    for value in range(1,54):
        actual_x.append(date.fromisocalendar(2020,week=value,day=1))
    for value in range(1,53):
        actual_x.append(date.fromisocalendar(2021,week=value,day=1))

    for factor in default_system_factors:
        if factor.find("Average") == -1:
            new_data = []
            p1 = df[factor][0]
            if factor.find("Restaurant") == -1 and factor.find("Season") == -1:
                for p in df[factor]:
                    new_data.append(p/p1)
                fig.add_trace(go.Scatter(
                    y=new_data,x=actual_x,
                    name = factor + " (% change since Jan 2019)"))
    for factor in ["Restaurant Bookings","University School Season"]:
        new_data = df[factor]
        fig.add_trace(go.Scatter(
            y=new_data,x=actual_x,
            name = factor
    ))
    fig.add_trace(go.Scatter(y=df['Total Boardings'],x=actual_x, name='Total Boardings'))
    fig.update(layout_yaxis_range = [-1,4])
    annotations = [(dict(xref='paper', yref='paper', x=0.5, y=1.05,
                              xanchor='center', yanchor='bottom',
                              text='Model Variables, 2019-2021',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))]
    fig.update_layout(bargap=0.2,annotations=annotations)
    x = py.plot(fig, filename = "Variable Change",auto_open=True)

#para_coord_chart()
#corr_matrix()
#graph_variables()

generate_model(print_stats=1  )
