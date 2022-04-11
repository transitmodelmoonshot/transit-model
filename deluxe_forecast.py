#pandas, sklearn, etc. - for the regression
import pandas
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
import math
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import chart_studio
import chart_studio.plotly as py
username = 'WillfulAlpaca' # your plotly username
api_key = 'qE6qePKjViNBblgo5VRS' # your plotly api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

import random
#file/directory/output management
import sys, os
from inspect import getsourcefile
from os.path import abspath
import warnings
from datetime import date
#latin hypercube, normal distribution
from pyDOE import *
from scipy.stats.distributions import norm
#a tool to quickly write to excel
from dump_to_excel import dump_to_excel
#generate regression model using sklearn
from model import generate_model

#Set active directory to forecast.py location
directory = abspath(getsourcefile(lambda:0))
#check if system uses forward or backslashes for writing directories
if(directory.rfind("/") != -1):
    newDirectory = directory[:(directory.rfind("/")+1)]
else:
    newDirectory = directory[:(directory.rfind("\\")+1)]
os.chdir(newDirectory)

warnings.filterwarnings("ignore")

routes = []
#Uncomment the following line to do a regression of Total Boardings
routes = ["Total Boardings"]

factors = ['Revenue Hours','Restaurant Bookings','Vic Employment', 'BC Vaccination Rate','Season 1','Season 2','Season 3','Average Temperature','Average Precip']

route_numbers = [4]
#route_numbers = [1,2,3,4,6,7,8,9,10,11,12,13,14,15,17,21,22,24,25,26,27,28,30,31,32,35,39,43,46,47,48,50,51,52,53,54,55,56,57,58,59,60,61,63,64,65,70,71,72,75,81,82,83,85,87,88]
if(routes != ['Total Boardings']):
    for route in route_numbers:
        routes.append("Route {}".format(route))

models = []
for y_var in routes:
    models.append(generate_model(factors,y_var))
    print("Generated model for factor: {}".format(y_var))

df = pandas.read_csv(r"data.csv")

def predict(sample): #takes in a scenario (sample), generates predictions for each route (y_var), and combines them. Returns an estimated ridership count. (float)
    total_ridership = 0
    for model in models:
        total_ridership = total_ridership + model.predict([sample])
    return(total_ridership)

def back_test(): #backtests the model. for each week from 2019,2021, generate predicts. Returns a list of values.
    results = []
    for week in range(0,157):
        sample = []
        for factor in factors:
            sample.append(df[factor][week])
        results.append(predict(sample)[0])
    return(results)

def point_in_time_forecast(week): #Generates a spread of possible results for any given week.
    zero = 0.00000001 #pseudo-zero. S.D. for LHS sample must be > than 0.
    data2019 = df[:52] #2019 data only
    year = 0 #represents 2020
    season_week = week     #season week represents the ISO week of the given year. (1-52) (ex. week 1 of 2019)
    while(1):
        if(season_week > 52):
            season_week = season_week-52
            year = year +1
        else:
            break
    season_week = season_week-1 #subtract one since data2019 is zero-indexed
    #season_week is used to get equivalent 2019 climate data. year is to predict future service hours.
    print("Point in time prediction for week {} (Week {} of 202{}).".format(week,season_week,year+1),end="")

    #means and STDevs for LHS sampling
    means = [data2019['Revenue Hours'][season_week]+400*year,-0.8437-0.132658*math.log(1.0/week), 6.04-0.0406*week,0.603301-0.059246*math.log(1.0/(week-31)),data2019['Season 1'][season_week],data2019['Season 2'][season_week],data2019['Season 3'][season_week],df['Average Temperature'][week-52],df['Average Precip'][week-52]] #Means and stdevs for prediction.
    stdvs = [zero,0.023*abs(math.log(1/week)+0.072), week*0.00518+0.158,.00257*abs(math.log(1/(week-31)))+.006,zero,zero,zero,zero,zero]

    #latin hypercube sampling. Generate 1000 scenarios, and predict ridership for each one.
    num_factors = len(factors)
    design = lhs(num_factors,samples=1000) #get latin hypercube samples
    for j in range(num_factors):
         design[:, j] = norm(loc=means[j], scale=stdvs[j]).ppf(design[:, j]) #scale values to a normal distribution

    results = []
    for sample in design:
        y_prediction = predict(sample) #make prediction using sample. Note the warning can be ignored.
        results.append(y_prediction[0])

    return(results)

def long_term_forecast(start_week,num_weeks): #run point_in_time_forecast for each week in the future and plot the results
    means = []
    min = []
    max = []
    x = []
    for week in range(start_week,start_week+num_weeks):
        s_week = week
        year = 2021
        while(s_week>52):
            s_week = s_week-52
            year = year+1
        r = date.fromisocalendar(year=year, week = s_week, day=1)
        x.append(r) #generate x values (time axis)
        data = point_in_time_forecast(week)
        data.sort()
        means.append(np.mean(data))
        min.append(data[50]) #get 5% and 95% of the data (90% confidence range)
        max.append(data[950])
        print("Forecast: {}".format(np.mean(data)))

    #from here it's just graphing
    y1 = means
    y1_lower = min
    y1_upper = max
    actual_x = []
#    dump_to_excel(y1,col="B")
#    dump_to_excel(min,col="C")

    #generate x values for 2019-2021
    for value in range(1,53):
        actual_x.append(date.fromisocalendar(2019,week=value,day=1))
    for value in range(1,54):
        actual_x.append(date.fromisocalendar(2020,week=value,day=1))
    for value in range(1,53):
        actual_x.append(date.fromisocalendar(2021,week=value,day=1))
    #dump_to_excel(x,col="A")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df["Total Boardings"],x=actual_x,
        name = "Actual"
    ))

    fig.add_trace(go.Scatter(
        x=x+x[::-1],
        y=y1_upper+y1_lower[::-1],
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        name='90% Uncertainty',
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y1,
        line_color='rgb(0,100,80)',
        name='Predicted'
    ))

    fig.add_trace(go.Scatter(
        y = back_test(),x = actual_x,
        line_color='rgb(0,100,80)',
        name='Predicted (Backtest)',
         line=dict(dash='dot')
    ))

    data_2022 = pandas.read_csv("2022 ridership data.csv")
    dates_2022 = []
    for d in data_2022['Week']:
        dates_2022.append(date.fromisocalendar(year = 2022, week = d, day = 1))
    fig.add_trace(go.Scatter(
        y = data_2022['Boardings'], x = dates_2022,
        name = 'Actual 2022'
    ))

    fig.update_traces(mode='lines')
    annotations = [(dict(xref='paper', yref='paper', x=0.5, y=1.05,
                              xanchor='center', yanchor='bottom',
                              text='Forecast of Weekly Ridership',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))]

    fig.update(layout_yaxis_range = [0,1000000])
    fig.update_layout(annotations=annotations)
    x = py.plot(fig, filename = "longtermpredictions",auto_open=True)

    print(x)
    return

def point_in_time_forecast_histogram(week): #make a nice histogram of point_in_time_forecast
    fig = px.histogram(point_in_time_forecast(week),histnorm = 'percent',nbins=40,labels={'value':'Weekly Ridership'})
    annotations = [(dict(xref='paper', yref='paper', x=0.5, y=1.05,
                              xanchor='center', yanchor='bottom',
                              text='June 2022 Ridership Forecast',
                              font=dict(family='Arial',
                                        size=30,
                                        color='rgb(37,37,37)'),
                              showarrow=False))]
    fig.update_layout(bargap=0.2,annotations=annotations)
    x = py.plot(fig, filename = 'Prediction over time', auto_open=False)

    print(x)
    return
#long_term_forecast(53,104)
point_in_time_forecast_histogram(100)
