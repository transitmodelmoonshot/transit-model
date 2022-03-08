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

from dump_to_excel import dump_to_excel

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


factors = ['Revenue Hours','Restaurant Bookings','Employment', 'BC Vaccination Rate','Season 1','Season 2','Season 3','Average Temperature','Average Precip']
routes = ["Total Boardings"]

models = []
for y_var in routes:
    models.append(generate_model(factors,y_var))


df = pandas.read_csv(r"data.csv")

def predict(sample): #takes in a scenario (sample), generates predictions for each route (y_var), and combines them.
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
    year = 0 #represents 2021
    season_week = week     #season week represents the week of the given year. (1-52) (ex. week 1 of 2019, week 0 of 2020.
    while(1):
        if(season_week > 52):
            season_week = season_week-52
            year = year +1
        else:
            break

    print("Point in time prediction for week {} (Week {} of 202{}).".format(week,season_week,year+1),end="")
    season_week = season_week-1 #subtract one since data2019 is zero-indexed
    means = [11800+400*year,-0.8437-0.132658*math.log(1.0/week),2606+1.877785*week,0.603301-0.059246*math.log(1.0/(week-31)),data2019['Season 1'][season_week],data2019['Season 2'][season_week],data2019['Season 3'][season_week],df['Average Temperature'][week-52],df['Average Precip'][week-52]] #Means and stdevs for prediction.
    stdvs = [zero,0.023*abs(math.log(1/week)+0.072),week*0.16+4.86,.00257*abs(math.log(1/(week-31)))+.006,zero,zero,zero,zero,zero]

    #latin hypercube
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
    x_rev = x[::-1]
    y1 = means
    y1_lower = min
    y1_upper = max
    y1_lower = y1_lower[::-1]
    actual_x = []
    for value in range(1,53):
        actual_x.append(date.fromisocalendar(2019,week=value,day=1))
    for value in range(1,54):
        actual_x.append(date.fromisocalendar(2020,week=value,day=1))
    for value in range(1,53):
        actual_x.append(date.fromisocalendar(2021,week=value,day=1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df["Total Boardings"],x=actual_x,
        name = "Actual"
    ))

    fig.add_trace(go.Scatter(
        x=x, y=y1,
        line_color='rgb(0,100,80)',
        name='Predicted'
    ))

    fig.add_trace(go.Scatter(
        x=x+x_rev,
        y=y1_upper+y1_lower,
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
        name='Uncertainty',
    ))


    fig.add_trace(go.Scatter(
        y = back_test(),x = actual_x,
        line_color='rgb(0,100,80)',
        name='Predicted (Backtest)',
         line=dict(dash='dot')
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
    fig.show()

    return
def point_in_time_forecast_histogram(week): #make a nice histogram of point_in_time_forecast
    username = 'WillfulAlpaca' # your username
    api_key = 'qE6qePKjViNBblgo5VRS' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    import chart_studio.plotly as py
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
long_term_forecast(53,105)
#point_in_time_forecast_histogram(100)
