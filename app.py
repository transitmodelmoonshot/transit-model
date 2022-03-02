
# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
from flask import Flask, render_template
#from forecast import point_in_time_forecast
import plotly.graph_objects as go
import plotly.express as px
import chart_studio
import chart_studio.plotly as py
# Flask constructor takes the name of
# current module (__name__) as argument.
app = Flask(__name__)

# The route() function of the Flask class is a decorator,
# which tells the application which URL should call
# the associated function.
@app.route('/')
def home():
    return 'Hello World'

@app.route('/<week>')
def prediction(week):
    """username = 'WillfulAlpaca' # your username
    api_key = 'qE6qePKjViNBblgo5VRS' # your api key - go to profile > settings > regenerate key
    chart_studio.tools.set_credentials_file(username=username, api_key=api_key)
    fig = px.histogram(point_in_time_forecast(int(week)),nbins=40,labels={'value':'Weekly Ridership'})
    x = py.plot(fig, filename = 'gdp_per_cap', auto_open=False)"""
    return(render_template('prediction.html'))
# main driver function
if __name__ == '__main__':

    # run() method of Flask class runs the application
    # on the local development server.
    app.run()
