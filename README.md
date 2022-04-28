# transit-model

There are three seperate items in this project:

- the multiple linear regression model

- a "deluxe" multiple linear regression, that attempts to model route-level ridership

- the web app, that summarizes the model results in an interactive way. Hosted here, using Github pages.

## Key functions
### regr_model.py
`generate_model(factors,y_var)`: Uses the specified list of factors to generate a regression for the specified y_var. 'Total Boardings' and 'Route N' are the typical y variables. Returns a model object that contains the regression object, along with various summary statistics.

'regression_no_tt_split()': Does a regression using pingouin library and exports details (coefficients, p values, etc.) to a csv file

There's also a couple functions for generating various explanatory statistics charts.

### deluxe_forecast.py

`predict(sample)`: takes in sample, which is a list that represents a specific scenario. It uses every model object in the `models` list to predict what ridership will be under the given scenario. Returns a float.

`backtest()`: backtests the model on 2019-2021 data. Returns a list of values representing the predicted ridership over 2019-2021.

`point_in_time_forecast(week)`: #Generates a spread of possible results for any given week. Week 53 represents the first week of 2022. Returns a list of 1000 values, representing the spread of possible ridership values for the given week. This function uses Latin Hypercube Sampling to generate the spread. The means and standard deviations for each variable are listed in this function - either based on a linear or log forecast, or 2019 data. Each sample produced by the LHS is fed into the `predict()` function to generate a ridership estimate.

`long_term_forecast(start_week,num_weeks)`: Generates a long term forecast, starting at week `start_week` and ending at week `start_week`+`num_weeks`. For each week, it uses `point_in_time_forecast()` to generate a spread of possible values, then plots the forecast, backtest, and true values using plotly.

`dump_to_excel(values,col=A)`: writes a list of `values` to predictions.xlsx, by default in column A.


##Other Misc. items
`data scraping/generate route data.py` - For preparing route level service hour and boardings estimates. Takes in an excel file with route level data named `VRTS data pull.xlsx`, and re-organizes the data so that each column represents a different route. Columns "Route N" represent boardings; "HN" represents service hours. The column "Total Calculated Revenue Hours" in `data.xlsx` is created by summing up the route level estimates. Outputs a file called `route level data.xlsx`.

`weatherScraping.py` - used for converting daily climate data from environment Canada to weekly data. 2019-2021 data is already in `data.xlsx`, and the p-values were pretty high, so you won't need this. It is recommended to keep climate data in the forecast to give the trends a more random (less artificial) look, while having a negligible impact on results.
