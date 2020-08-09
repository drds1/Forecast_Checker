# Forecast Performance Analysis

## Intro
These notebooks make use of a python package built to 
interpret the performance of a timeseries forecaster. 

Typically, when forecasting a timeseries, we want to understand the performance of 
the 1, 5, 10, 20-step forecasting performance. We expect forecasting performance 
to degrade with the forecast length but the measure of a good forecaster over a bad one 
might be how far ahead we can make reliable forward predictions before
the performance degrades to below some threshold.

## ARIMA Process with Exogoneous inputs

A common timeseries forecasting approach utilises a feature common to many natural and soci-economic processes:
that the next value depends directly on the previous (auto regressive), or indirectly responds to some earlier shock (moving average). 
We often also find that timeseries are non-stationary and required differencing or modelling of exogeneous variables.
[This notebook](https://github.com/dstarkey23/Forecast_Checker/blob/master/notebooks/Forecast_Checker.ipynb) uses a custom-built class to generate some synthetic data for timeseries modelling based on an ARMA process
with polynomial exogeneous variables.


## Forecast Checker

This package (installable from PyPi `pip install forecast_checker`) ingests a timeseries and measures the correlation coefficient
for various forecast step sizes by iteratively chopping-off the final data point from the timeseries, forecasting the future and repeating
for different forecast step sizes. This allows us to calculate the correlation coefficient as a function of forecast length
and impose a cut-off threshold above which we consider the forecast performance too low to base a prediction.


### Example Output

##### Iterative Correlation Plot
![Iterative Correlation Plot](https://github.com/dstarkey23/Forecast_Checker/blob/master/images/test_eval_plot.png)


##### Synthetic Timeseries Decomposition (ARIMA + Exogoneous Variables + Simple Forecast)
![Synthetic Timeseries](https://github.com/dstarkey23/Forecast_Checker/blob/master/images/arima_test.png)
