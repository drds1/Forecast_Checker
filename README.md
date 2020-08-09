# Forecast Performance Analysis

## Intro
These notebooks make use of a python package I developed to 
interpret the performance of a timeseries forecaster. 

Typically, when forecasting a timeseries, we want to understand the performance of 
the 1, 5, 10, 20-step forecasting performance. We expect forecasting performance 
to degrade with the forecast length but the measure of a good forecaster over a bad one 
might be how far ahead we can make reliable forward predictions before
the performance degrades to below some threshold.

## Forecast Checker

this package (installable from PyPi `pip install forecast_checker`) ingests a timeseries
forecast class

Performance analysis library on stats models-type arima problems

test on lucky
