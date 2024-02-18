# Analysing GDP using Machine Learning algorithms and World Bank data

This project comes out as part of the work done in my Master thesis. Here ML algorithms are used - using World Bank data - to obtain models with higher performance - measured in MAE and MSE - than standard econometric models - such as those used by the IMF. Years 2002 to 2021 are selected.

## Overview of the project:
The following steps are taken into consideration:
- **Data downloading:** using the World Bank API
- **Data preparation:** cleaning data, selecting years, etcetera
- **Exploratory analysis:** initial plots
- **Outlier detection:** to assess whether certain countries may have data quality issues
- **Training ML models:** different models are attempted and trained, using walk forward CV for time series to tune hyperparameters
- **Forecast combination**

----------------------

## Results
Best performing algorithm is the **Random Forest regression** using the scikit-learn library

| Model | MAE    |  MSE   |
| :---:   | :---: | :---: |
| RF regression | 0.5703   | 0.8681   |
| WEO (spring report) | 0.6955   | 0.9915   |
| WEO (fall report) | 0.6504   | 0.9364   |
