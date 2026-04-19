Time Series Forecasting — ARIMA, GARCH & Bootstrap
Objective

Develop a robust end-to-end time series forecasting pipeline by diagnosing and correcting a flawed ARIMA implementation, extending it to seasonal dynamics, and integrating volatility modeling and distribution-free uncertainty quantification.

Methodology
ARIMA Pipeline Diagnostics
Identified critical specification errors in the original model:
Incorrect use of d=0 on a non-stationary CPI series
Omission of seasonal dynamics in monthly data
Absence of residual diagnostics prior to forecasting
Model Correction (SARIMA)
Applied both first-order and seasonal differencing to ensure stationarity
Transitioned from ARIMA to SARIMA to capture annual seasonality (s = 12)
Performed systematic model selection across candidate specifications
Validated model adequacy using Ljung–Box tests at seasonal lags
Residual Diagnostics
Evaluated autocorrelation structure via ACF/PACF
Confirmed approximate white-noise residuals before proceeding to forecasting
Volatility Modeling (GARCH)
Modeled S&P 500 daily returns using a GARCH(1,1) specification
Estimated conditional variance dynamics to capture volatility clustering
Verified covariance stationarity via α+β<1
Forecast Evaluation Module
Designed a reusable evaluation module (forecast_evaluation.py)
Implemented:
Mean Absolute Scaled Error (MASE) for scale-free comparison
Expanding-window backtesting for realistic out-of-sample evaluation
Bootstrap Forecast Intervals
Constructed block bootstrap simulations of residuals
Generated distribution-free prediction intervals
Addressed limitations of normality-based confidence intervals
