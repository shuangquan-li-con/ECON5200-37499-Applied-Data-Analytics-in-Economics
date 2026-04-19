# Time Series Forecasting — ARIMA, GARCH & Bootstrap

## Objective
Develop a robust, diagnostic-driven time series forecasting pipeline by correcting a flawed ARIMA implementation, incorporating seasonal dynamics, and extending the analysis to volatility modeling and distribution-free uncertainty quantification.

---

## Methodology

- **ARIMA Pipeline Diagnostics**
  - Identified three critical issues in the original pipeline:
    - Incorrect use of \( d = 0 \) on a non-stationary CPI series  
    - Failure to model seasonality in monthly data  
    - Omission of residual diagnostics (Ljung–Box test) prior to forecasting  

- **Model Correction (SARIMA)**
  - Applied both first-order and seasonal differencing to ensure stationarity  
  - Transitioned from ARIMA to SARIMA to capture annual seasonality (s = 12)  
  - Conducted systematic model selection across candidate specifications  
  - Evaluated model adequacy using Ljung–Box tests at lags 12 and 24  

- **Residual Diagnostics**
  - Analyzed residual structure using ACF and PACF  
  - Confirmed approximate white-noise residuals before generating forecasts  

- **Volatility Modeling (GARCH)**
  - Modeled S&P 500 daily returns using a GARCH(1,1) specification  
  - Captured volatility clustering and persistence in financial returns  
  - Verified covariance stationarity through \( \alpha + \beta < 1 \)  

- **Forecast Evaluation Module**
  - Built a reusable module (`forecast_evaluation.py`)  
  - Implemented:
    - Mean Absolute Scaled Error (MASE) for scale-independent evaluation  
    - Expanding-window backtesting for realistic out-of-sample validation  

- **Bootstrap Forecast Intervals**
  - Implemented block bootstrap resampling of residuals  
  - Generated distribution-free prediction intervals  
  - Addressed limitations of Gaussian-based confidence intervals  

---

## Key Findings

- The corrected SARIMA model effectively captures both trend and seasonal structure in CPI data. Residual diagnostics indicate no statistically significant autocorrelation at key seasonal lags, suggesting a well-specified model.

- The GARCH(1,1) model reveals strong volatility persistence in S&P 500 returns, with  
  \[
  \alpha + \beta \approx 0.96
  \]  
  indicating a highly persistent but mean-reverting volatility process.

- The estimated half-life of volatility shocks is approximately  
  \[
  17.3 \text{ days}
  \]  
  implying that market shocks dissipate gradually rather than immediately.

- Bootstrap-based forecast intervals provide a more robust representation of uncertainty, particularly under non-normal and heteroskedastic residual structures.

- The modular evaluation framework supports scalable and reproducible comparison of forecasting models across different time series contexts.

---

## Project Structure

