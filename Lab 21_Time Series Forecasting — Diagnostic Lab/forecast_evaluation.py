"""
forecast_evaluation.py — Forecast Evaluation & Backtesting Module

Reusable functions for computing MASE and running expanding-window
backtests on time series forecasting models.

Author: Your Name
Course: ECON 5200, Lab 21
"""

import numpy as np
import pandas as pd
from typing import Callable


def compute_mase(
    actual: np.ndarray,
    forecast: np.ndarray,
    insample: np.ndarray,
    m: int = 1
) -> float:
    """Compute Mean Absolute Scaled Error (MASE).

    Args:
        actual: True out-of-sample values
        forecast: Model predictions
        insample: In-sample data used to build naive benchmark
        m: Seasonal period (1 = nonseasonal naive, 12 = monthly seasonal naive)

    Returns:
        MASE value
    """
    actual = np.asarray(actual, dtype=float)
    forecast = np.asarray(forecast, dtype=float)
    insample = np.asarray(insample, dtype=float)

    if len(actual) != len(forecast):
        raise ValueError("actual and forecast must have the same length.")
    if len(insample) <= m:
        raise ValueError("insample length must be greater than m.")

    mae_forecast = np.mean(np.abs(actual - forecast))
    naive_errors = insample[m:] - insample[:-m]
    mae_naive = np.mean(np.abs(naive_errors))

    if mae_naive == 0:
        raise ValueError("Naive benchmark MAE is zero, cannot compute MASE.")

    return mae_forecast / mae_naive


def backtest_expanding_window(
    series: pd.Series,
    model_fn: Callable,
    min_train: int = 120,
    horizon: int = 12,
    step: int = 1,
    m: int = 1
) -> pd.DataFrame:
    """Run an expanding-window backtest.

    Args:
        series: Time series with index
        model_fn: Function that takes a training series and horizon, returns forecasts
        min_train: Initial training window size
        horizon: Forecast horizon
        step: Step size between forecast origins
        m: Seasonal period for MASE

    Returns:
        DataFrame with forecast evaluation metrics
    """
    if len(series) < min_train + horizon:
        raise ValueError("Series is too short for the requested backtest setup.")

    records = []

    for origin in range(min_train, len(series) - horizon + 1, step):
        train = series.iloc[:origin]
        test = series.iloc[origin:origin + horizon]

        forecast = model_fn(train, horizon)
        forecast = np.asarray(forecast, dtype=float)

        if len(forecast) != len(test):
            raise ValueError("model_fn must return forecast of length equal to horizon.")

        mae = np.mean(np.abs(test.values - forecast))
        rmse = np.sqrt(np.mean((test.values - forecast) ** 2))
        mase = compute_mase(
            actual=test.values,
            forecast=forecast,
            insample=train.values,
            m=m
        )

        records.append({
            'train_end': train.index[-1],
            'test_start': test.index[0],
            'test_end': test.index[-1],
            'mae': mae,
            'rmse': rmse,
            'mase': mase
        })

    return pd.DataFrame(records)
