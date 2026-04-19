# Time Series Diagnostics & Advanced Decomposition

## Objective
This project diagnoses common time-series decomposition and stationarity-testing mistakes, then extends the analysis with multi-seasonal decomposition, block bootstrap uncertainty estimation, and structural break detection.

## Project Components
- STL decomposition with log transform for multiplicative data
- ADF and KPSS testing with corrected deterministic specification
- MSTL decomposition for multiple seasonal cycles
- Moving block bootstrap confidence bands for trend uncertainty
- PELT structural break detection and per-regime stationarity testing
- Reusable Python module in `src/decompose.py`
- Interactive Streamlit app for FRED-based diagnostics

## Repo Structure
```text
econ-lab-20-time-series/
├── README.md
├── requirements.txt
├── app.py
├── notebooks/
│   └── lab_20_time_series.ipynb
├── src/
│   └── decompose.py
├── figures/
│   ├── stl_decomposition.png
│   ├── bootstrap_ci.png
│   └── structural_breaks.png
└── verification-log.md
