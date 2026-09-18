# AI-Based Forecasting and Adaptive Energy Dispatch for Microgrids

An AI-based forecasting framework for **short-term load demand, solar PV
generation, and wind power prediction**, followed by **net-load
estimation and adaptive energy dispatch** for a grid-connected
microgrid.

## Project Overview

Microgrids integrate conventional loads with renewable energy sources
such as solar PV and wind generation. Because both electricity demand
and renewable generation vary with time and weather, accurate
forecasting is important for effective energy management.

This project develops an integrated machine-learning pipeline that:

1.  Preprocesses historical load, solar, and wind datasets.
2.  Extracts time-based and physical/weather-related features.
3.  Normalizes the input and target variables using `MinMaxScaler`.
4.  Uses a chronological **80:20 train-test split** without shuffling.
5.  Creates **24-hour sliding-window sequences** for time-series
    learning.
6.  Trains separate **LSTM networks** for load, solar PV, and wind
    forecasting.
7.  Evaluates the forecasting models using MSE, RMSE, MAE, MAPE, and R².
8.  Combines predicted renewable generation with predicted load to
    estimate net load.
9.  Uses the predicted net load to generate battery/grid dispatch
    decisions.

## System Pipeline

``` text
Historical Load Data ───────┐
                            │
Historical Solar Data ──────┼──> Preprocessing & Feature Engineering
                            │
Historical Wind Data ───────┘
                                      │
                                      ▼
                              Normalization
                                      │
                                      ▼
                           24-Hour Time Windows
                                      │
             ┌────────────────────────┼────────────────────────┐
             ▼                        ▼                        ▼
       Load LSTM                 Solar LSTM                Wind LSTM
             │                        │                        │
             └────────────────────────┼────────────────────────┘
                                      ▼
                           Forecasted Load / Solar / Wind
                                      │
                                      ▼
                           Renewable Generation
                             = Solar + Wind
                                      │
                                      ▼
                    Net Load = Load − Renewable Generation
                                      │
                                      ▼
                         Energy Management System
                                      │
                         ┌────────────┴────────────┐
                         ▼                         ▼
                  Battery Discharge         Battery Charging
                    / Grid Import             / Energy Storage
```

## Dataset

The implementation uses three time-series datasets:

-   **Load demand:** historical electricity load data.
-   **Solar PV:** historical solar power generation data.
-   **Wind power:** historical wind generation data with meteorological
    variables.

The notebook expects the following CSV files:

``` text
loadv2.csv
solarv2.csv
windv2.csv
```

The current notebook reads them from Google Drive:

``` text
/content/drive/MyDrive/loadv2.csv
/content/drive/MyDrive/solarv2.csv
/content/drive/MyDrive/windv2.csv
```

The datasets themselves are not included in this repository unless
explicitly added by the project author.

## Data Preprocessing

### Timestamp Processing

The datasets use different timestamp formats. A custom parser is used
for the load dataset, while solar and wind timestamps are converted
using Pandas datetime parsing.

### Missing Values

Missing target values are converted to numeric values where necessary
and treated using **linear interpolation**. Any remaining missing target
values are removed.

### Time Features

The following temporal features are generated:

-   Hour
-   Day of week
-   Day of year
-   Month
-   Season

Additional physical/weather features are used where available.

### Model Inputs

**Load model**

Uses time features together with weather variables `w1`--`w25`.

**Solar model**

Uses time-based features.

**Wind model**

Uses time features together with:

-   `U10`
-   `V10`
-   `U100`
-   `V100`

## LSTM Architecture

Each forecasting model follows the same basic architecture:

``` text
Input
  │
  ▼
LSTM (128 units, return_sequences=True)
  │
  ▼
Dropout (0.2)
  │
  ▼
LSTM (64 units)
  │
  ▼
Dropout (0.2)
  │
  ▼
Dense (32 units, ReLU)
  │
  ▼
Dense (1)
  │
  ▼
Forecast
```

### Training Configuration

  Parameter                               Value
  ------------------ --------------------------
  Lookback window                      24 hours
  Train/Test split                    80% / 20%
  Optimizer                                Adam
  Loss function        Mean Squared Error (MSE)
  Load batch size                            32
  Solar batch size                           64
  Wind batch size                            64
  Maximum epochs                            100
  Dropout                                   0.2
  Early stopping                        Enabled

Early stopping monitors validation loss and restores the best model
weights.

A fixed random seed of **42** is used for NumPy and TensorFlow.

## Forecasting Models

Three independent LSTM models are trained:

### 1. Load Demand Forecasting

Predicts future electricity demand from historical temporal and
weather-related information.

### 2. Solar PV Forecasting

Predicts solar power generation using temporal information.

### 3. Wind Power Forecasting

Predicts wind power generation using temporal and wind-related
meteorological features.

## Evaluation Metrics

The models are evaluated using:

-   **MSE --- Mean Squared Error**
-   **RMSE --- Root Mean Squared Error**
-   **MAE --- Mean Absolute Error**
-   **MAPE --- Mean Absolute Percentage Error**
-   **R² --- Coefficient of Determination**

These metrics provide complementary information about prediction error
and explained variance.

## Reported Project Results

The project/thesis reports the following forecasting results:

  Forecast            R²        MAE       RMSE
  ------------- -------- ---------- ----------
  Load Demand     0.9155   12.16 MW   14.84 MW
  Solar PV        0.8257    0.08 MW    0.13 MW
  Wind Power      0.5652    0.18 MW    0.23 MW

The reported results indicate different forecasting performance across
the three resources, with wind generation presenting greater forecasting
difficulty due to its variability.

## Net Load and Energy Dispatch

The forecasted renewable generation is combined as:

``` text
Renewable Generation = Solar + Wind
```

The predicted net load is calculated as:

``` text
Net Load = Load − (Solar + Wind)
```

The resulting net-load forecast is passed to a simple energy-management
decision layer.

Conceptually:

``` text
Net Load > 0  → Battery discharge / energy import
Net Load < 0  → Battery charging / renewable surplus
Net Load = 0  → Balanced condition
```

The notebook generates a dispatch DataFrame containing timestamps,
normalized load, solar, wind, renewable generation, net load, and the
corresponding battery action.

## Project Outputs

The notebook generates analysis and visualization outputs including:

-   Load forecasting plots
-   Solar PV forecasting plots
-   Wind forecasting plots
-   Training and validation loss curves
-   Feature correlation matrices
-   Top correlated features
-   Forecasting result plots
-   Normalized battery-dispatch results
-   LSTM evaluation metrics

Example generated files include:

``` text
microgrid_forecasting_results.png
load_predictions.png
solar_predictions.png
wind_predictions.png
load_correlation.png
solar_correlation.png
wind_correlation.png
load_top_correlations.png
solar_top_correlations.png
wind_top_correlations.png
normalized_battery_dispatch.csv
lstm_metrics_normalized_scale.csv
```

## Technologies Used

-   Python
-   Google Colab
-   TensorFlow / Keras
-   NumPy
-   Pandas
-   Scikit-learn
-   Matplotlib

## How to Run

### 1. Open the notebook

Open:

``` text
btechproject.ipynb
```

in Google Colab or a compatible Jupyter environment.

### 2. Provide the datasets

Place:

``` text
loadv2.csv
solarv2.csv
windv2.csv
```

in the expected Google Drive location, or update the file paths in the
notebook.

### 3. Run the notebook

Execute the cells sequentially.

The notebook will:

``` text
Load data
   ↓
Parse timestamps
   ↓
Handle missing values
   ↓
Create features
   ↓
Normalize data
   ↓
Chronological train/test split
   ↓
Create 24-hour sequences
   ↓
Train LSTM models
   ↓
Generate forecasts
   ↓
Evaluate models
   ↓
Calculate net load
   ↓
Generate dispatch decisions
```

## Limitations

-   The study is based on historical data and simulation rather than
    physical microgrid hardware.
-   Forecasting accuracy depends on the quality and representativeness
    of the available datasets.
-   Renewable generation, particularly wind power, is affected by highly
    variable atmospheric conditions.
-   The energy-management layer is a simplified dispatch strategy rather
    than a complete industrial microgrid EMS.
-   Real-world deployment would require additional considerations such
    as battery state-of-charge constraints, grid limits, electricity
    pricing, demand response, and real-time control.

## Future Work

Potential extensions include:

-   Model Predictive Control (MPC) for dispatch.
-   Reinforcement-learning-based energy management.
-   Real-time electricity pricing.
-   Demand-response integration.
-   More detailed battery and microgrid models.
-   Real-time sensor and fault-handling mechanisms.
-   Hardware-in-the-loop or real microgrid validation.
-   Improved forecasting architectures and uncertainty estimation.

## Author

**Suva Pratyush Tripathy**\
B.Tech --- Electrical and Electronics Engineering\
IIIT Bhubaneswar

## Note

This repository contains the computational notebook for the project.
Dataset files and generated outputs may be omitted from the repository
because of file size, licensing, or reproducibility considerations.
Check the notebook paths and dataset availability before running the
complete pipeline.
