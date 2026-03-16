# Data Analysis Plan
## Public Transportation Passenger Statistics in Thailand

This project analyzes daily passenger statistics from Thailand's public transportation systems.

The analysis focuses on:

1. Modal share of public transportation systems
2. Passenger behavior across urban rail lines
3. Detection of special events and anomalies
4. Forecasting passenger demand using Facebook Prophet

Dataset period:
~14 months of daily passenger statistics (2025–2026)

---

# Project Constraints

Submission format:

- Single Jupyter Notebook
- Google Colab link
- Visualizations + Insights
- Forecasting model

---

# Workflow Overview

Phase 1 — Data Loading  
Phase 2 — Data Cleaning  
Phase 3 — Data Transformation  
Phase 4 — Modal Share Analysis  
Phase 5 — Urban Rail Comparison  
Phase 6 — Event Detection  
Phase 7 — Forecasting (Prophet Model)  
Phase 8 — Insights & Conclusion  

---

# Phase 1 — Data Loading

## Goal

Load datasets and inspect structure.

Datasets:

- `passengers68.csv`
- `passengers69.csv`

## Steps

1. Load both datasets
2. Concatenate datasets
3. Inspect structure

## Example Code

```python
import pandas as pd

df68 = pd.read_csv("passengers68.csv")
df69 = pd.read_csv("passengers69.csv")

df = pd.concat([df68, df69])
```

---

# Phase 2 — Data Cleaning

## Convert Date Format

```python
df["date"] = pd.to_datetime(df["วันที่"])
```

## Sort Data

```python
df = df.sort_values("date")
```

---

## Filter Dataset

Dataset contains multiple transport modes.

For this challenge we focus on:

**Rail Transport**

Filter conditions:

- `รูปแบบการเดินทาง == "ทางราง"`
- `สาธารณะ/ส่วนบุคคล == "สาธารณะ"`

---

## Check Data Quality

Check:
- Missing values
- Zero values
- Duplicate rows

Example:

```python
df.isna().sum()
```

---

# Phase 3 — Data Transformation

Dataset is in long format:

| date | transport | passengers |

We convert it into time-series wide format:

| date | BTS | MRT Blue | MRT Purple | ARL | SRT Red |

Using pivot:

```python
pivot_df = df.pivot_table(
    index="date",
    columns="ยานพาหนะ/ท่า",
    values="ปริมาณ",
    aggfunc="sum"
)
```

---

## Feature Engineering

Add useful time features:

```python
pivot_df["year"] = pivot_df.index.year
pivot_df["month"] = pivot_df.index.month
pivot_df["day_of_week"] = pivot_df.index.day_name()
pivot_df["is_weekend"] = pivot_df["day_of_week"].isin(["Saturday", "Sunday"])
```

---

# Phase 4 — Modal Share Analysis

## Objective

Identify the most used transportation system.

## Transport Groups

| Mode | Lines |
|------|-------|
| BTS | BTS |
| MRT | Blue + Purple + Yellow + Pink |
| ARL | Airport Rail Link |
| SRT | Red Line |

---

## Calculate Total Passengers

```python
modal_total = {
    "BTS": pivot_df["BTS"].sum(),
    "MRT": pivot_df[["MRT Blue", "MRT Purple", "MRT Yellow", "MRT Pink"]].sum().sum(),
    "ARL": pivot_df["ARL"].sum(),
    "SRT": pivot_df["SRT Red"].sum()
}
```

---

## Visualization

Charts:
1. Modal share pie chart
2. Modal share stacked area chart
3. Growth comparison (2025 vs 2026)

---

# Phase 5 — Urban Rail Comparison

Compare passenger behavior across lines.

## Lines

- BTS
- MRT Blue
- MRT Purple
- MRT Yellow
- MRT Pink
- Airport Rail Link
- SRT Red

---

## Time Series Plot

Daily ridership comparison: `date` vs `passengers`

**Visualization:** Multi-line time series chart

---

## Average Ridership

Compute mean passenger count:

```python
pivot_df.mean()
```

**Ranking:**

| Line | Avg Daily Passenger |

---

## Volatility Analysis

Passenger stability metric: **standard deviation**

- **High volatility** → event-sensitive systems
- **Low volatility** → stable commuter base

---

# Phase 6 — Event Detection

## Objective

Detect unusual passenger patterns.

---

## Total Passenger Trend

Compute total ridership:

```python
pivot_df["total_passengers"] = pivot_df.sum(axis=1)
```

---

## Moving Average

```python
pivot_df["rolling_7"] = pivot_df["total_passengers"].rolling(7).mean()
```

---

## Anomaly Detection

Using Z-score:

$$z = \frac{x - \mu}{\sigma}$$

If $|z| > 3$ → anomaly detected

---

## Event Mapping

Match anomalies with events:

| Event | Expected Impact |
|-------|----------------|
| Songkran | passenger drop |
| New Year | drop |
| Long weekend | drop |
| Special event | spike |

---

# Phase 7 — Passenger Forecasting (Facebook Prophet)

## Objective

Predict passenger demand for the next 30 days.

Facebook Prophet is a time-series forecasting model developed by Meta that decomposes a time series into **trend**, **seasonality**, and **holiday effects**.

Prophet expects a dataset with two columns:

- `ds` (date)
- `y` (value to forecast)

---

## Prepare Data for Prophet

```python
prophet_df = pivot_df.reset_index()[["date", "total_passengers"]]
prophet_df.columns = ["ds", "y"]
```

---

## Train Prophet Model

```python
from prophet import Prophet

model = Prophet()
model.fit(prophet_df)
```

---

## Create Future Dates & Forecast

Predict next 30 days:

```python
future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)
```

---

## Visualization

Forecast plot:

```python
model.plot(forecast)
```

Components:

```python
model.plot_components(forecast)
```

Prophet will automatically show:
- Trend
- Weekly seasonality
- Yearly seasonality

---

# Phase 8 — Insights & Storytelling

Summarize key findings.

## Example Insights

**Insight 1**  
BTS carries the largest share of passengers among rail systems.

**Insight 2**  
MRT ridership is growing faster due to expansion of new lines.

**Insight 3**  
Airport Rail Link shows strong weekend patterns.

**Insight 4**  
Songkran causes a significant drop in ridership.

**Insight 5**  
Prophet forecasts show continued growth in rail passenger demand over the next 30 days.

---

# Notebook Structure

Final notebook structure:

1. Introduction
2. Load Dataset
3. Data Cleaning
4. Data Transformation
5. Modal Share Analysis
6. Urban Rail Comparison
7. Event Detection
8. Prophet Forecast
9. Key Insights

---

# Expected Visualizations

**Minimum:**

1. Modal Share Pie Chart
2. Modal Share Stacked Area
3. Ridership Trend
4. Rail Line Comparison
5. Rolling Trend Chart
6. Anomaly Detection Plot
7. Forecast Plot

**Total:** 7–10 visualizations

---

# Final Deliverable

Submission includes:
- Google Colab Notebook
- Visualization graphs
- Key insights
- Passenger forecast

---

# คำแนะนำเพิ่ม (สำคัญมากสำหรับ Hackathon)

แนะนำ **เพิ่ม 3 อย่างนี้ใน notebook จะดูเทพขึ้นมาก**

## 1️⃣ Holiday Feature ใน Prophet

ใส่:
- Songkran
- New Year
- Long weekend

Prophet รองรับ **holiday effect** โดยตรง:

```python
from prophet import Prophet
import pandas as pd

holidays = pd.DataFrame({
    "holiday": ["songkran", "songkran", "songkran",
                "new_year", "new_year"],
    "ds": pd.to_datetime([
        "2025-04-13", "2025-04-14", "2025-04-15",
        "2026-01-01", "2025-01-01"
    ]),
    "lower_window": 0,
    "upper_window": 1,
})

model = Prophet(holidays=holidays)
model.fit(prophet_df)
```

---

## 2️⃣ Forecast แยกแต่ละสาย

แทนที่จะ forecast total อย่างเดียว ให้ forecast:
- BTS
- MRT Blue
- ARL

กรรมการจะชอบมาก

---

## 3️⃣ Interactive Graph

ใช้ **Plotly** แทน Matplotlib:

```python
import plotly.express as px

fig = px.line(pivot_df, x=pivot_df.index, y=pivot_df.columns,
              title="Daily Ridership by Rail Line")
fig.show()
```
