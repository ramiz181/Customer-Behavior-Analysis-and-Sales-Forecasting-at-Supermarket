import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

# =====================================================
# PATHS
# =====================================================

INPUT_PATH = "../data/features/monthly_sales_features.csv"

OUTPUT_DIR = "../outputs/forecasting"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("LINEAR REGRESSION — SALES FORECASTING")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")
print(df.head())

# =====================================================
# FEATURES & TARGET
# =====================================================

X = df[["month", "quarter"]]

y = df["total_sales"]

# =====================================================
# TRAIN MODEL
# =====================================================

print("\nTraining Linear Regression Model...")

model = LinearRegression()

model.fit(X, y)

print("✓ Model Trained Successfully")

# =====================================================
# PREDICT EXISTING DATA
# =====================================================

y_pred = model.predict(X)

# =====================================================
# MODEL EVALUATION
# =====================================================

print("\nMODEL PERFORMANCE")

r2 = r2_score(y, y_pred)

mae = mean_absolute_error(y, y_pred)

rmse = np.sqrt(mean_squared_error(y, y_pred))

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae:.2f}")
print(f"RMSE     : {rmse:.2f}")

# =====================================================
# FORECAST NEXT 3 MONTHS
# =====================================================

print("\nForecasting Future Sales...")

future_months = pd.DataFrame({
    "month": [13, 14, 15],
    "quarter": [1, 1, 1]
})

future_predictions = model.predict(future_months)

forecast_df = pd.DataFrame({
    "month": ["Jan 2024", "Feb 2024", "Mar 2024"],
    "predicted_sales": future_predictions
})

print("\nFuture Forecast:\n")

print(forecast_df)

# =====================================================
# SAVE FORECAST
# =====================================================

forecast_path = f"{OUTPUT_DIR}/sales_forecast.csv"

forecast_df.to_csv(forecast_path, index=False)

print(f"\n✓ Forecast saved:")
print(forecast_path)

# =====================================================
# VISUALIZATION
# =====================================================

print("\nGenerating Forecast Visualization...")

plt.figure(figsize=(10, 6))

# Actual Sales
plt.plot(
    df["month"],
    y,
    marker="o",
    linewidth=2,
    label="Actual Sales"
)

# Predicted Sales
plt.plot(
    df["month"],
    y_pred,
    linestyle="--",
    marker="o",
    linewidth=2,
    label="Predicted Sales"
)

# Future Forecast
future_x = [13, 14, 15]

plt.plot(
    future_x,
    future_predictions,
    linestyle=":",
    marker="o",
    linewidth=3,
    label="Forecast Sales"
)

plt.title(
    "Monthly Sales Forecast using Linear Regression",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Month")
plt.ylabel("Sales Revenue")

plt.xticks(
    ticks=list(range(1, 16)),
    labels=[
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec",
        "Jan24", "Feb24", "Mar24"
    ],
    rotation=45
)

plt.legend()

plot_path = f"{OUTPUT_DIR}/sales_forecast_plot.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"✓ Forecast chart saved:")
print(plot_path)

# =====================================================
# BUSINESS INTERPRETATION
# =====================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

growth = (
    (future_predictions[-1] - y.iloc[-1])
    / y.iloc[-1]
) * 100

print(
    f"\nPredicted sales growth over next "
    f"3 months: {growth:.2f}%"
)

print(
    "\nThe forecasting model helps management "
    "estimate future demand and improve "
    "inventory planning."
)

print("\n✓ Sales Forecasting Completed Successfully")
print("=" * 60)