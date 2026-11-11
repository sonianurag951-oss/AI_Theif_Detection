

import pandas as pd
import numpy as np
from pathlib import Path

# File paths
BASE_DIR = Path(
    r"D:\deep learning\AI_Theif_Detection\Dataset"
)

INPUT_FILE = BASE_DIR / "gridguard_cleaned.csv"

OUTPUT_FILE = BASE_DIR / "gridguard_random_consumption.csv"

# Load dataset
df = pd.read_csv(INPUT_FILE)

# Random generator
rng = np.random.default_rng(42)

# Daily consumption columns
days = [f"day_{i}" for i in range(1, 27)]

# Consumption ranges in kWh/day
consumption_ranges = {
    "Residential": (3, 12),
    "Commercial": (15, 70),
    "Industrial": (80, 350)
}

# Validate categories
if not df["tariff_category"].isin(
    consumption_ranges.keys()
).all():
    raise ValueError("Unexpected tariff_category found.")

# Validate FLAG
df["FLAG"] = pd.to_numeric(df["FLAG"], errors="raise")

if not df["FLAG"].isin([0, 1]).all():
    raise ValueError("FLAG must contain only 0 and 1.")

# Generate base consumption per customer
low = df["tariff_category"].map(
    lambda x: consumption_ranges[x][0]
).to_numpy()

high = df["tariff_category"].map(
    lambda x: consumption_ranges[x][1]
).to_numpy()

base_consumption = rng.uniform(low, high)

# Generate daily variations
daily_variation = rng.normal(
    loc=1.0,
    scale=0.15,
    size=(len(df), 26)
)

daily_variation = np.clip(
    daily_variation, 0.65, 1.40
)

# Generate daily readings
consumption = (
    base_consumption[:, None] * daily_variation
)

# Simulate reduced recorded consumption for theft records
theft_mask = df["FLAG"].to_numpy() == 1

theft_factor = rng.uniform(
    0.25, 0.65, size=len(df)
)

consumption[theft_mask] *= (
    theft_factor[theft_mask, None]
)

# Round to 2 decimal places
consumption = np.round(consumption, 2)

# Update day columns
df.loc[:, days] = consumption

# Save new dataset
df.to_csv(OUTPUT_FILE, index=False)

print("Random daily consumption generated successfully!")
print("Saved file:", OUTPUT_FILE)

print("\nSample data:")
print(df[["customer_id", "tariff_category", "FLAG"] + days[:5]].head())

print("\nConsumption statistics:")
print(df[days].describe())

