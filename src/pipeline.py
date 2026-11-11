
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")


BASE_DIR = Path(
    r"D:\deep learning\AI_Theif_Detection"
)

DATA_DIR = BASE_DIR / "Dataset"
RESULT_DIR = BASE_DIR / "Result"

EDA_DIR = RESULT_DIR / "EDA"
OUTLIER_DIR = RESULT_DIR / "Outliers"
STAT_DIR = RESULT_DIR / "Statistics"
REPORT_DIR = RESULT_DIR / "Reports"

for folder in [
    DATA_DIR,
    RESULT_DIR,
    EDA_DIR,
    OUTLIER_DIR,
    STAT_DIR,
    REPORT_DIR
]:
    folder.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DATA_DIR / "gridguard_random_consumption.csv"

# CLEAN_FILE = DATA_DIR / "gridguard_cleaned.csv"
# FEATURE_FILE = DATA_DIR / "gridguard_features.csv"

days_col = [f"day_{i}" for i in range(1, 27)]

ID_COLS = ["customer_id", "meter_id"]

CATEGORICAL_COLS = [
    "meter_type",
    "region",
    "tariff_category"
]

REQUIRED_COLS = (
    ID_COLS
    + days_col
    + ["FLAG", "installation_date"]
    + CATEGORICAL_COLS
)

sns.set_theme(style="whitegrid")

df = pd.read_csv(INPUT_FILE)
print(df.shape)
print(df.head())
print(df.info())

missing_columns = [
    col for col in REQUIRED_COLS
    if col not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )

# DATA QUALITY

original_rows = len(df)
duplicate_rows = int(df.duplicated().sum())

duplicate_customers = int(
    df["customer_id"].duplicated().sum()
)
duplicate_meters = int(
    df["meter_id"].duplicated().sum()
)

missing_before = df.isnull().sum()

print( original_rows)
print(duplicate_rows)
print(duplicate_customers)
print(duplicate_meters)
print(missing_before)


df = df.drop_duplicates().copy()

for col in ID_COLS:
    df[col] = df[col].astype("string").str.strip()

for col in ID_COLS:
    df[col] = df[col].replace("", pd.NA)

#  DAILY CONSUMPTION

for col in days_col:
    df[col] = pd.to_numeric(
        df[col],
        errors="coerce"
    )

# Count negative readings
negative_counts = int(
    (df[days_col] < 0).sum().sum()
)

print(negative_counts)

df[days_col] = df[days_col].mask(
    df[days_col] < 0
)

# Count missing readings
missing_readings = int(
    df[days_col].isnull().sum().sum()
)

print( missing_readings)

# CLEAN TARGET VARIABLE

df["FLAG"] = pd.to_numeric(
    df["FLAG"],
    errors="coerce"
)

invalid_target = (
    df["FLAG"].isna()
    | ~df["FLAG"].isin([0, 1])
)

invalid_target_count = int(invalid_target.sum())

print("Invalid target rows:", invalid_target_count)

# Target is essential for supervised learning
df = df.loc[~invalid_target].copy()

df["FLAG"] = df["FLAG"].astype(int)

# CLEAN CATEGORICAL COLUMNS

for col in CATEGORICAL_COLS:
    df[col] = (
        df[col]
        .astype("string")
        .str.strip()
        .str.title()
    )

    df[col] = df[col].replace("", pd.NA)

# CLEAN INSTALLATION DATE

df["installation_date"] = pd.to_datetime(
    df["installation_date"],
    errors="coerce"
)

# Future installation dates 
future_dates = (
    df["installation_date"] > pd.Timestamp.today()
)

future_date_count = int(future_dates.sum())

print( future_date_count)


#  IDENTIFIER VALIDATION

missing_id_rows = df[ID_COLS].isnull().any(axis=1)

print(
    int(missing_id_rows.sum())
)

df = df.loc[~missing_id_rows].copy()

duplicate_meter_ids = df.loc[
    df["meter_id"].duplicated(keep=False),
    ["customer_id", "meter_id"]
]




print( df.shape)

#  TARGET DISTRIBUTION

target_counts = df["FLAG"].value_counts().sort_index()

target_percentage = (
    df["FLAG"].value_counts(normalize=True).sort_index()
    * 100
)

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="FLAG",
    hue="FLAG",
    palette="Set2",
    legend=False
)

plt.title("Normal vs Electricity Theft Customers")
plt.xlabel("0 = Normal, 1 = Theft")
plt.ylabel("Number of Customers")

plt.savefig(
    EDA_DIR / "target_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

# DAILY CONSUMPTION TREND

daily_mean = df[days_col].mean()

plt.figure(figsize=(13, 5))

plt.plot(
    range(1, 27),
    daily_mean.values,
    marker="o"
)

plt.title("Average Daily Electricity Consumption")
plt.xlabel("Day")
plt.ylabel("Average Consumption")

plt.xticks(range(1, 27))

plt.savefig(
    EDA_DIR / "daily_consumption_trend.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

#  NORMAL VS THEFT CONSUMPTION

group_mean = df.groupby("FLAG")[days_col].mean()

plt.figure(figsize=(14, 6))

for flag, label in [
    (0, "Normal Customers"),
    (1, "Theft Customers")
]:

    if flag in group_mean.index:

        plt.plot(
            range(1, 27),
            group_mean.loc[flag].values,
            marker="o",
            label=label
        )

plt.title("Normal vs Theft Consumption Pattern")
plt.xlabel("Day")
plt.ylabel("Average Consumption")
plt.xticks(range(1, 27))
plt.legend()

plt.savefig(
    EDA_DIR / "normal_vs_theft.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

#  CONSUMPTION DISTRIBUTION

df["average_consumption"] = df[days_col].mean(axis=1)

plt.figure(figsize=(9, 5))

sns.histplot(
    data=df,
    x="average_consumption",
    hue="FLAG",
    bins=40,
    kde=True
)

plt.title("Customer Consumption Distribution")

plt.savefig(
    EDA_DIR / "consumption_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

#  OUTLIER DETECTION USING IQR

outlier_cols = []

for day in days_col:

    q1 = df[day].quantile(0.25)
    q3 = df[day].quantile(0.75)

    iqr = q3 - q1

    lower = max(0, q1 - 1.5 * iqr)
    upper = q3 + 1.5 * iqr

    col = f"{day}_outlier"

    df[col] = (
        (df[day] < lower)
        | (df[day] > upper)
    ).astype(int)

    outlier_cols.append(col)

# Outlier statistics
df["total_outliers"] = df[outlier_cols].sum(axis=1)

df["has_outlier"] = (
    df["total_outliers"] > 0
).astype(int)

df["outlier_ratio"] = (
    df["total_outliers"] / len(days_col)
)

total_outliers = int(
    df[outlier_cols].sum().sum()
)

print("\nTotal outlier readings:", total_outliers)

# Outlier by target
outlier_comparison = df.groupby("FLAG")[
    "total_outliers"
].agg(["mean", "median", "max"])

outlier_comparison.to_csv(
    STAT_DIR / "outlier_comparison.csv"
)

# Boxplot
plt.figure(figsize=(16, 7))

sns.boxplot(
    data=df[days_col],
    orient="h"
)

plt.title("Daily Electricity Consumption Outliers")

plt.savefig(
    OUTLIER_DIR / "daily_outliers.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

#  CORRELATION ANALYSIS

correlation = df[days_col].corr()

plt.figure(figsize=(16, 12))

sns.heatmap(
    correlation,
    cmap="coolwarm",
    annot=False
)

plt.title("Daily Consumption Correlation")

plt.savefig(
    EDA_DIR / "correlation_heatmap.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

#  CATEGORICAL FEATURE ANALYSIS

for col in CATEGORICAL_COLS:

    plt.figure(figsize=(9, 5))

    sns.countplot(
        data=df,
        x=col,
        hue="FLAG",
        palette="Set2"
    )

    plt.title(f"{col} vs Electricity Theft")

    plt.xticks(rotation=30)

    plt.savefig(
        EDA_DIR / f"{col}_distribution.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    category_report = pd.crosstab(
        df[col],
        df["FLAG"],
        normalize="index"
    ) * 100

    category_report.to_csv(
        STAT_DIR / f"{col}_theft_percentage.csv"
    )

#  INSTALLATION DATE ANALYSIS

df["installation_year"] = (
    df["installation_date"].dt.year
)

df["installation_month"] = (
    df["installation_date"].dt.month
)

today = pd.Timestamp.today().normalize()

df["meter_age_days"] = (
    today - df["installation_date"]
).dt.days

df.loc[
    df["meter_age_days"] < 0,
    "meter_age_days"
] = np.nan

plt.figure(figsize=(10, 5))

sns.boxplot(
    data=df,
    x="FLAG",
    y="meter_age_days"
)

plt.title("Meter Age vs Theft Classification")

plt.savefig(
    EDA_DIR / "meter_age_vs_theft.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()

#  CONSUMPTION STATISTICS

statistics = df[days_col].describe().T

statistics.to_csv(
    STAT_DIR / "consumption_statistics.csv"
)

comparison = df.groupby("FLAG")[
    "average_consumption"
].agg([
    "mean",
    "median",
    "std",
    "min",
    "max"
])

comparison.to_csv(
    STAT_DIR / "theft_normal_comparison.csv"
)

#  FEATURE ENGINEERING

features = pd.DataFrame(index=df.index)

# Basic statistics
features["consumption_mean"] = df[days_col].mean(axis=1)
features["consumption_median"] = df[days_col].median(axis=1)
features["consumption_std"] = df[days_col].std(axis=1)
features["consumption_min"] = df[days_col].min(axis=1)
features["consumption_max"] = df[days_col].max(axis=1)

features["consumption_total"] = df[days_col].sum(axis=1)

features["consumption_range"] = (
    features["consumption_max"]
    - features["consumption_min"]
)

features["consumption_iqr"] = (
    df[days_col].quantile(0.75, axis=1)
    - df[days_col].quantile(0.25, axis=1)
)

safe_mean = features["consumption_mean"].replace(0, np.nan)

features["consumption_cv"] = (
    features["consumption_std"] / safe_mean
)

features["peak_to_average_ratio"] = (
    features["consumption_max"] / safe_mean
)

# Zero consumption features
features["zero_consumption_days"] = (
    df[days_col] == 0
).sum(axis=1)

features["zero_consumption_ratio"] = (
    features["zero_consumption_days"] / 26
)

features["has_zero_consumption"] = (
    features["zero_consumption_days"] > 0
).astype(int)

# Low consumption
features["low_consumption_days"] = (
    df[days_col]
    .lt(safe_mean * 0.25, axis=0)
    .sum(axis=1)
)

# Daily consumption changes
daily_change = df[days_col].diff(axis=1)
features["average_daily_change"] = daily_change.mean(axis=1)
features["daily_change_std"] = daily_change.std(axis=1)
features["max_daily_increase"] = daily_change.max(axis=1)
features["max_daily_decrease"] = daily_change.min(axis=1)
features["sudden_increase_days"] = (
    daily_change.gt(safe_mean * 0.5, axis=0)
    .sum(axis=1)
)

features["sudden_decrease_days"] = (
    daily_change.lt(-safe_mean * 0.5, axis=0)
    .sum(axis=1)
)

# First half vs second half
first_half = days_col[:13]
second_half = days_col[13:]

features["first_half_mean"] = df[first_half].mean(axis=1)

features["second_half_mean"] = df[second_half].mean(axis=1)

features["half_period_change"] = (
    features["second_half_mean"]
    - features["first_half_mean"]
)

features["half_period_change_ratio"] = (
    features["half_period_change"]
    / features["first_half_mean"].replace(0, np.nan)
)


# Outlier features
features["total_outliers"] = df[outlier_cols].sum(axis=1)

features["outlier_ratio"] = (
    features["total_outliers"] / 26
)

features["has_outlier"] = (
    features["total_outliers"] > 0
).astype(int)

# Meter metadata features
features["meter_age_days"] = df["meter_age_days"]

features["installation_year"] = df["installation_year"]

features["installation_month"] = df["installation_month"]

for col in CATEGORICAL_COLS:
    features[col] = df[col]

# Original daily readings retained
for col in days_col:
    features[col] = df[col]

#  CREATE FINAL FEATURE DATASET

engineered_df = pd.concat(
    [
        df[ID_COLS],
        features,
        df[["FLAG"]]
    ],
    axis=1
)

# Feature names for model development
feature_cols = list(features.columns)

pd.DataFrame({
    "feature_name": feature_cols
}).to_csv(
    RESULT_DIR / "feature_names.csv",
    index=False
)

# DATA QUALITY REPORT

normal_count = int((df["FLAG"] == 0).sum())

theft_count = int((df["FLAG"] == 1).sum())

total = len(df)

theft_percentage = (
    theft_count / total * 100 if total else 0
)

report = f"""
          GRIDGUARD AI DATA QUALITY REPORT
ORIGINAL ROWS          : {original_rows}
CLEANED ROWS           : {len(df)}
REMOVED EXACT DUPLICATES: {duplicate_rows}
DUPLICATE CUSTOMER IDs : {duplicate_customers}
DUPLICATE METER IDs    : {duplicate_meters}
NEGATIVE READINGS      : {negative_counts}
MISSING DAILY READINGS : {missing_readings}
INVALID TARGET ROWS    : {invalid_target_count}
FUTURE INSTALL DATES   : {future_date_count}

TARGET DISTRIBUTION

NORMAL CUSTOMERS       : {normal_count}
THEFT CUSTOMERS        : {theft_count}
THEFT PERCENTAGE       : {theft_percentage:.2f}%

OUTLIER ANALYSIS

TOTAL OUTLIER READINGS : {total_outliers}

DAILY CONSUMPTION STATISTICS

{statistics.round(3).to_string()}

THEFT VS NORMAL

{comparison.round(3).to_string()}

MISSING VALUES AFTER CLEANING

{df.isnull().sum().to_string()}

FEATURE ENGINEERING

TOTAL ENGINEERED FEATURES : {len(feature_cols)}

FEATURE MATRIX SHAPE      : {features.shape}

OUTPUT FILES

EDA RESULTS      : {EDA_DIR}
STATISTICS       : {STAT_DIR}
REPORTS          : {REPORT_DIR}

"""

(REPORT_DIR / "gridguard_quality_report.txt").write_text(
    report,
    encoding="utf-8"
)

print(report)

print("Total features:", len(feature_cols))
print("Final shape:", engineered_df.shape)
