import pandas as pd
import re
import html

# Read original file
file_path = "D:\deep learning\AI_Theif_Detection\Dataset\data.csv"

# Read and clean raw text
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()

cleaned_rows = []

for line in lines:
    values = line.strip().split(",")

    # Remove HTML entities and Markdown formatting
    values = [
        html.unescape(value).replace("*", "").strip()
        for value in values
    ]

    # Remove empty lines
    if not values or len(values) < 3:
        continue

    cleaned_rows.append(values)

# Find customer ID and create columns
data = []

for row in cleaned_rows:
    customer_id_index = next(
        (i for i, value in enumerate(row)
         if re.fullmatch(r"[0-9A-Fa-f]{32}", value)),
        None
    )

    if customer_id_index is None:
        continue

    customer_id = row[customer_id_index]
    flag = row[-1]

    readings = row[:customer_id_index]

    data.append([customer_id] + readings + [flag])

# Determine number of reading columns
number_of_days = len(data[0]) - 2

columns = (
    ["customer_id"]
    + [f"day_{i}" for i in range(1, number_of_days + 1)]
    + ["FLAG"]
)

df = pd.DataFrame(data, columns=columns)

# Convert data types
reading_columns = columns[1:-1]

df[reading_columns] = df[reading_columns].apply(
    pd.to_numeric, errors="coerce"
)

df["FLAG"] = pd.to_numeric(df["FLAG"], errors="coerce")

# Save cleaned dataset
df.to_csv("gridguard_cleaned.csv", index=False)

print(df.head())
print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())