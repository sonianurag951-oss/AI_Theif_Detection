
import os
import pandas as pd
import psycopg

from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

def fetch_data():

    with psycopg.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        dbname=os.getenv("PGDATABASE"),
        user=os.getenv("PGUSER"),
        password=os.getenv("PGPASSWORD")
    ) as conn:

        query = """
            SELECT *
            FROM public.gridguard_data
        """

        df = pd.read_sql(query, conn)

    print("Data fetched successfully!")
    print("Shape:", df.shape)

    return df


if __name__ == "__main__":

    df = fetch_data()

    print("\nDataset Preview:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTarget Distribution:")
    print(df["flag"].value_counts())
