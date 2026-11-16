import os
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
import pandas as pd
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "Dataset"
CSV_FILE = DATASET_DIR / "gridguard_random_consumption.csv"
FALLBACK_CSV_FILE = DATASET_DIR / "gridguard.csv"

load_dotenv(PROJECT_ROOT / ".env")

DAY_COLS = [f"day_{i}" for i in range(1, 27)]
_CACHED_DF: Optional[pd.DataFrame] = None


def get_dataset_dataframe() -> Optional[pd.DataFrame]:
    """Load and cache the customer readings dataset from CSV."""
    global _CACHED_DF
    if _CACHED_DF is not None:
        return _CACHED_DF

    if CSV_FILE.exists():
        _CACHED_DF = pd.read_csv(CSV_FILE)
    elif FALLBACK_CSV_FILE.exists():
        _CACHED_DF = pd.read_csv(FALLBACK_CSV_FILE)
    return _CACHED_DF


def fetch_from_postgres(
    customer_id: Optional[str] = None,
    meter_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Query PostgreSQL if available."""
    try:
        import psycopg
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_db = os.getenv("PGDATABASE")
        pg_user = os.getenv("PGUSER")
        pg_password = os.getenv("PGPASSWORD")

        if not all([pg_db, pg_user, pg_password]):
            return None

        with psycopg.connect(
            host=pg_host,
            port=pg_port,
            dbname=pg_db,
            user=pg_user,
            password=pg_password,
            connect_timeout=2
        ) as conn:
            with conn.cursor() as cur:
                clauses, params = [], []
                if customer_id:
                    clauses.append("UPPER(TRIM(customer_id)) = UPPER(TRIM(%s))")
                    params.append(customer_id)
                if meter_id:
                    clauses.append("UPPER(TRIM(meter_id)) = UPPER(TRIM(%s))")
                    params.append(meter_id)

                if not clauses:
                    return None

                query = f"""
                    SELECT * FROM public.gridguard_data
                    WHERE {" OR ".join(clauses) if not (customer_id and meter_id) else " AND ".join(clauses)}
                    LIMIT 1;
                """
                cur.execute(query, params)
                row = cur.fetchone()

                if row:
                    col_names = [desc[0].lower() for desc in cur.description]
                    data_dict = dict(zip(col_names, row))
                    readings = [float(data_dict.get(col, 0.0)) for col in DAY_COLS]
                    return {
                        "found": True,
                        "source": "PostgreSQL Database",
                        "customer_id": str(data_dict.get("customer_id", "")),
                        "meter_id": str(data_dict.get("meter_id", "")),
                        "readings": readings,
                        "flag": int(data_dict.get("flag", 0)) if data_dict.get("flag") is not None else None,
                        "meter_type": str(data_dict.get("meter_type", "Standard Smart Meter")),
                        "region": str(data_dict.get("region", "Unknown")),
                        "tariff_category": str(data_dict.get("tariff_category", "Residential")),
                        "installation_date": str(data_dict.get("installation_date", ""))
                    }
    except Exception:
        pass
    return None


def fetch_from_csv(
    customer_id: Optional[str] = None,
    meter_id: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Query local CSV dataset for customer reading."""
    df = get_dataset_dataframe()
    if df is None or df.empty:
        return None

    cust_clean = customer_id.strip().upper() if customer_id else None
    meter_clean = meter_id.strip().upper() if meter_id else None
    has_meter_col = "meter_id" in df.columns

    match = None
    # 1. Match both if provided
    if cust_clean and meter_clean and has_meter_col:
        cond = (
            (df["customer_id"].astype(str).str.strip().str.upper() == cust_clean) &
            (df["meter_id"].astype(str).str.strip().str.upper() == meter_clean)
        )
        subset = df[cond]
        if not subset.empty:
            match = subset.iloc[0]

    # 2. Match by customer_id
    if match is None and cust_clean:
        cond = df["customer_id"].astype(str).str.strip().str.upper() == cust_clean
        subset = df[cond]
        if not subset.empty:
            match = subset.iloc[0]

    # 3. Match by meter_id
    if match is None and meter_clean and has_meter_col:
        cond = df["meter_id"].astype(str).str.strip().str.upper() == meter_clean
        subset = df[cond]
        if not subset.empty:
            match = subset.iloc[0]

    if match is not None:
        readings = [float(match.get(col, 0.0)) for col in DAY_COLS]
        flag_val = match.get("FLAG", match.get("flag"))
        return {
            "found": True,
            "source": "CSV Dataset",
            "customer_id": str(match.get("customer_id", "")),
            "meter_id": str(match.get("meter_id", "N/A")),
            "readings": readings,
            "flag": int(flag_val) if pd.notna(flag_val) else None,
            "meter_type": str(match.get("meter_type", "Standard Smart Meter")),
            "region": str(match.get("region", "Unknown")),
            "tariff_category": str(match.get("tariff_category", "Residential")),
            "installation_date": str(match.get("installation_date", ""))
        }
    return None


def fetch_customer_reading(
    customer_id: Optional[str] = None,
    meter_id: Optional[str] = None
) -> Dict[str, Any]:
    """Retrieve customer 26-day readings by Customer ID and/or Meter ID."""
    customer_id = customer_id.strip() if customer_id else None
    meter_id = meter_id.strip() if meter_id else None

    if not customer_id and not meter_id:
        return {"found": False, "error": "Please provide either a Customer ID or a Meter ID."}

    res = fetch_from_postgres(customer_id, meter_id)
    if res:
        return res

    res = fetch_from_csv(customer_id, meter_id)
    if res:
        return res

    return {"found": False, "error": f"No customer reading found for Customer ID='{customer_id or ''}' and Meter ID='{meter_id or ''}'."}


def get_sample_customers(limit: int = 10) -> List[Dict[str, Any]]:
    """Return sample accounts from the dataset."""
    df = get_dataset_dataframe()
    if df is None or df.empty:
        return []

    theft_df = df[df.get("FLAG", 0) == 1].head(limit // 2) if "FLAG" in df.columns else pd.DataFrame()
    normal_df = df[df.get("FLAG", 0) == 0].head(limit - len(theft_df)) if "FLAG" in df.columns else df.head(limit)
    samples_df = pd.concat([theft_df, normal_df]).drop_duplicates()

    return [
        {
            "customer_id": str(row.get("customer_id", "")),
            "meter_id": str(row.get("meter_id", "N/A")),
            "flag": int(row.get("FLAG", 0)) if pd.notna(row.get("FLAG")) else 0,
            "tariff_category": str(row.get("tariff_category", "Residential")),
            "region": str(row.get("region", "Unknown"))
        }
        for _, row in samples_df.iterrows()
    ]