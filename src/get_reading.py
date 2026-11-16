#!/usr/bin/env python3
import sys
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Sql.fetch_data import fetch_customer_reading, get_sample_customers


def display_customer_readings(data: dict):
    if not data.get("found"):
        print("\n❌ Error:", data.get("error", "Customer reading not found."))
        return

    print("=" * 65)
    print(" ⚡ GRIDGUARD SMART METER READING RECORD")
    print("=" * 65)
    print(f" Source:              {data.get('source', 'Unknown')}")
    print(f" Customer ID:         {data.get('customer_id')}")
    print(f" Meter Serial ID:     {data.get('meter_id')}")
    print(f" Tariff Category:     {data.get('tariff_category')}")
    print(f" Region / Grid Node:  {data.get('region')}")
    flag_str = "🚨 Confirmed Theft (1)" if data.get('flag') == 1 else ("🟢 Normal (0)" if data.get('flag') == 0 else "Unlabelled")
    print(f" Historical Record:   {flag_str}")
    print("-" * 65)

    readings = data.get("readings", [])
    total_kwh = sum(readings)
    avg_kwh = total_kwh / len(readings) if readings else 0.0
    zero_days = sum(1 for x in readings if x <= 0)

    print(f" Summary: 26-Day Total = {total_kwh:.2f} kWh | Daily Average = {avg_kwh:.2f} kWh | Zero Days = {zero_days}")
    print("-" * 65)
    print(" Daily Consumption (kWh):")
    for row_idx in range(13):
        d1, d2 = row_idx + 1, row_idx + 14
        val1 = readings[row_idx] if row_idx < len(readings) else 0.0
        val2 = readings[row_idx + 13] if (row_idx + 13) < len(readings) else 0.0
        print(f"   Day {d1:2d}: {val1:6.2f} kWh    |    Day {d2:2d}: {val2:6.2f} kWh")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="Query 26-day smart meter readings by Customer ID or Meter ID.")
    parser.add_argument("--customer-id", "-c", type=str, help="Customer ID")
    parser.add_argument("--meter-id", "-m", type=str, help="Meter ID")
    parser.add_argument("--samples", "-s", action="store_true", help="List sample customers")
    args = parser.parse_args()

    if args.samples:
        samples = get_sample_customers(limit=10)
        print("\n--- Verified Utility Accounts in Database ---")
        for i, s in enumerate(samples, 1):
            tag = "Theft Risk" if s['flag'] == 1 else "Normal Profile"
            print(f" [{i:2d}] Customer ID: {s['customer_id']} | Meter ID: {s['meter_id']} | Type: {tag}")
        return

    result = fetch_customer_reading(customer_id=args.customer_id, meter_id=args.meter_id)
    display_customer_readings(result)


if __name__ == "__main__":
    main()