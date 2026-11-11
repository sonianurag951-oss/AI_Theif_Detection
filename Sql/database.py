
import os
from pathlib import Path
import psycopg
from dotenv import load_dotenv

# Project root
PROJECT_ROOT = Path(r"D:\deep learning\AI_Theif_Detection")
# Load environment variables
ENV_FILE = PROJECT_ROOT / ".env"
load_dotenv(ENV_FILE)

# Required credentials
required = [
    "PGDATABASE",
    "PGUSER",
    "PGPASSWORD"
]

missing = [
    key for key in required
    if not os.getenv(key)
]

if missing:
    raise RuntimeError(
        f"Missing settings in .env: {', '.join(missing)}"
    )


def connect_database():

    try:
      with psycopg.connect(
     host=os.getenv("PGHOST", "localhost"),
     port=int(os.getenv("PGPORT", "2004")),
     dbname=os.environ["PGDATABASE"],
     user=os.environ["PGUSER"],
     password=os.environ["PGPASSWORD"],
     connect_timeout=5
     ) as conn:

            with conn.cursor() as cur:
                # Read-only transaction
                cur.execute("SET TRANSACTION READ ONLY")
                print("\nPostgreSQL connected successfully!")
                # Database information
                cur.execute("SELECT current_database()")
                print("Database:", cur.fetchone()[0])
                # Check table existence
                cur.execute("""
                    SELECT EXISTS (
                        SELECT 1
                        FROM information_schema.tables
                        WHERE table_schema = 'public'
                        AND table_name = 'gridguard_data'
                    )
                """)

                table_exists = cur.fetchone()[0]

                if not table_exists:
                    print("gridguard_data table not found!")
                    return


                for row in cur.fetchall():
                    print(row)

                for row in cur.fetchall():
                    print(row)

                # Get table columns
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_schema = 'public'
                    AND table_name = 'gridguard_data'
                    ORDER BY ordinal_position
                """)

                print("\nTable Structure:")

                for column in cur.fetchall():
                    print(column)
                print("\nGridGuard database test completed!")

    except psycopg.Error as error:
        print("\nPostgreSQL Error:", error)

if __name__ == "__main__":
    connect_database()
