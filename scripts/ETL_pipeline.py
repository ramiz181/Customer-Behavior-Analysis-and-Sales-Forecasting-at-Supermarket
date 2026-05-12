"""
========================================================
  Naheed Supermarket — ETL Pipeline
  Project: Customer Behavior Analysis & Sales Forecasting
========================================================

PURPOSE:
    Extract CSV files
    Transform & validate data
    Load into SQLite Data Warehouse

OUTPUT:
    naheed_warehouse.db

TABLES:
    - customers_dim
    - products_dim
    - store_dim
    - time_dim
    - sales_fact

HOW TO RUN:
    python etl_pipeline.py

LIBRARIES REQUIRED:
    pip install pandas
"""

import pandas as pd
import sqlite3
import os

# =====================================================
# PATH CONFIGURATION
# =====================================================

DATA_DIR = "../data/"
WAREHOUSE_DIR = "warehouse"

os.makedirs(WAREHOUSE_DIR, exist_ok=True)

DB_PATH = f"{WAREHOUSE_DIR}/naheed_warehouse.db"

print("=" * 60)
print(" Naheed Supermarket — ETL Pipeline Started")
print("=" * 60)

# =====================================================
# STEP 1 — EXTRACT CSV FILES
# =====================================================

print("\n[1/5] Extracting CSV files...")

customers_df = pd.read_csv(f"{DATA_DIR}/customers_dim.csv")
products_df = pd.read_csv(f"{DATA_DIR}/products_dim.csv")
stores_df = pd.read_csv(f"{DATA_DIR}/store_dim.csv")
time_df = pd.read_csv(f"{DATA_DIR}/time_dim.csv")
sales_df = pd.read_csv(f"{DATA_DIR}/sales_fact.csv")

print("    ✓ CSV files loaded successfully")

# =====================================================
# STEP 2 — TRANSFORM DATA
# =====================================================

print("\n[2/5] Transforming and validating data...")

# -------------------------
# Remove duplicates
# -------------------------

customers_df.drop_duplicates(inplace=True)
products_df.drop_duplicates(inplace=True)
stores_df.drop_duplicates(inplace=True)
time_df.drop_duplicates(inplace=True)
sales_df.drop_duplicates(inplace=True)

# -------------------------
# Handle missing values
# -------------------------

customers_df.fillna("Unknown", inplace=True)
products_df.fillna(0, inplace=True)
stores_df.fillna("Unknown", inplace=True)
time_df.fillna("", inplace=True)
sales_df.fillna(0, inplace=True)

# -------------------------
# Data type conversions
# -------------------------

sales_df["quantity"] = sales_df["quantity"].astype(int)

sales_df["unit_price"] = sales_df["unit_price"].astype(float)

sales_df["total_amount"] = sales_df["total_amount"].astype(float)

sales_df["discount"] = sales_df["discount"].astype(float)

products_df["cost_price"] = products_df["cost_price"].astype(float)

products_df["selling_price"] = products_df["selling_price"].astype(float)

time_df["date"] = pd.to_datetime(time_df["date"])

# -------------------------
# Validate calculations
# -------------------------

sales_df["calculated_total"] = (
    sales_df["quantity"] *
    sales_df["unit_price"] *
    (1 - sales_df["discount"])
).round(2)

# Compare stored total vs calculated total
mismatch_count = (
    sales_df["calculated_total"] !=
    sales_df["total_amount"]
).sum()

print(f"    ✓ Validation completed")
print(f"    ✓ Total mismatches found: {mismatch_count}")

# Drop helper column
sales_df.drop(columns=["calculated_total"], inplace=True)

# =====================================================
# STEP 3 — CREATE SQLITE CONNECTION
# =====================================================

print("\n[3/5] Creating SQLite Data Warehouse...")

conn = sqlite3.connect(DB_PATH)

cursor = conn.cursor()

# Enable foreign keys
cursor.execute("PRAGMA foreign_keys = ON;")

print(f"    ✓ Database created: {DB_PATH}")

# =====================================================
# STEP 4 — CREATE STAR SCHEMA TABLES
# =====================================================

print("\n[4/5] Creating Star Schema tables...")

# -----------------------------------------------------
# DROP TABLES IF EXIST
# -----------------------------------------------------

cursor.executescript("""

DROP TABLE IF EXISTS sales_fact;
DROP TABLE IF EXISTS customers_dim;
DROP TABLE IF EXISTS products_dim;
DROP TABLE IF EXISTS store_dim;
DROP TABLE IF EXISTS time_dim;

""")

# -----------------------------------------------------
# CUSTOMERS DIMENSION
# -----------------------------------------------------

cursor.execute("""

CREATE TABLE customers_dim (

    customer_id TEXT PRIMARY KEY,
    name TEXT,
    gender TEXT,
    age INTEGER,
    city TEXT,
    area TEXT,
    loyalty_tier TEXT

);

""")

# -----------------------------------------------------
# PRODUCTS DIMENSION
# -----------------------------------------------------

cursor.execute("""

CREATE TABLE products_dim (

    product_id TEXT PRIMARY KEY,
    product_name TEXT,
    category TEXT,
    sub_category TEXT,
    brand TEXT,
    cost_price REAL,
    selling_price REAL

);

""")

# -----------------------------------------------------
# STORE DIMENSION
# -----------------------------------------------------

cursor.execute("""

CREATE TABLE store_dim (

    store_id TEXT PRIMARY KEY,
    store_name TEXT,
    branch TEXT,
    area TEXT,
    city TEXT

);

""")

# -----------------------------------------------------
# TIME DIMENSION
# -----------------------------------------------------

cursor.execute("""

CREATE TABLE time_dim (

    date_id TEXT PRIMARY KEY,
    date TEXT,
    day INTEGER,
    day_name TEXT,
    month INTEGER,
    month_name TEXT,
    quarter INTEGER,
    year INTEGER,
    is_weekend INTEGER,
    is_holiday INTEGER,
    holiday_name TEXT

);

""")

# -----------------------------------------------------
# SALES FACT TABLE
# -----------------------------------------------------

cursor.execute("""

CREATE TABLE sales_fact (

    transaction_id TEXT PRIMARY KEY,

    customer_id TEXT,
    product_id TEXT,
    store_id TEXT,
    date_id TEXT,

    quantity INTEGER,
    unit_price REAL,
    total_amount REAL,
    discount REAL,

    FOREIGN KEY (customer_id)
        REFERENCES customers_dim(customer_id),

    FOREIGN KEY (product_id)
        REFERENCES products_dim(product_id),

    FOREIGN KEY (store_id)
        REFERENCES store_dim(store_id),

    FOREIGN KEY (date_id)
        REFERENCES time_dim(date_id)

);

""")

print("    ✓ Star Schema tables created")

# =====================================================
# STEP 5 — LOAD DATA INTO DATABASE
# =====================================================

print("\n[5/5] Loading data into warehouse...")

customers_df.to_sql(
    "customers_dim",
    conn,
    if_exists="append",
    index=False
)

products_df.to_sql(
    "products_dim",
    conn,
    if_exists="append",
    index=False
)

stores_df.to_sql(
    "store_dim",
    conn,
    if_exists="append",
    index=False
)

time_df.to_sql(
    "time_dim",
    conn,
    if_exists="append",
    index=False
)

sales_df.to_sql(
    "sales_fact",
    conn,
    if_exists="append",
    index=False
)

print("    ✓ All data loaded successfully")

# =====================================================
# CREATE INDEXES FOR PERFORMANCE
# =====================================================

print("\nCreating indexes for faster queries...")

cursor.executescript("""

CREATE INDEX idx_sales_customer
ON sales_fact(customer_id);

CREATE INDEX idx_sales_product
ON sales_fact(product_id);

CREATE INDEX idx_sales_store
ON sales_fact(store_id);

CREATE INDEX idx_sales_date
ON sales_fact(date_id);

""")

print("    ✓ Indexes created")

# =====================================================
# VERIFY RECORD COUNTS
# =====================================================

print("\nVerifying warehouse tables...\n")

tables = [
    "customers_dim",
    "products_dim",
    "store_dim",
    "time_dim",
    "sales_fact"
]

for table in tables:

    query = f"SELECT COUNT(*) FROM {table}"

    count = pd.read_sql(query, conn).iloc[0, 0]

    print(f"{table:<20} : {count:,} rows")

# =====================================================
# SAMPLE ANALYTICAL QUERY
# =====================================================

print("\nRunning sample analytical query...\n")

query = """

SELECT
    p.category,
    ROUND(SUM(s.total_amount), 2) AS revenue

FROM sales_fact s

JOIN products_dim p
ON s.product_id = p.product_id

GROUP BY p.category

ORDER BY revenue DESC

LIMIT 10;

"""

result = pd.read_sql(query, conn)

print(result)

# =====================================================
# CLOSE CONNECTION
# =====================================================

conn.commit()
conn.close()

# =====================================================
# COMPLETED
# =====================================================

print("\n" + "=" * 60)
print(" ✅ ETL PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)

print(f"\n📦 Data Warehouse Created:")
print(f"   {DB_PATH}")

print("\n⭐ Your Star Schema Warehouse is Ready!")
print("\nNext Steps:")
print("   1. Run analytical SQL queries")
print("   2. Perform K-Means clustering")
print("   3. Build sales forecasting model")
print("   4. Apply Apriori association rules")
print("   5. Create Power BI dashboards")

print("\n" + "=" * 60)