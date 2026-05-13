import pandas as pd
import sqlite3
import os

# ============================================
# PATHS
# ============================================

DB_PATH = "../warehouse/naheed_warehouse.db"

OUTPUT_DIR = "../data/features"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================
# CONNECT TO DATABASE
# ============================================

print("=" * 60)
print("PREPARING FEATURE DATASETS")
print("=" * 60)

conn = sqlite3.connect(DB_PATH)

# ============================================
# 1. CUSTOMER FEATURES
# ============================================

print("\n[1/3] Creating customer features dataset...")

customer_query = """

SELECT
    s.customer_id,

    ROUND(SUM(s.total_amount), 2) AS total_spent,

    COUNT(s.transaction_id) AS purchase_frequency,

    ROUND(AVG(s.quantity), 2) AS avg_basket_size,

    c.age,
    c.gender,
    c.loyalty_tier

FROM sales_fact s

JOIN customers_dim c
ON s.customer_id = c.customer_id

GROUP BY s.customer_id

"""

customer_df = pd.read_sql(customer_query, conn)

customer_df.to_csv(
    f"{OUTPUT_DIR}/customer_features.csv",
    index=False
)

print("✓ customer_features.csv created")

# ============================================
# 2. MONTHLY SALES FEATURES
# ============================================

print("\n[2/3] Creating monthly sales dataset...")

monthly_query = """

SELECT

    t.month,
    t.month_name,
    t.quarter,

    ROUND(SUM(s.total_amount), 2) AS total_sales,

    COUNT(s.transaction_id) AS total_orders,

    ROUND(AVG(s.total_amount), 2) AS avg_order_value

FROM sales_fact s

JOIN time_dim t
ON s.date_id = t.date_id

GROUP BY t.month, t.month_name, t.quarter

ORDER BY t.month

"""

monthly_df = pd.read_sql(monthly_query, conn)

monthly_df.to_csv(
    f"{OUTPUT_DIR}/monthly_sales_features.csv",
    index=False
)

print("✓ monthly_sales_features.csv created")

# ============================================
# 3. BASKET FEATURES (FOR APRIORI)
# ============================================

print("\n[3/3] Creating basket dataset...")

basket_query = """

SELECT

    s.transaction_id,
    p.product_name

FROM sales_fact s

JOIN products_dim p
ON s.product_id = p.product_id

ORDER BY s.transaction_id

"""

basket_df = pd.read_sql(basket_query, conn)

basket_df.to_csv(
    f"{OUTPUT_DIR}/basket_features.csv",
    index=False
)

print("✓ basket_features.csv created")

# ============================================
# CLOSE CONNECTION
# ============================================

conn.close()

print("\n" + "=" * 60)
print("FEATURE DATASETS CREATED SUCCESSFULLY")
print("=" * 60)

print("\nOutput Location:")
print(OUTPUT_DIR)