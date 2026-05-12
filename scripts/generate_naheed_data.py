"""
========================================================
  Naheed Supermarket — Simulated Dataset Generator
  SINGLE STORE VERSION
  Project: Customer Behavior Analysis & Sales Forecasting
  Data Range: Jan 2023 – Dec 2023 | ~10,000 Transactions
========================================================

LIBRARIES REQUIRED:
    pip install faker pandas numpy

HOW TO RUN:
    python generate_naheed_data.py

OUTPUT:
    5 CSV files inside naheed_data/
    - customers_dim.csv
    - products_dim.csv
    - store_dim.csv
    - time_dim.csv
    - sales_fact.csv
"""

import pandas as pd
import numpy as np
from faker import Faker
import random
from datetime import date, timedelta
import os

# =====================================================
# SETUP
# =====================================================

fake = Faker('en_US')

Faker.seed(42)
np.random.seed(42)
random.seed(42)

OUTPUT_DIR = "naheed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print(" Naheed Supermarket — Single Store Data Generation")
print("=" * 60)

# =====================================================
# TABLE 1: customers_dim.csv
# =====================================================

print("\n[1/5] Generating customers_dim.csv ...")

NUM_CUSTOMERS = 500

male_first = [
    "Ahmed", "Ali", "Usman", "Hassan", "Omar",
    "Bilal", "Zain", "Faisal", "Tariq", "Imran",
    "Junaid", "Kamran", "Naeem", "Salman", "Asad",
    "Raza", "Hamza", "Fahad", "Saad", "Waseem"
]

female_first = [
    "Fatima", "Aisha", "Zara", "Hina", "Sana",
    "Nadia", "Mehwish", "Rabia", "Amna", "Sara",
    "Mariam", "Bushra", "Uzma", "Saima", "Noor",
    "Iqra", "Maryam", "Ayesha", "Sobia", "Aroha"
]

last_names = [
    "Khan", "Ahmed", "Ali", "Sheikh", "Siddiqui",
    "Ansari", "Qureshi", "Butt", "Malik", "Hussain",
    "Mirza", "Baig", "Chaudhry", "Raza", "Hashmi",
    "Rizvi", "Farooq", "Iqbal"
]

karachi_areas = [
    "Gulshan-e-Iqbal",
    "North Nazimabad",
    "Clifton",
    "Defence (DHA)",
    "Saddar",
    "Nazimabad",
    "Malir",
    "Korangi",
    "Federal B Area",
    "Bahadurabad",
    "Gulistan-e-Johar",
    "PECHS",
    "Tariq Road",
    "Lyari"
]

loyalty_tiers = ["Bronze", "Silver", "Gold"]
loyalty_weights = [0.60, 0.30, 0.10]

customers = []

for i in range(1, NUM_CUSTOMERS + 1):

    gender = random.choice(["Male", "Female"])

    first = random.choice(
        male_first if gender == "Male" else female_first
    )

    last = random.choice(last_names)

    age = int(np.random.normal(loc=35, scale=10))
    age = max(18, min(70, age))

    customers.append({
        "customer_id": f"C{i:04d}",
        "name": f"{first} {last}",
        "gender": gender,
        "age": age,
        "city": "Karachi",
        "area": random.choice(karachi_areas),
        "loyalty_tier": random.choices(
            loyalty_tiers,
            loyalty_weights
        )[0]
    })

customers_df = pd.DataFrame(customers)

customers_df.to_csv(
    f"{OUTPUT_DIR}/customers_dim.csv",
    index=False
)

print(f"    ✓ {len(customers_df)} customers saved.")

# =====================================================
# TABLE 2: products_dim.csv
# =====================================================

print("\n[2/5] Generating products_dim.csv ...")

PRODUCT_CATALOG = [

    ("Beverages", "Juices", "Nestle", 45, 80),
    ("Beverages", "Juices", "Shezan", 40, 70),
    ("Beverages", "Soft Drinks", "Coca-Cola", 50, 85),
    ("Beverages", "Soft Drinks", "Pepsi", 48, 82),
    ("Beverages", "Water", "Aquafina", 20, 35),
    ("Beverages", "Tea", "Lipton", 85, 140),
    ("Beverages", "Milk", "Olpers", 60, 100),

    ("Dairy", "Yogurt", "Nestle", 40, 68),
    ("Dairy", "Butter", "Nurpur", 90, 155),
    ("Dairy", "Cheese", "Adams", 120, 200),

    ("Bakery", "Bread", "Bake Parlour", 55, 90),
    ("Bakery", "Biscuits", "LU", 30, 55),
    ("Bakery", "Cakes", "Bake Parlour", 180, 300),

    ("Snacks", "Chips", "Lays", 45, 80),
    ("Snacks", "Chocolate", "Cadbury", 80, 140),

    ("Staples", "Rice", "Guard", 120, 200),
    ("Staples", "Flour (Atta)", "Sunridge", 90, 150),
    ("Staples", "Sugar", "Thal", 65, 105),
    ("Staples", "Cooking Oil", "Dalda", 180, 300),

    ("Condiments", "Spices", "National", 60, 100),
    ("Condiments", "Ketchup", "National", 70, 120),

    ("Frozen", "Frozen Nuggets", "Menu", 100, 170),
    ("Frozen", "Ice Cream", "Walls", 120, 200),

    ("Personal Care", "Shampoo", "Pantene", 190, 325),
    ("Personal Care", "Soap", "Lux", 55, 90),
    ("Personal Care", "Toothpaste", "Colgate", 95, 160),

    ("Household", "Detergent", "Surf Excel", 150, 255),
    ("Household", "Dishwash", "Vim", 80, 135),

    ("Baby", "Diapers", "Pampers", 700, 1150),

    ("Stationery", "Notebooks", "Sinar", 50, 85),

    ("Produce", "Fruits", "Local Farm", 20, 40),
    ("Produce", "Vegetables", "Local Farm", 15, 30),
]

products = []

product_id_counter = 1

sizes = [
    "200ml",
    "500ml",
    "1L",
    "250g",
    "500g",
    "1kg",
    "Pack of 6",
    "Family Pack"
]

for cat, sub_cat, brand, cost, sell in PRODUCT_CATALOG:

    num_skus = random.randint(4, 7)

    for _ in range(num_skus):

        size = random.choice(sizes)

        products.append({
            "product_id": f"P{product_id_counter:03d}",
            "product_name": f"{brand} {sub_cat} ({size})",
            "category": cat,
            "sub_category": sub_cat,
            "brand": brand,
            "cost_price": round(
                cost * random.uniform(0.90, 1.10), 2
            ),
            "selling_price": round(
                sell * random.uniform(0.90, 1.10), 2
            )
        })

        product_id_counter += 1

products_df = pd.DataFrame(products)

products_df = products_df.head(200).copy()

products_df["product_id"] = [
    f"P{i+1:03d}"
    for i in range(len(products_df))
]

products_df.to_csv(
    f"{OUTPUT_DIR}/products_dim.csv",
    index=False
)

print(f"    ✓ {len(products_df)} products saved.")

# =====================================================
# TABLE 3: store_dim.csv
# SINGLE STORE ONLY
# =====================================================

print("\n[3/5] Generating store_dim.csv ...")

stores = [
    {
        "store_id": "S001",
        "store_name": "Naheed Supermarket",
        "branch": "Main Branch",
        "area": "Tariq Road",
        "city": "Karachi"
    }
]

stores_df = pd.DataFrame(stores)

stores_df.to_csv(
    f"{OUTPUT_DIR}/store_dim.csv",
    index=False
)

print(f"    ✓ {len(stores_df)} store saved.")

# =====================================================
# TABLE 4: time_dim.csv
# =====================================================

print("\n[4/5] Generating time_dim.csv ...")

HOLIDAYS_2023 = {
    date(2023, 2, 5): "Kashmir Day",
    date(2023, 3, 23): "Pakistan Day",
    date(2023, 4, 21): "Eid ul-Fitr",
    date(2023, 6, 28): "Eid ul-Adha",
    date(2023, 8, 14): "Independence Day",
    date(2023, 12, 25): "Quaid Day",
}

time_records = []

start_date = date(2023, 1, 1)
end_date = date(2023, 12, 31)

current = start_date
date_id = 1

while current <= end_date:

    time_records.append({
        "date_id": f"D{date_id:03d}",
        "date": current.strftime("%Y-%m-%d"),
        "day": current.day,
        "day_name": current.strftime("%A"),
        "month": current.month,
        "month_name": current.strftime("%B"),
        "quarter": (current.month - 1) // 3 + 1,
        "year": current.year,
        "is_weekend": 1 if current.weekday() >= 4 else 0,
        "is_holiday": 1 if current in HOLIDAYS_2023 else 0,
        "holiday_name": HOLIDAYS_2023.get(current, "")
    })

    current += timedelta(days=1)
    date_id += 1

time_df = pd.DataFrame(time_records)

time_df.to_csv(
    f"{OUTPUT_DIR}/time_dim.csv",
    index=False
)

print(f"    ✓ {len(time_df)} date records saved.")

# =====================================================
# TABLE 5: sales_fact.csv
# =====================================================

print("\n[5/5] Generating sales_fact.csv ...")

TOTAL_TRANSACTIONS = 10_000

CUSTOMER_TIER = customers_df.set_index(
    "customer_id"
)["loyalty_tier"].to_dict()

DISCOUNT_RANGE = {
    "Bronze": (0.00, 0.05),
    "Silver": (0.05, 0.12),
    "Gold": (0.10, 0.20)
}

sales = []

for tx_id in range(1, TOTAL_TRANSACTIONS + 1):

    customer_id = random.choice(
        customers_df["customer_id"].tolist()
    )

    product_row = products_df.sample(1).iloc[0]

    time_row = time_df.sample(1).iloc[0]

    quantity = np.random.choice(
        [1, 2, 3, 4, 5],
        p=[0.40, 0.30, 0.15, 0.10, 0.05]
    )

    unit_price = round(
        product_row["selling_price"] *
        random.uniform(0.97, 1.03),
        2
    )

    loyalty = CUSTOMER_TIER[customer_id]

    discount_min, discount_max = DISCOUNT_RANGE[loyalty]

    discount = round(
        random.uniform(discount_min, discount_max),
        3
    )

    total_amount = round(
        quantity * unit_price * (1 - discount),
        2
    )

    sales.append({
        "transaction_id": f"T{tx_id:06d}",
        "customer_id": customer_id,
        "product_id": product_row["product_id"],
        "store_id": "S001",
        "date_id": time_row["date_id"],
        "quantity": quantity,
        "unit_price": unit_price,
        "total_amount": total_amount,
        "discount": discount
    })

sales_df = pd.DataFrame(sales)

sales_df.to_csv(
    f"{OUTPUT_DIR}/sales_fact.csv",
    index=False
)

print(f"    ✓ {len(sales_df)} transactions saved.")

# =====================================================
# SUMMARY
# =====================================================

print("\n" + "=" * 60)
print(" ✅ ALL FILES GENERATED SUCCESSFULLY")
print("=" * 60)

print(f"\n📁 Output Folder : ./{OUTPUT_DIR}/")
print(f"👥 Customers     : {len(customers_df)}")
print(f"📦 Products      : {len(products_df)}")
print(f"🏪 Stores        : {len(stores_df)}")
print(f"📅 Dates         : {len(time_df)}")
print(f"🧾 Transactions  : {len(sales_df)}")

print(
    f"\n💰 Total Revenue : PKR "
    f"{sales_df['total_amount'].sum():,.0f}"
)

print(
    f"📊 Avg Order Value : PKR "
    f"{sales_df['total_amount'].mean():.2f}"
)

print("\n" + "=" * 60)