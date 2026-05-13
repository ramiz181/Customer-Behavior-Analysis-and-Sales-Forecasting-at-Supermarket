# generate_basket_features.py

import pandas as pd
import random
import os

# =====================================================
# OUTPUT PATH
# =====================================================

OUTPUT_PATH = "../data/features/basket_features.csv"

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True
)

# =====================================================
# PRODUCT LIST
# =====================================================

products = [
    "Guard Rice (1kg)",
    "Dalda Cooking Oil (500ml)",
    "Tapal Tea (250g)",
    "Olpers Milk (Pack of 6)",
    "National Ketchup (1kg)",
    "Thal Sugar (200ml)",
    "Lays Chips",
    "Pepsi 1L",
    "Nestle Water",
    "Shan Biryani Masala",
    "Surf Excel",
    "Lux Soap",
    "Bread",
    "Eggs",
    "Butter",
    "Jam",
    "Biscuits",
    "Noodles",
    "Chicken",
    "Frozen Paratha"
]

# =====================================================
# GENERATE TRANSACTIONS
# =====================================================

transactions = []

NUM_TRANSACTIONS = 5000

for transaction_id in range(1, NUM_TRANSACTIONS + 1):

    # Each customer buys 2–6 products
    num_products = random.randint(2, 6)

    # Randomly select products
    selected_products = random.sample(
        products,
        num_products
    )

    # Save each product separately
    for product in selected_products:

        transactions.append([
            f"T{transaction_id:06}",
            product
        ])

# =====================================================
# CREATE DATAFRAME
# =====================================================

df = pd.DataFrame(
    transactions,
    columns=[
        "transaction_id",
        "product_name"
    ]
)

# =====================================================
# SAVE CSV
# =====================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("=" * 60)
print("BASKET FEATURE DATA GENERATED")
print("=" * 60)

print(f"\nTotal Rows: {len(df)}")

print(
    f"Unique Transactions: "
    f"{df['transaction_id'].nunique()}"
)

print("\nSample Data:")
print(df.head(10))

print(f"\nSaved to:")
print(OUTPUT_PATH)