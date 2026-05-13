import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from mlxtend.frequent_patterns import (
    apriori,
    association_rules
)

# =====================================================
# PATHS
# =====================================================

INPUT_PATH = "../data/features/basket_features.csv"

OUTPUT_DIR = "../outputs/apriori"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =====================================================
# LOAD DATA
# =====================================================

print("=" * 60)
print("APRIORI ALGORITHM — MARKET BASKET ANALYSIS")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print("\nDataset Loaded Successfully")
print(df.head())

# =====================================================
# CREATE BASKET MATRIX
# =====================================================

print("\nCreating Basket Matrix...")

basket = (
    df.groupby([
        "transaction_id",
        "product_name"
    ])["product_name"]

    .count()

    .unstack()

    .fillna(0)
)

# Convert quantities into binary values
basket = (basket > 0)

print("✓ Basket Matrix Created")

# =====================================================
# APPLY APRIORI
# =====================================================

print("\nApplying Apriori Algorithm...")

frequent_itemsets = apriori(
    basket,
    min_support=0.005,
    use_colnames=True
)

print("✓ Frequent Itemsets Generated")

# =====================================================
# GENERATE ASSOCIATION RULES
# =====================================================

print("\nGenerating Association Rules...")

rules = association_rules(
    frequent_itemsets,
    metric="confidence",
    min_threshold=0.1
)

# Filter strong rules
# rules = rules[
#     rules["lift"] > 1.0
# ]

print("✓ Association Rules Generated")

# =====================================================
# SORT RULES
# =====================================================

rules = rules.sort_values(
    by="confidence",
    ascending=False
)

# =====================================================
# DISPLAY TOP RULES
# =====================================================

print("\nTOP ASSOCIATION RULES\n")

top_rules = rules[[
    "antecedents",
    "consequents",
    "support",
    "confidence",
    "lift"
]].head(10)

print(top_rules)

# =====================================================
# SAVE RULES
# =====================================================

rules_path = f"{OUTPUT_DIR}/association_rules.csv"

rules.to_csv(rules_path, index=False)

print(f"\n✓ Rules saved:")
print(rules_path)

# =====================================================
# VISUALIZATION — SUPPORT VS CONFIDENCE
# =====================================================

print("\nGenerating Association Rule Plot...")

plt.figure(figsize=(10, 6))

sns.scatterplot(
    data=rules,
    x="support",
    y="confidence",
    size="lift",
    hue="lift",
    sizes=(50, 300)
)

plt.title(
    "Association Rules — Support vs Confidence",
    fontsize=14,
    fontweight="bold"
)

plt.xlabel("Support")
plt.ylabel("Confidence")

plot_path = f"{OUTPUT_DIR}/association_rules_plot.png"

plt.savefig(plot_path, dpi=300, bbox_inches="tight")

plt.show()

print(f"✓ Plot saved:")
print(plot_path)

# =====================================================
# HEATMAP
# =====================================================

print("\nGenerating Heatmap...")

if len(rules) > 0:

    heatmap_data = rules.pivot_table(
        index="confidence",
        columns="support",
        values="lift"
    )

    plt.figure(figsize=(10, 6))

    sns.heatmap(
        heatmap_data,
        cmap="YlGnBu"
    )

    plt.title(
        "Association Rules Heatmap",
        fontsize=14,
        fontweight="bold"
    )

    heatmap_path = (
        f"{OUTPUT_DIR}/association_heatmap.png"
    )

    plt.savefig(
        heatmap_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print(f"✓ Heatmap saved:")
    print(heatmap_path)

else:

    print(
        "No association rules available "
        "for heatmap generation."
    )

# =====================================================
# BUSINESS INSIGHTS
# =====================================================

print("\n" + "=" * 60)
print("BUSINESS INSIGHTS")
print("=" * 60)

if len(rules) > 0:

    best_rule = rules.iloc[0]

    print(
        f"\nCustomers who buy "
        f"{list(best_rule['antecedents'])}"
    )

    print(
        f"are also likely to buy "
        f"{list(best_rule['consequents'])}"
    )

    print(
        f"\nConfidence: "
        f"{best_rule['confidence']:.2f}"
    )

    print(
        f"Lift Score: "
        f"{best_rule['lift']:.2f}"
    )

else:

    print(
        "\nNo strong association rules found."
    )

print(
    "\nMarket basket analysis helps management "
    "improve product placement and promotional "
    "bundling strategies."
)

print("\n✓ Apriori Analysis Completed Successfully")
print("=" * 60)